"""
MCP 适配器 - 模型上下文协议（Model Context Protocol）集成

将外部 MCP 服务器提供的工具动态注册到 ActionRoom 工具注册中心，
支持 stdio、SSE、HTTP 三种传输方式。实现了 JSON-RPC 2.0 协议、
自动重连、速率限制和健康检查等机制。

核心组件：
- MCPProtocolHandler: JSON-RPC 2.0 协议处理器
- MCPServer: 单个 MCP 服务器连接管理
- MCPAdapter: MCP 适配器主类（多服务器管理）
- MCPServerConfig / MCPTool: 配置和数据模型
"""

import asyncio
import hashlib
import httpx
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urljoin

from nan_agent.action_room.registry import Tool
from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

MCP_HANDSHAKE_TIMEOUT = 10.0
MCP_TOOL_CALL_TIMEOUT = 60.0
MCP_RESTART_DELAY = 2.0
MCP_MAX_RESTART_ATTEMPTS = 3

JSONRPC_VERSION = "2.0"

JSONRPC_ERROR_CODES = {
    -32700: "Parse error",
    -32600: "Invalid Request",
    -32601: "Method not found",
    -32602: "Invalid params",
    -32603: "Internal error",
    -32000: "Server error",
}

SUPPORTED_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPServerConfig:
    name: str
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    transport_type: str = "stdio"


@dataclass
class MCPTool:
    name: str
    description: str
    inputSchema: Dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


class MCPProtocolHandler:
    """JSON-RPC 2.0 协议处理器。

    负责构建和解析 MCP 协议消息（请求、响应、通知），
    管理待处理请求的 Future 映射和通知处理器注册。
    """

    def __init__(self):
        self._request_counter = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._notification_handlers: Dict[str, List[Callable]] = {}

    def create_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._request_counter += 1
        request_id = str(self._request_counter)
        message = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params
        return message

    def create_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        message = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
        }
        if params is not None:
            message["params"] = params
        return message

    def create_response(self, request_id: str, result: Any) -> Dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "result": result,
        }

    def create_error_response(self, request_id: str, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": error,
        }

    def is_request(self, message: Dict[str, Any]) -> bool:
        return "method" in message and "id" in message

    def is_notification(self, message: Dict[str, Any]) -> bool:
        return "method" in message and "id" not in message

    def is_response(self, message: Dict[str, Any]) -> bool:
        return "result" in message or "error" in message

    def validate_message(self, message: Dict[str, Any]) -> Optional[str]:
        if not isinstance(message, dict):
            return "Message must be a JSON object"
        if message.get("jsonrpc") != JSONRPC_VERSION:
            return f"Invalid jsonrpc version: {message.get('jsonrpc')}"
        return None

    def register_pending_request(self, request_id: str, future: asyncio.Future) -> None:
        self._pending_requests[request_id] = future

    def resolve_pending_request(self, request_id: str, result: Any) -> bool:
        future = self._pending_requests.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(result)
            return True
        return False

    def reject_pending_request(self, request_id: str, exception: Exception) -> bool:
        future = self._pending_requests.pop(request_id, None)
        if future is not None and not future.done():
            future.set_exception(exception)
            return True
        return False

    def on_notification(self, method: str, handler: Callable) -> None:
        if method not in self._notification_handlers:
            self._notification_handlers[method] = []
        self._notification_handlers[method].append(handler)

    async def handle_notification(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        handlers = self._notification_handlers.get(method, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(params)
                else:
                    handler(params)
            except Exception:
                logger.exception("notification_handler_error", method=method)

    def get_error_message(self, code: int) -> str:
        return JSONRPC_ERROR_CODES.get(code, "Unknown error")

    def clear_pending(self) -> None:
        for request_id, future in self._pending_requests.items():
            if not future.done():
                future.set_exception(ActionError(
                    f"Request {request_id} cancelled: connection closed",
                    error_code="E502",
                ))
        self._pending_requests.clear()


class MCPServer:
    """单个 MCP 服务器的连接和生命周期管理。

    负责建立/断开与 MCP 服务器的连接，执行 MCP 握手（initialize），
    列出可用工具，处理工具调用和健康检查。支持自动重连（最多 3 次）。

    Attributes:
        config: 服务器配置
        connected: 是否已连接
        tools: 服务器提供的工具字典（name -> MCPTool）
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._protocol = MCPProtocolHandler()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._connected = False
        self._server_capabilities: Dict[str, Any] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._restart_attempts = 0
        self._lock = asyncio.Lock()
        self._buffer = b""
        self._http_client: Optional[httpx.AsyncClient] = None
        self._sse_response: Optional[httpx.Response] = None
        self._sse_post_endpoint: Optional[str] = None
        self._http_endpoint: Optional[str] = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> Dict[str, MCPTool]:
        return self._tools.copy()

    async def start(self) -> None:
        """启动 MCP 服务器连接。

        根据配置的传输类型（stdio/sse/http）选择对应的连接方式，
        执行 MCP 握手（initialize 请求），获取服务器能力列表，
        并调用 tools/list 获取可用工具。

        Raises:
            ActionError: 连接失败或握手失败时抛出
        """
        async with self._lock:
            if self._connected:
                return
            await self._connect()

    async def stop(self) -> None:
        async with self._lock:
            await self._disconnect()

    async def _connect(self) -> None:
        transport = self.config.transport_type
        logger.info("mcp_server_connecting", name=self.config.name, transport=transport)

        if transport == "stdio":
            await self._connect_stdio()
        elif transport == "sse":
            await self._connect_sse()
        elif transport == "http":
            await self._connect_http()
        else:
            raise ActionError(
                f"Unsupported transport type: {transport}",
                error_code="E502",
                details={"transport_type": transport},
            )

        await self._initialize()

    async def _disconnect(self) -> None:
        self._connected = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        if self._process is not None:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError as e:
                logger.debug("mcp_process_already_exited", pid=self._process.pid if self._process else None, error=str(e))
            self._process = None
        await self._cleanup_http()
        self._protocol.clear_pending()
        self._tools.clear()
        self._buffer = b""
        logger.info("mcp_server_disconnected", name=self.config.name)

    async def _cleanup_http(self) -> None:
        """清理 HTTP 客户端和 SSE/HTTP 相关状态。"""
        if self._sse_response is not None:
            try:
                await self._sse_response.aclose()
            except Exception:
                pass
            self._sse_response = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
        self._sse_post_endpoint = None
        self._http_endpoint = None

    async def _connect_stdio(self) -> None:
        if not self.config.command:
            raise ActionError(
                "stdio transport requires a command",
                error_code="E502",
                details={"server": self.config.name},
            )

        env = {}
        env.update(self.config.env)
        env["PATH"] = env.get("PATH", "/usr/local/bin:/usr/bin:/bin")

        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._read_stdio_messages())
        self._connected = True
        logger.info("mcp_server_stdio_connected", name=self.config.name)

    async def _connect_sse(self) -> None:
        if not self.config.url:
            raise ActionError(
                "sse transport requires a url",
                error_code="E502",
                details={"server": self.config.name},
            )

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(MCP_TOOL_CALL_TIMEOUT, connect=MCP_HANDSHAKE_TIMEOUT),
        )

        # 打开 SSE 连接（GET 请求，长连接，无读取超时）
        try:
            request = self._http_client.build_request(
                "GET",
                self.config.url,
                timeout=httpx.Timeout(None, connect=MCP_HANDSHAKE_TIMEOUT),
            )
            self._sse_response = await self._http_client.send(request, stream=True)
            self._sse_response.raise_for_status()
        except httpx.HTTPStatusError as e:
            await self._cleanup_http()
            raise ActionError(
                f"SSE connection failed for server '{self.config.name}': HTTP {e.response.status_code}",
                error_code="E502",
                details={"server": self.config.name, "status_code": e.response.status_code},
            )
        except httpx.RequestError as e:
            await self._cleanup_http()
            raise ActionError(
                f"SSE connection failed for server '{self.config.name}': {e}",
                error_code="E502",
                details={"server": self.config.name, "error": str(e)},
            )

        # 读取 SSE 事件流，等待 endpoint 事件获取 POST URL
        try:
            endpoint_received = False
            async for event_type, event_data in self._iter_sse_events(self._sse_response):
                if event_type == "endpoint":
                    self._sse_post_endpoint = urljoin(self.config.url, event_data)
                    endpoint_received = True
                    break
                else:
                    try:
                        message = json.loads(event_data)
                        await self._handle_message(message)
                    except json.JSONDecodeError:
                        logger.warning(
                            "mcp_sse_invalid_json_before_endpoint",
                            name=self.config.name,
                            data=event_data[:200],
                        )

            if not endpoint_received:
                await self._cleanup_http()
                raise ActionError(
                    f"SSE stream ended without endpoint event from server '{self.config.name}'",
                    error_code="E502",
                    details={"server": self.config.name},
                )
        except ActionError:
            raise
        except Exception:
            await self._cleanup_http()
            raise ActionError(
                f"Failed to read SSE endpoint event from server '{self.config.name}'",
                error_code="E502",
                details={"server": self.config.name},
            )

        self._connected = True
        # 启动后台任务持续读取 SSE 事件
        self._reader_task = asyncio.create_task(self._read_sse_messages())
        logger.info("mcp_server_sse_connected", name=self.config.name, url=self.config.url)

    async def _connect_http(self) -> None:
        if not self.config.url:
            raise ActionError(
                "http transport requires a url",
                error_code="E502",
                details={"server": self.config.name},
            )

        self._http_endpoint = self.config.url
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(MCP_TOOL_CALL_TIMEOUT, connect=MCP_HANDSHAKE_TIMEOUT),
        )
        self._connected = True
        logger.info("mcp_server_http_connected", name=self.config.name, url=self.config.url)

    async def _initialize(self) -> None:
        init_request = self._protocol.create_request("initialize", {
            "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "NAN-Agent-v2.0",
                "version": "0.1.0",
            },
        })

        try:
            result = await asyncio.wait_for(
                self._send_request(init_request),
                timeout=MCP_HANDSHAKE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            await self._disconnect()
            raise ActionError(
                f"MCP handshake timed out for server '{self.config.name}'",
                error_code="E502",
            )
        except ActionError:
            await self._disconnect()
            raise

        self._server_capabilities = result.get("capabilities", {})
        logger.info(
            "mcp_server_initialized",
            name=self.config.name,
            capabilities=list(self._server_capabilities.keys()),
        )

        initialized_notification = self._protocol.create_notification("notifications/initialized")
        await self._send_message(initialized_notification)

        await self._list_tools()

    async def _list_tools(self) -> None:
        list_request = self._protocol.create_request("tools/list")
        result = await self._send_request(list_request)

        self._tools.clear()
        for tool_data in result.get("tools", []):
            mcp_tool = MCPTool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                inputSchema=tool_data.get("inputSchema", {}),
                server_name=self.config.name,
            )
            self._tools[mcp_tool.name] = mcp_tool

        logger.info(
            "mcp_tools_listed",
            server_name=self.config.name,
            tool_count=len(self._tools),
        )

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """调用 MCP 服务器上的工具。

        发送 tools/call JSON-RPC 请求，等待响应并提取结果。

        Args:
            tool_name: 工具名称
            parameters: 工具参数

        Returns:
            工具执行结果

        Raises:
            ActionError: 工具调用失败时抛出
        """
        call_request = self._protocol.create_request("tools/call", {
            "name": tool_name,
            "arguments": parameters,
        })

        try:
            result = await asyncio.wait_for(
                self._send_request(call_request),
                timeout=MCP_TOOL_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise ActionError(
                f"MCP tool call '{tool_name}' timed out on server '{self.config.name}'",
                error_code="E503",
            )

        content = result.get("content", [])
        texts = []
        for item in content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))

        return {
            "content": content,
            "text": "\n".join(texts) if texts else "",
            "isError": result.get("isError", False),
        }

    async def health_check(self) -> bool:
        if not self._connected:
            return False
        try:
            ping_request = self._protocol.create_request("ping")
            await asyncio.wait_for(
                self._send_request(ping_request),
                timeout=5.0,
            )
            return True
        except Exception:
            return False

    async def _send_request(self, request: Dict[str, Any]) -> Any:
        request_id = request["id"]
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._protocol.register_pending_request(request_id, future)

        await self._send_message(request)

        try:
            result = await future
            return result
        except Exception:
            self._protocol.reject_pending_request(request_id, ActionError(
                f"Request {request_id} failed",
                error_code="E502",
            ))
            raise

    async def _send_message(self, message: Dict[str, Any]) -> None:
        transport = self.config.transport_type
        if transport == "stdio":
            await self._send_stdio_message(message)
        elif transport == "sse":
            await self._send_sse_message(message)
        elif transport == "http":
            await self._send_http_message(message)

    async def _send_stdio_message(self, message: Dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise ActionError(
                f"Server '{self.config.name}' is not connected",
                error_code="E502",
            )
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode("utf-8"))
        await self._process.stdin.drain()

    async def _send_sse_message(self, message: Dict[str, Any]) -> None:
        """通过 SSE POST endpoint 发送 JSON-RPC 消息。

        SSE 传输模式下，消息通过 HTTP POST 发送到服务端，
        响应通过 SSE 事件流异步返回（由 _read_sse_messages 处理）。
        """
        if self._http_client is None or self._sse_post_endpoint is None:
            raise ActionError(
                f"Server '{self.config.name}' SSE is not connected",
                error_code="E502",
            )

        try:
            response = await self._http_client.post(
                self._sse_post_endpoint,
                json=message,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ActionError(
                f"SSE POST failed for server '{self.config.name}': HTTP {e.response.status_code}",
                error_code="E502",
                details={"server": self.config.name, "status_code": e.response.status_code},
            )
        except httpx.RequestError as e:
            raise ActionError(
                f"SSE POST request failed for server '{self.config.name}': {e}",
                error_code="E502",
                details={"server": self.config.name, "error": str(e)},
            )

    async def _send_http_message(self, message: Dict[str, Any]) -> None:
        """通过 HTTP POST 发送 JSON-RPC 消息并处理响应。

        HTTP (Streamable HTTP) 传输模式下，消息通过 HTTP POST 发送，
        服务端直接返回 JSON-RPC 响应或 SSE 事件流。
        响应立即解析并 resolve/reject 对应的 pending request。
        """
        if self._http_client is None or self._http_endpoint is None:
            raise ActionError(
                f"Server '{self.config.name}' HTTP is not connected",
                error_code="E502",
            )

        headers = {"Content-Type": "application/json"}

        # 通知消息：仅发送，无需处理响应
        if self._protocol.is_notification(message):
            try:
                response = await self._http_client.post(
                    self._http_endpoint,
                    json=message,
                    headers=headers,
                )
                response.raise_for_status()
            except httpx.RequestError as e:
                logger.warning(
                    "mcp_http_notification_failed",
                    name=self.config.name,
                    error=str(e),
                )
            return

        # 请求消息：发送并处理响应
        try:
            async with self._http_client.stream(
                "POST",
                self._http_endpoint,
                json=message,
                headers=headers,
            ) as response:
                if response.status_code == 202:
                    return

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")

                if "text/event-stream" in content_type:
                    # 服务端返回 SSE 事件流（Streamable HTTP 模式）
                    async for event_type, event_data in self._iter_sse_events(response):
                        if event_data:
                            try:
                                msg = json.loads(event_data)
                                await self._handle_message(msg)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "mcp_http_sse_invalid_json",
                                    name=self.config.name,
                                    data=event_data[:200],
                                )
                else:
                    # 服务端直接返回 JSON 响应
                    content = await response.aread()
                    if content:
                        try:
                            msg = json.loads(content)
                            await self._handle_message(msg)
                        except json.JSONDecodeError:
                            raise ActionError(
                                f"Invalid JSON response from server '{self.config.name}'",
                                error_code="E502",
                                details={"server": self.config.name},
                            )
        except httpx.HTTPStatusError as e:
            raise ActionError(
                f"HTTP POST failed for server '{self.config.name}': HTTP {e.response.status_code}",
                error_code="E502",
                details={"server": self.config.name, "status_code": e.response.status_code},
            )
        except httpx.RequestError as e:
            raise ActionError(
                f"HTTP POST request failed for server '{self.config.name}': {e}",
                error_code="E502",
                details={"server": self.config.name, "error": str(e)},
            )

    async def _read_stdio_messages(self) -> None:
        if self._process is None or self._process.stdout is None:
            return

        try:
            while self._connected:
                line = await self._process.stdout.readline()
                if not line:
                    logger.warning(
                        "mcp_server_stdout_closed",
                        name=self.config.name,
                    )
                    await self._handle_disconnect()
                    return

                try:
                    message = json.loads(line.decode("utf-8").strip())
                except json.JSONDecodeError:
                    logger.warning(
                        "mcp_server_invalid_json",
                        name=self.config.name,
                        line=line[:200],
                    )
                    continue

                await self._handle_message(message)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "mcp_server_read_error",
                name=self.config.name,
            )
            await self._handle_disconnect()

    async def _read_sse_messages(self) -> None:
        """后台任务：持续读取 SSE 事件流并分发消息。

        在 _connect_sse 中获取到 endpoint 事件后启动，
        持续读取 SSE 事件流中的 JSON-RPC 消息并交给 _handle_message 处理。
        连接断开时触发 _handle_disconnect 进行自动重连。
        """
        if self._sse_response is None:
            return

        try:
            async for event_type, event_data in self._iter_sse_events(self._sse_response):
                try:
                    message = json.loads(event_data)
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    logger.warning(
                        "mcp_sse_invalid_json",
                        name=self.config.name,
                        data=event_data[:200],
                    )

            # SSE 流正常结束 — 服务端关闭了连接
            if self._connected:
                logger.warning(
                    "mcp_sse_stream_closed",
                    name=self.config.name,
                )
                await self._handle_disconnect()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "mcp_sse_read_error",
                name=self.config.name,
            )
            if self._connected:
                await self._handle_disconnect()
        finally:
            if self._sse_response is not None:
                try:
                    await self._sse_response.aclose()
                except Exception:
                    pass
                self._sse_response = None

    async def _iter_sse_events(self, response: httpx.Response):
        """从 httpx 流式响应中解析 SSE 事件。

        SSE 协议格式：
            event: <type>
            data: <payload>
            <blank line>

        多行 data 字段以换行符拼接；无 event 字段时默认类型为 "message"；
        以冒号开头的行为注释，忽略。

        Args:
            response: httpx 流式响应对象（已通过 stream=True 打开）

        Yields:
            (event_type, data) 元组
        """
        event_type: Optional[str] = None
        data_buffer: List[str] = []

        async for line in response.aiter_lines():
            if line == "":
                # 空行 = 事件边界
                if data_buffer:
                    yield event_type or "message", "\n".join(data_buffer)
                event_type = None
                data_buffer = []
            elif line.startswith(":"):
                # SSE 注释，忽略
                pass
            elif line.startswith("event:"):
                event_type = line[6:].strip() or None
            elif line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data_buffer.append(value)

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        validation_error = self._protocol.validate_message(message)
        if validation_error is not None:
            logger.warning(
                "mcp_invalid_message",
                name=self.config.name,
                error=validation_error,
            )
            return

        if self._protocol.is_response(message):
            request_id = message.get("id")
            if "result" in message:
                self._protocol.resolve_pending_request(request_id, message["result"])
            elif "error" in message:
                error_info = message["error"]
                error_msg = error_info.get("message", "Unknown error")
                error_code = error_info.get("code", -32603)
                self._protocol.reject_pending_request(
                    request_id,
                    ActionError(
                        f"MCP error [{error_code}]: {error_msg}",
                        error_code="E503",
                        details={"server": self.config.name, "rpc_error": error_info},
                    ),
                )
        elif self._protocol.is_notification(message):
            await self._protocol.handle_notification(
                message["method"],
                message.get("params"),
            )
        else:
            logger.warning(
                "mcp_unhandled_message",
                name=self.config.name,
                message_type="unknown",
            )

    async def _handle_disconnect(self) -> None:
        self._connected = False
        self._protocol.clear_pending()

        if self._restart_attempts < MCP_MAX_RESTART_ATTEMPTS:
            self._restart_attempts += 1
            logger.warning(
                "mcp_server_restarting",
                name=self.config.name,
                attempt=self._restart_attempts,
                max_attempts=MCP_MAX_RESTART_ATTEMPTS,
            )
            await asyncio.sleep(MCP_RESTART_DELAY)
            try:
                await self._connect()
                self._restart_attempts = 0
            except Exception:
                logger.exception(
                    "mcp_server_restart_failed",
                    name=self.config.name,
                    attempt=self._restart_attempts,
                )


class MCPAdapter:
    """MCP 适配器 - 管理多个 MCP 服务器连接。

    负责加载配置（代码配置/环境变量）、连接所有服务器、
    将远程工具转换为 ActionRoom Tool 格式并注册。

    Attributes:
        has_configured_servers: 是否有已配置的服务器
        connected_count: 已连接的服务器数量
        servers: 服务器字典（name -> MCPServer）
    """

    def __init__(
        self,
        server_configs: Optional[List[MCPServerConfig]] = None,
        config: Optional[Dict[str, Any]] = None,
        rate_limit_per_second: float = 10.0,
    ):
        """初始化 MCP 适配器。

        Args:
            server_configs: 服务器配置列表（可选）
            config: 全局配置字典，从 action_room.mcp_servers 读取服务器列表
            rate_limit_per_second: 每秒最大请求数，默认 10
        """
        self._servers: Dict[str, MCPServer] = {}
        self._rate_limit_per_second = rate_limit_per_second
        self._rate_limiters: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._pending_configs: List[MCPServerConfig] = []

        if server_configs:
            self._pending_configs.extend(server_configs)

        if config:
            mcp_cfg = config.get("action_room", {}).get("mcp_servers", [])
            for srv in mcp_cfg:
                if isinstance(srv, dict):
                    self._pending_configs.append(MCPServerConfig(**srv))

        if not self._pending_configs:
            self._pending_configs = self._load_from_env()

    @property
    def has_configured_servers(self) -> bool:
        return len(self._pending_configs) > 0 or len(self._servers) > 0

    @property
    def connected_count(self) -> int:
        return sum(1 for s in self._servers.values() if s.connected)

    async def initialize(self) -> None:
        """Connect to all configured MCP servers. Call after construction."""
        for cfg in self._pending_configs:
            try:
                await self.connect_server(cfg)
            except Exception:
                logger.warning("mcp_server_connect_failed", name=cfg.name)
        self._pending_configs.clear()

    @staticmethod
    def _load_from_env() -> List[MCPServerConfig]:
        configs: List[MCPServerConfig] = []
        mcp_json = os.environ.get("MCP_SERVERS_CONFIG", "")
        if mcp_json:
            try:
                servers = json.loads(mcp_json)
                for srv in servers:
                    configs.append(MCPServerConfig(**srv))
            except (json.JSONDecodeError, TypeError):
                logger.warning("mcp_env_config_parse_failed")
        return configs

    @property
    def servers(self) -> Dict[str, MCPServer]:
        return self._servers.copy()

    async def connect_server(self, config: MCPServerConfig) -> None:
        if config.name in self._servers:
            raise ActionError(
                f"MCP server '{config.name}' is already connected",
                error_code="E502",
                details={"server_name": config.name},
            )

        server = MCPServer(config)
        await server.start()
        self._servers[config.name] = server
        self._rate_limiters[config.name] = 0.0
        logger.info(
            "mcp_adapter_server_connected",
            server_name=config.name,
            transport=config.transport_type,
        )

    async def disconnect_server(self, server_name: str) -> bool:
        server = self._servers.pop(server_name, None)
        self._rate_limiters.pop(server_name, None)
        if server is not None:
            await server.stop()
            logger.info("mcp_adapter_server_disconnected", server_name=server_name)
            return True
        return False

    async def disconnect_all(self) -> None:
        server_names = list(self._servers.keys())
        for name in server_names:
            await self.disconnect_server(name)

    async def list_all_tools(self) -> Dict[str, MCPTool]:
        all_tools: Dict[str, MCPTool] = {}
        for server_name, server in self._servers.items():
            for tool_name, tool in server.tools.items():
                qualified_name = f"{server_name}.{tool_name}"
                all_tools[qualified_name] = tool
        return all_tools

    async def convert_to_action_room_tools(self) -> List[Tool]:
        all_mcp_tools = await self.list_all_tools()
        converted: List[Tool] = []

        for qualified_name, mcp_tool in all_mcp_tools.items():
            async def make_handler(srv_name, tl_name):
                async def handler(**kwargs):
                    return await self.call_tool(srv_name, tl_name, kwargs)
                return handler

            tool = Tool(
                name=qualified_name,
                description=f"[MCP:{mcp_tool.server_name}] {mcp_tool.description}",
                parameters=mcp_tool.inputSchema or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=await make_handler(mcp_tool.server_name, mcp_tool.name),
                category="mcp",
                tags=["mcp", mcp_tool.server_name],
            )
            converted.append(tool)

        return converted

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        await self._apply_rate_limit(server_name)

        server = self._servers.get(server_name)
        if server is None:
            raise ActionError(
                f"MCP server '{server_name}' is not connected",
                error_code="E502",
                details={"server_name": server_name},
            )

        return await server.call_tool(tool_name, arguments)

    async def call_qualified_tool(
        self,
        qualified_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        parts = qualified_name.split(".", 1)
        if len(parts) != 2:
            raise ActionError(
                f"Invalid qualified tool name: '{qualified_name}'. Expected format: 'server_name.tool_name'",
                error_code="E502",
            )
        server_name, tool_name = parts
        return await self.call_tool(server_name, tool_name, arguments)

    async def health_check_all(self) -> Dict[str, bool]:
        results = {}
        for server_name, server in self._servers.items():
            results[server_name] = await server.health_check()
        return results

    async def health_check_server(self, server_name: str) -> Optional[bool]:
        server = self._servers.get(server_name)
        if server is None:
            return None
        return await server.health_check()

    async def discover_server(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> List[MCPTool]:
        temp_name = f"discovery_{uuid.uuid4().hex[:8]}"
        config = MCPServerConfig(
            name=temp_name,
            command=command,
            args=args or [],
            env=env or {},
            transport_type="stdio" if command else "http",
        )

        server = MCPServer(config)
        try:
            await server.start()
            tools = list(server.tools.values())
            return tools
        finally:
            await server.stop()

    async def _apply_rate_limit(self, server_name: str) -> None:
        async with self._lock:
            now = time.monotonic()
            last_call = self._rate_limiters.get(server_name, 0.0)
            min_interval = 1.0 / self._rate_limit_per_second
            wait_time = min_interval - (now - last_call)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._rate_limiters[server_name] = time.monotonic()