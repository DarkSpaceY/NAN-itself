"""
工具注册中心 - 统一的工具注册、校验与执行框架

管理所有 Agent 可调用工具的生命周期，包括注册、查找、参数校验、
超时执行、调用统计和事件通知。通过 EventBus 发出工具执行事件。

核心组件：
- Tool: 工具定义（名称、描述、参数schema、处理器、分类标签）
- ToolRegistry: 工具注册中心
- ToolResult: 工具执行结果
- ToolStatistics: 工具调用统计
"""

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

from nan_agent.event_bus import EventBus
from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Union[Callable, Callable[..., Any]]
    category: str = "uncategorized"
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    enabled: bool = True

    def __hash__(self):
        return hash(self.name)


@dataclass
class ToolResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tool_name: str = ""


@dataclass
class ToolStatistics:
    calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0

    def record_call(self, success: bool, execution_time_ms: float) -> None:
        self.calls += 1
        self.total_execution_time_ms += execution_time_ms
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.average_execution_time_ms = self.total_execution_time_ms / self.calls


class ToolRegistry:
    """工具注册中心。

    管理所有 Agent 工具的注册、查找、参数校验、超时执行和调用统计。
    通过 EventBus 发布工具注册和执行事件。

    特性：
    - 防止重复注册
    - 按分类/标签索引
    - JSON Schema 参数校验
    - 自动超时控制
    - 调用统计追踪
    """

    def __init__(self, event_bus: Optional[EventBus] = None, default_timeout: float = 30.0):
        """初始化工具注册中心。

        Args:
            event_bus: 事件总线，用于发布工具注册/执行事件
            default_timeout: 默认工具执行超时（秒）
        """
        self._tools: Dict[str, Tool] = {}
        self._statistics: Dict[str, ToolStatistics] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._by_tag: Dict[str, List[str]] = {}
        self._event_bus = event_bus
        self._default_timeout = default_timeout
        self._logger = logger.bind(component="ToolRegistry")

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools.copy()

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ActionError(
                f"Tool '{tool.name}' is already registered",
                error_code="E501",
                details={"tool_name": tool.name},
            )

        self._tools[tool.name] = tool
        self._statistics[tool.name] = ToolStatistics()

        if tool.category not in self._by_category:
            self._by_category[tool.category] = []
        self._by_category[tool.category].append(tool.name)

        for tag in tool.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(tool.name)

        self._logger.info("tool_registered", tool_name=tool.name, category=tool.category, tags=tool.tags)

        if self._event_bus is not None:
            self._schedule_event("tool.registered", tool=tool)

    def register_tool(
        self,
        name: Optional[str] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        category: str = "uncategorized",
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        """声明式工具注册装饰器。

        从被装饰函数的签名自动推断 JSON Schema 参数定义，
        无需手动编写 parameters 字典。

        Args:
            name: 工具名称，默认使用函数名
            description: 工具描述，默认使用函数 docstring
            parameters: 手动指定的参数 schema（覆盖自动推断）
            category: 工具分类
            tags: 工具标签列表
            enabled: 是否启用

        Returns:
            装饰器函数

        Example::

            @registry.register_tool(
                name="read_file",
                description="Read a file",
                category="filesystem",
                tags=["file", "read"],
            )
            async def read_file(path: str, encoding: str = "utf-8") -> str:
                ...
        """
        def decorator(func):
            tool_name = name or func.__name__
            tool_description = description or (func.__doc__ or "").strip().split("\n")[0]
            tool_tags = tags or []

            if parameters is not None:
                tool_parameters = parameters
            else:
                tool_parameters = self._infer_schema_from_signature(func)

            tool = Tool(
                name=tool_name,
                description=tool_description,
                parameters=tool_parameters,
                handler=func,
                category=category,
                tags=tool_tags,
                enabled=enabled,
            )

            self.register(tool)
            return func

        return decorator

    def register_from_config(
        self,
        config: Dict[str, Any],
        handler_resolver: Callable[[str], Optional[Callable]],
    ) -> None:
        """从配置字典批量注册工具。

        Args:
            config: 工具配置字典，格式为 {"tools": [...]}
            handler_resolver: 处理器解析函数，将 handler 路径字符串
                解析为可调用对象，解析失败返回 None

        Example::

            config = {"tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "category": "filesystem",
                    "parameters": {"type": "object", ...},
                    "handler": "filesystem.read_file",
                },
            ]}
            registry.register_from_config(config, my_resolver)
        """
        tools_config = config.get("tools", [])

        for tool_config in tools_config:
            handler_path = tool_config.get("handler")
            if not handler_path:
                self._logger.warning("tool_config_missing_handler", config=tool_config)
                continue

            handler = handler_resolver(handler_path)
            if not handler:
                self._logger.warning("tool_handler_not_found", handler=handler_path)
                continue

            tool = Tool(
                name=tool_config.get("name", ""),
                description=tool_config.get("description", ""),
                parameters=tool_config.get("parameters", {}),
                handler=handler,
                category=tool_config.get("category", "uncategorized"),
                tags=tool_config.get("tags", []),
                enabled=tool_config.get("enabled", True),
            )

            try:
                self.register(tool)
            except ActionError:
                self._logger.warning("tool_config_register_failed", tool_name=tool.name)

    def _infer_schema_from_signature(self, func: Callable) -> Dict[str, Any]:
        """从函数签名推断 JSON Schema 参数定义。

        自动提取参数名、类型和是否必需（无默认值则为必需）。
        支持 str/int/float/bool/list/dict 及 Optional[T] 类型。

        Args:
            func: 要推断的函数

        Returns:
            JSON Schema 格式的参数定义
        """
        try:
            sig = inspect.signature(func)
            hints = get_type_hints(func)
        except Exception:
            return {"type": "object", "properties": {}, "required": []}

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls", "**kw", "kw"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            param_type = hints.get(param_name, str)
            param_schema = self._type_to_schema(param_type)

            has_default = param.default is not inspect.Parameter.empty
            if not has_default:
                required.append(param_name)

            properties[param_name] = param_schema

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @staticmethod
    def _type_to_schema(python_type: Any) -> Dict[str, Any]:
        """将 Python 类型映射到 JSON Schema 类型定义。

        支持 Optional[T] 解包（映射为 T 的 schema）。

        Args:
            python_type: Python 类型对象

        Returns:
            JSON Schema 类型定义字典
        """
        type_mapping: Dict[type, Dict[str, str]] = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }

        # 处理 Optional[T]（即 Union[T, None]）
        origin = getattr(python_type, "__origin__", None)
        if origin is Union:
            args = getattr(python_type, "__args__", ())
            non_none = [t for t in args if t is not type(None)]
            if non_none:
                return ToolRegistry._type_to_schema(non_none[0])

        return type_mapping.get(python_type, {"type": "string"})

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        tool = self._tools.get(tool_name)
        if tool is None or not tool.enabled:
            return None
        return tool

    def list_by_category(self, category: str) -> List[Tool]:
        tool_names = self._by_category.get(category, [])
        return [self._tools[name] for name in tool_names if self._tools[name].enabled]

    def get_statistics(self, tool_name: str) -> Optional[ToolStatistics]:
        return self._statistics.get(tool_name)

    def get_all_statistics(self) -> Dict[str, ToolStatistics]:
        return self._statistics.copy()

    def _validate_parameters(self, parameters: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
        required = schema.get("required", [])

        for field in required:
            if field not in parameters:
                return f"Parameter validation failed: '{field}' is a required property"

        properties = schema.get("properties", {})
        allowed_keys = set(properties.keys())

        for key in parameters:
            if key not in allowed_keys:
                return f"Parameter validation failed: Additional properties are not allowed ('{key}' was unexpected)"

            prop_schema = properties.get(key, {})
            prop_type = prop_schema.get("type")

            value = parameters[key]
            if prop_type and not self._check_type(value, prop_type):
                return f"Parameter validation failed: '{key}' should be of type '{prop_type}'"

        return None

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "array":
            return isinstance(value, list)
        return True

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """执行工具。

        依次执行：工具查找 → 参数校验 → 超时控制执行 → 统计记录 → 事件通知。

        Args:
            tool_name: 工具名称
            parameters: 工具参数字典
            timeout: 超时时间（秒），None 使用默认值

        Returns:
            ToolResult 包含执行结果或错误信息
        """
        start_time = time.perf_counter()
        tool = self.get_tool(tool_name)

        if tool is None:
            execution_time = (time.perf_counter() - start_time) * 1000
            result = ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found or disabled",
                execution_time_ms=execution_time,
                tool_name=tool_name,
            )
            await self._emit_execution_event(result)
            return result

        validation_error = self._validate_parameters(parameters, tool.parameters)
        if validation_error is not None:
            execution_time = (time.perf_counter() - start_time) * 1000
            result = ToolResult(
                success=False,
                error=validation_error,
                execution_time_ms=execution_time,
                tool_name=tool_name,
            )
            self._record_statistics(tool_name, False, execution_time)
            await self._emit_execution_event(result)
            return result

        timeout = timeout if timeout is not None else self._default_timeout
        result = await self._execute_with_timeout(tool, parameters, timeout, start_time)

        self._record_statistics(tool_name, result.success, result.execution_time_ms)
        await self._emit_execution_event(result)

        return result

    async def _execute_with_timeout(
        self,
        tool: Tool,
        parameters: Dict[str, Any],
        timeout: float,
        start_time: float,
    ) -> ToolResult:
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                data = await asyncio.wait_for(tool.handler(**parameters), timeout=timeout)
            else:
                data = await asyncio.wait_for(
                    asyncio.to_thread(tool.handler, **parameters),
                    timeout=timeout,
                )

            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=True,
                data=data,
                execution_time_ms=execution_time,
                tool_name=tool.name,
            )

        except asyncio.TimeoutError:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._logger.warning("tool_execution_timeout", tool_name=tool.name, timeout=timeout)
            return ToolResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds",
                execution_time_ms=execution_time,
                tool_name=tool.name,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            self._logger.exception("tool_execution_error", tool_name=tool.name, error=str(e))
            return ToolResult(
                success=False,
                error=f"Execution error: {str(e)}",
                execution_time_ms=execution_time,
                tool_name=tool.name,
            )

    def _record_statistics(self, tool_name: str, success: bool, execution_time_ms: float) -> None:
        stats = self._statistics.get(tool_name)
        if stats is not None:
            stats.record_call(success, execution_time_ms)

    async def _emit_event(self, event_name: str, **kwargs) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.emit(event_name, **kwargs)
        except Exception:
            self._logger.exception("event_emit_error", event=event_name)

    def _schedule_event(self, event_name: str, **kwargs) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_event(event_name, **kwargs))
        except RuntimeError as e:
            self._logger.debug("schedule_event_no_running_loop", error=str(e))

    async def _emit_execution_event(self, result: ToolResult) -> None:
        await self._emit_event(
            "tool.executed",
            tool_name=result.tool_name,
            success=result.success,
            execution_time_ms=result.execution_time_ms,
        )

    def list_all_tools(self, include_disabled: bool = False) -> List[Tool]:
        if include_disabled:
            return list(self._tools.values())
        return [tool for tool in self._tools.values() if tool.enabled]

    def enable_tool(self, tool_name: str) -> bool:
        tool = self._tools.get(tool_name)
        if tool is None:
            return False
        tool.enabled = True
        return True

    def disable_tool(self, tool_name: str) -> bool:
        tool = self._tools.get(tool_name)
        if tool is None:
            return False
        tool.enabled = False
        return True

    def unregister(self, tool_name: str) -> bool:
        """注销工具。

        如果工具存在则删除并返回 True，否则返回 False。
        同时清理 _tools, _statistics, _by_category, _by_tag 中的相关数据。
        如果 event_bus 存在，发出 "tool.unregistered" 事件。

        Args:
            tool_name: 要注销的工具名称

        Returns:
            工具是否存在并被成功注销
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return False

        # 清理 _tools
        del self._tools[tool_name]

        # 清理 _statistics
        self._statistics.pop(tool_name, None)

        # 清理 _by_category
        category_names = self._by_category.get(tool.category, [])
        if tool_name in category_names:
            category_names.remove(tool_name)
        if not category_names:
            self._by_category.pop(tool.category, None)

        # 清理 _by_tag
        for tag in tool.tags:
            tag_names = self._by_tag.get(tag, [])
            if tool_name in tag_names:
                tag_names.remove(tool_name)
            if not tag_names:
                self._by_tag.pop(tag, None)

        self._logger.info("tool_unregistered", tool_name=tool_name)

        if self._event_bus is not None:
            self._schedule_event("tool.unregistered", tool_name=tool_name)

        return True

    def search_by_tag(self, tag: str) -> List[Tool]:
        """按标签搜索工具。

        返回包含指定标签的已启用工具列表。

        Args:
            tag: 要搜索的标签

        Returns:
            包含该标签的已启用工具列表
        """
        tool_names = self._by_tag.get(tag, [])
        return [self._tools[name] for name in tool_names if name in self._tools and self._tools[name].enabled]
