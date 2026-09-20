"""
WS gateway: bridge the EventBus to the UI.

    ws://127.0.0.1:8765/ws
    http://127.0.0.1:8765/  (frontend static files)

Server-authoritative stream:

    - on connect: hello + history replay
    - live: forward subscribed events
    - inbound: {"t":"input","text"} -> on_input(text)

The gateway owns nothing about agent semantics; it only formats
the stream and inserts a date divider when the wall-clock date
rolls over.
"""

from __future__ import annotations

import asyncio
import json
import socket
import webbrowser
from pathlib import Path
from typing import Any, Callable

from loguru import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from .events import EventBus, local_date_label


class Gateway:
    def __init__(
        self,
        *,
        bus: EventBus,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_input: Callable[[str, Any], None] | None = None,
        state_provider: Callable[[], dict] | None = None,
        frontend_dir: str | Path | None = None,
    ) -> None:
        self.bus = bus
        self.host = host
        self.port = port
        self.on_input = on_input
        self.state_provider = state_provider
        self.frontend_dir = Path(frontend_dir) if frontend_dir else None

        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        self._clients: set[WebSocket] = set()

        # 回执去重：客户端断线重连会以同一 mid 补发（可能已投递过），
        # 这里按 mid 保证恰好一次；容量有界，旧 mid 淘汰。
        self._seen_mids: dict[str, None] = {}

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def serve(self, open_browser: bool = True) -> None:
        """启动 HTTP + WebSocket 服务器，若 open_browser 则自动打开浏览器"""
        app = FastAPI()

        # ---------- WebSocket 端点 ----------
        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            while websocket.client_state.name != "CONNECTED":
                await asyncio.sleep(0.01)
            await self._ws_handler(websocket)

        # ---------- 前端静态文件托管 ----------
        if self.frontend_dir and self.frontend_dir.exists():
            assets_dir = self.frontend_dir / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str):
                index_path = self.frontend_dir / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))
                return {"error": "Frontend not built"}

            logger.info(f"Frontend static files served from {self.frontend_dir}")
        else:
            logger.warning("Frontend directory not found, skipping static file serving")

        # ---------- 启动服务器 ----------
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        # 将服务器启动放到后台任务，以便我们检查端口并打开浏览器
        server_task = asyncio.create_task(self._server.serve())

        if open_browser:
            # 等待服务器真正启动（尝试建立TCP连接）
            start = asyncio.get_event_loop().time()
            connected = False
            while not connected and (asyncio.get_event_loop().time() - start) < 10:
                try:
                    reader, writer = await asyncio.open_connection(self.host, self.port)
                    writer.close()
                    await writer.wait_closed()
                    connected = True
                except Exception:
                    await asyncio.sleep(0.2)
            if connected:
                url = f"http://{self.host}:{self.port}/"
                if webbrowser.open(url):
                    logger.info(f"🌐 Opened browser at {url}")
                else:
                    logger.warning(f"Could not open browser, please navigate to {url}")
            else:
                logger.warning("Server did not start in time, browser not opened")

        # 等待服务器任务（阻塞直到服务停止）
        await server_task

    async def close(self) -> None:
        """关闭服务器"""
        if self._server:
            self._server.should_exit = True
            await self._server.shutdown()

        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass

        logger.info("Gateway closed")

    # ==================================================================
    # WebSocket connection handling
    # ==================================================================

    async def _ws_handler(self, websocket: WebSocket) -> None:
        peer = websocket.client
        client_id = f"{peer.host}:{peer.port}" if peer else "unknown"

        self._clients.add(websocket)
        queue = self.bus.subscribe()

        logger.info(f"Client {client_id} connected")

        try:
            # 发送 hello + divider + 历史
            await self._send(websocket, self._hello_payload())
            await self._send(websocket, {"t": "divider", "label": local_date_label()})
            for event in self.bus.history():
                await self._send(websocket, event)

            # 启动转发任务
            forwarder = asyncio.create_task(
                self._forward(websocket, queue, client_id),
                name=f"forward-{client_id}",
            )

            # 启动监控任务（仅检测 forwarder 是否意外死亡）
            monitor = asyncio.create_task(
                self._monitor_forwarder(websocket, forwarder, client_id),
                name=f"monitor-{client_id}",
            )

            try:
                # 接收客户端消息
                async for raw in websocket.iter_text():
                    self._handle_message(websocket, raw)
            finally:
                forwarder.cancel()
                monitor.cancel()
                try:
                    await forwarder
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Forwarder error on cleanup: {e}")
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Monitor error on cleanup: {e}")

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.exception(f"Gateway connection error for {client_id}: {e}")
        finally:
            self.bus.unsubscribe(queue)
            self._clients.discard(websocket)
            logger.info(f"Client {client_id} cleaned up")

    async def _forward(self, websocket: WebSocket, queue: asyncio.Queue, client_id: str) -> None:
        """转发事件到客户端（带异常处理）"""
        last_date = local_date_label()
        counter = 0
        logger.debug(f"Forwarder started for {client_id}")

        try:
            while True:
                # 带超时获取事件，避免永久阻塞
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    # 超时检查连接状态
                    if websocket.client_state.name != "CONNECTED":
                        logger.warning(f"Forwarder: WebSocket not connected for {client_id}, stopping")
                        break
                    continue

                counter += 1

                # 处理日期分隔
                event_date = local_date_label(event.get("ts"))
                if event_date != last_date:
                    last_date = event_date
                    await self._send(websocket, {"t": "divider", "label": event_date})

                # 发送事件
                logger.info(f"[Debug] Send Event:{event}")
                await self._send(websocket, event)

        except asyncio.CancelledError:
            logger.debug(f"Forwarder cancelled for {client_id} after {counter} events")
            raise
        except Exception as e:
            logger.error(f"Forwarder error for {client_id}: {e}", exc_info=True)
            # 发生错误，退出循环
        finally:
            logger.debug(f"Forwarder stopped for {client_id} (processed {counter} events)")

    async def _monitor_forwarder(self, websocket: WebSocket, forwarder: asyncio.Task, client_id: str) -> None:
        """监控 forwarder 是否意外退出，若退出则主动关闭连接"""
        try:
            while True:
                await asyncio.sleep(3)
                if forwarder.done():
                    logger.error(f"Monitor: forwarder unexpectedly done for {client_id}, closing connection")
                    try:
                        await websocket.close(code=1011, reason="Forwarder died")
                    except Exception:
                        pass
                    break
                # 如果连接已断开，也退出
                if websocket.client_state.name != "CONNECTED":
                    logger.debug(f"Monitor: WebSocket not connected for {client_id}, stopping")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Monitor error for {client_id}: {e}")

    def _handle_message(self, websocket: WebSocket, raw: str) -> None:
        """处理客户端消息（仅记录错误）"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {raw[:100]}")
            return
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return

        if not isinstance(data, dict):
            return

        kind = data.get("t")
        if kind == "input":
            text = str(data.get("text") or "").strip()
            mid = data.get("mid")
            mid = mid if isinstance(mid, str) else None

            if text and self.on_input is not None:
                if not self._accept_mid(mid):
                    return

                try:
                    self.on_input(text, mid)
                except Exception as e:
                    logger.error(f"on_input callback error: {e}")
            elif not text:
                logger.debug("Empty input ignored")
        elif kind == "ping":
            asyncio.create_task(self._safe_send(websocket, {"t": "pong"}))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _accept_mid(self, mid: str | None) -> bool:
        """按 mid 去重：重复补发直接丢弃；容量有界，旧 mid 淘汰。"""
        if mid is None:
            return True

        if mid in self._seen_mids:
            logger.info("duplicate input dropped (mid={})", mid)
            return False

        self._seen_mids[mid] = None

        if len(self._seen_mids) > 256:
            for key in list(self._seen_mids)[:128]:
                self._seen_mids.pop(key, None)

        return True

    def _latest_status(self) -> dict:
        """bus 历史里最近一条 status 事件的快照；没有则 idle。"""
        for event in reversed(self.bus.history()):
            if event.get("t") == "status":
                return {
                    key: event[key]
                    for key in ("state", "tools", "subagents", "next_hop")
                    if key in event
                }

        return {"state": "idle"}

    async def _send(self, websocket: WebSocket, payload: dict) -> None:
        """发送消息，失败时记录并抛出异常（让上层处理）"""
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Send failed for {payload.get('t')}: {e}")
            # 重新抛出，让调用者知道发送失败
            raise

    async def _safe_send(self, websocket: WebSocket, payload: dict) -> None:
        """安全发送（忽略异常，用于 ping 响应）"""
        try:
            await self._send(websocket, payload)
        except Exception:
            pass

    def _hello_payload(self) -> dict:
        state = {}
        if self.state_provider is not None:
            try:
                state = self.state_provider() or {}
            except Exception as e:
                logger.error(f"State provider error: {e}")

        # status 快照属于 gateway 职责：app 级 state 未携带时，
        # 从 bus 历史取最近一条 status。
        if "status" not in state:
            state["status"] = self._latest_status()

        return {"t": "hello", "seq": self.bus.seq, **state}