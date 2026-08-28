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
        frontend_dir: str | Path | None = None,   # 新增：前端构建目录
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

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def serve(self) -> None:
        """启动 HTTP + WebSocket 服务器"""
        app = FastAPI()

        # ---------- WebSocket 端点 ----------
        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            await self._ws_handler(websocket)

        # ---------- 前端静态文件托管 ----------
        if self.frontend_dir and self.frontend_dir.exists():
            assets_dir = self.frontend_dir / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

            # SPA 回退：所有未匹配路由返回 index.html
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
        await self._server.serve()

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

        self._clients.add(websocket)
        queue = self.bus.subscribe()

        logger.info("gateway: client connected {}", peer)

        try:
            # 发送 hello
            await self._send(websocket, self._hello_payload())
            await self._send(
                websocket,
                {"t": "divider", "label": local_date_label()},
            )

            # 回放历史
            for event in self.bus.history():
                await self._send_with_date(websocket, event)

            # 启动转发任务
            forwarder = asyncio.create_task(
                self._forward(websocket, queue),
                name="gateway-forward",
            )

            try:
                # 接收客户端消息
                async for raw in websocket.iter_text():
                    self._handle_message(websocket, raw)
            finally:
                forwarder.cancel()
                try:
                    await forwarder
                except (asyncio.CancelledError, Exception):
                    pass

        except WebSocketDisconnect:
            logger.info("gateway: client {} disconnected", peer)
        except Exception:
            logger.exception("Gateway connection error")
        finally:
            self.bus.unsubscribe(queue)
            self._clients.discard(websocket)
            logger.info("gateway: client {} cleaned up", peer)

    async def _forward(self, websocket: WebSocket, queue: asyncio.Queue) -> None:
        last_date = local_date_label()

        while True:
            event = await queue.get()
            event_date = local_date_label(event.get("ts"))

            if event_date != last_date:
                last_date = event_date
                await self._send(
                    websocket,
                    {"t": "divider", "label": event_date},
                )

            await self._send(websocket, event)

    def _handle_message(self, websocket: WebSocket, raw: str) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            return

        if not isinstance(data, dict):
            return

        kind = data.get("t")

        if kind == "input":
            text = str(data.get("text") or "").strip()
            mid = data.get("mid")

            if text and self.on_input is not None:
                self.on_input(text, mid if isinstance(mid, str) else None)

        elif kind == "ping":
            asyncio.create_task(
                self._safe_send(websocket, {"t": "pong"}),
                name="gateway-pong",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send(self, websocket: WebSocket, payload: dict) -> None:
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    async def _safe_send(self, websocket: WebSocket, payload: dict) -> None:
        await self._send(websocket, payload)

    async def _send_with_date(self, websocket: WebSocket, event: dict) -> None:
        await self._send(websocket, event)

    def _hello_payload(self) -> dict:
        state: dict = {}
        if self.state_provider is not None:
            try:
                state = self.state_provider() or {}
            except Exception:
                state = {}
        return {
            "t": "hello",
            "seq": self.bus.seq,
            **state,
        }