"""
WS gateway: bridge the process-wide event sink to the UI.

    ws://127.0.0.1:8765/ws
    http://127.0.0.1:8765/  (frontend static files)

Server-authoritative stream:

    - on connect: hello + history replay
    - live: forward subscribed events
    - inbound: {"t":"input","text","mid"} -> on_input(text, mid)
               (mid dedup is handled here, bounded cache)
              {"t":"pause"}             -> on_pause(True)
              {"t":"resume"}            -> on_pause(False)
              {"t":"ping"}              -> {"t":"pong"}

The gateway only formats the stream. Date dividers are rendered
by the frontend from the events' timestamps.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .events import sink


class Gateway:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_input: Callable[[str, Any], None] | None = None,
        on_pause: Callable[[bool], None] | None = None,
        state_provider: Callable[[], dict] | None = None,
        frontend_dir: str | Path | None = None,
        dedup_cache_size: int = 256,
    ) -> None:
        self.host = host
        self.port = port
        self.on_input = on_input
        self.on_pause = on_pause
        self.state_provider = state_provider
        self.frontend_dir = Path(frontend_dir) if frontend_dir else None
        self.dedup_cache_size = dedup_cache_size

        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        self._clients: set[WebSocket] = set()

        # Input dedup by mid: a reconnecting client may re-send
        # the same mid; seen mids are dropped, the cache is
        # bounded and evicts oldest entries first.
        self._seen_mids: dict[str, None] = {}

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def serve(self, open_browser: bool = True) -> None:
        """Start the HTTP + WebSocket server; open the browser when asked."""
        app = FastAPI()

        # ---------- Entry guard (loopback authentication) ----------
        # The socket is loopback-only; the guard closes the
        # remaining hole, DNS rebinding: a hostile page resolving
        # its domain to 127.0.0.1 produces requests with the
        # attacker's Host and Origin, both rejected here. Browser
        # requests always carry Origin and cannot forge it.
        allowed_hosts = {
            f"127.0.0.1:{self.port}",
            f"localhost:{self.port}",
            "[::1]:" + str(self.port),
            "127.0.0.1",
            "localhost",
        }
        allowed_origins = {
            f"http://127.0.0.1:{self.port}",
            f"http://localhost:{self.port}",
            f"http://[::1]:{self.port}",
        }

        def entry_allowed(headers: Any) -> bool:
            host = headers.get("host", "")

            if host not in allowed_hosts:
                return False

            origin = headers.get("origin")

            # Native clients (curl, local ws tools) send no Origin;
            # browsers always do and cannot forge it.
            return origin is None or origin in allowed_origins

        @app.middleware("http")
        async def entry_guard_http(request, call_next):
            if not entry_allowed(request.headers):
                return JSONResponse({"error": "forbidden"}, status_code=403)

            return await call_next(request)

        # ---------- WebSocket endpoint ----------
        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            if not entry_allowed(websocket.headers):
                await websocket.close(code=1008)
                return

            await websocket.accept()
            while websocket.client_state.name != "CONNECTED":
                await asyncio.sleep(0.01)
            await self._ws_handler(websocket)

        # ---------- Frontend static files ----------
        if self.frontend_dir and self.frontend_dir.exists():
            frontend_dir = self.frontend_dir

            assets_dir = frontend_dir / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str):
                index_path = frontend_dir / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))
                return {"error": "Frontend not built"}

            logger.info(f"Frontend static files served from {self.frontend_dir}")
        else:
            logger.warning("Frontend directory not found, skipping static file serving")

        # ---------- Server start ----------
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        # Run the server as a background task so the port can be
        # probed and the browser opened.
        server_task = asyncio.create_task(self._server.serve())

        if open_browser:
            # Wait for the server to come up (probe a TCP connection).
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

        # Block until the server task ends.
        await server_task

    async def close(self) -> None:
        """Shut the server down."""
        if self._server:
            self._server.should_exit = True
            await self._server.shutdown()

        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                logger.exception("WebSocket close failed during gateway shutdown")

        logger.info("Gateway closed")

    # ==================================================================
    # WebSocket connection handling
    # ==================================================================

    async def _ws_handler(self, websocket: WebSocket) -> None:
        peer = websocket.client
        client_id = f"{peer.host}:{peer.port}" if peer else "unknown"

        self._clients.add(websocket)
        queue = sink.subscribe()

        logger.info(f"Client {client_id} connected")

        try:
            # Send hello + history replay (date dividers are
            # rendered by the frontend from event timestamps).
            await self._send(websocket, self._hello_payload())
            for event in sink.history():
                await self._send(websocket, event)

            # Start the forwarder task.
            forwarder = asyncio.create_task(
                self._forward(websocket, queue, client_id),
                name=f"forward-{client_id}",
            )

            # Start the monitor task (detects an unexpected
            # forwarder death).
            monitor = asyncio.create_task(
                self._monitor_forwarder(websocket, forwarder, client_id),
                name=f"monitor-{client_id}",
            )

            try:
                # Receive client messages.
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
            sink.unsubscribe(queue)
            self._clients.discard(websocket)
            logger.info(f"Client {client_id} cleaned up")

    async def _forward(self, websocket: WebSocket, queue: asyncio.Queue, client_id: str) -> None:
        """Forward events to the client."""
        counter = 0
        logger.debug(f"Forwarder started for {client_id}")

        try:
            while True:
                # Get the next event with a timeout so a dead
                # socket cannot block the loop forever.
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                except TimeoutError:
                    # Re-check the connection state.
                    if websocket.client_state.name != "CONNECTED":
                        logger.warning(
                            f"Forwarder: WebSocket not connected for {client_id}, stopping"
                        )
                        break
                    continue

                counter += 1

                logger.info(f"[Debug] Send Event:{event}")
                await self._send(websocket, event)

        except asyncio.CancelledError:
            logger.debug(f"Forwarder cancelled for {client_id} after {counter} events")
            raise
        except Exception as e:
            logger.error(f"Forwarder error for {client_id}: {e}", exc_info=True)
        finally:
            logger.debug(f"Forwarder stopped for {client_id} (processed {counter} events)")

    async def _monitor_forwarder(
        self, websocket: WebSocket, forwarder: asyncio.Task, client_id: str
    ) -> None:
        """Watch the forwarder; close the connection if it dies."""
        try:
            while True:
                await asyncio.sleep(3)
                if forwarder.done():
                    logger.error(
                        f"Monitor: forwarder unexpectedly done for {client_id}, closing connection"
                    )
                    with contextlib.suppress(Exception):
                        await websocket.close(code=1011, reason="Forwarder died")
                    break
                # Also stop once the connection is gone.
                if websocket.client_state.name != "CONNECTED":
                    logger.debug(f"Monitor: WebSocket not connected for {client_id}, stopping")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Monitor error for {client_id}: {e}")

    def _handle_message(self, websocket: WebSocket, raw: str) -> None:
        """Parse one client message; malformed input is logged."""
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
            asyncio.create_task(
                self._safe_send(
                    websocket,
                    {"t": "pong", "content": {}},
                )
            )
        elif kind in ("pause", "resume"):
            if self.on_pause is not None:
                try:
                    self.on_pause(kind == "pause")
                except Exception as e:
                    logger.error(f"on_pause callback error: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _accept_mid(self, mid: str | None) -> bool:
        """Dedup by mid: duplicates are dropped; the cache is
        bounded, oldest entries evicted."""
        if mid is None:
            return True

        if mid in self._seen_mids:
            logger.info("duplicate input dropped (mid={})", mid)
            return False

        self._seen_mids[mid] = None

        if len(self._seen_mids) > self.dedup_cache_size:
            for key in list(self._seen_mids)[: max(1, self.dedup_cache_size // 2)]:
                self._seen_mids.pop(key, None)

        return True

    def _latest_status(self) -> dict:
        """The most recent status event in history; idle when none."""
        for event in reversed(sink.history()):
            if event.get("t") == "status":
                content = event.get("content") or {}

                return {
                    "state": content.get("state", "idle"),
                }

        return {"state": "idle"}

    async def _send(self, websocket: WebSocket, payload: dict) -> None:
        """Send a message; failures are logged and re-raised."""
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Send failed for {payload.get('t')}: {e}")
            raise

    async def _safe_send(self, websocket: WebSocket, payload: dict) -> None:
        """Send while ignoring errors (ping responses)."""
        with contextlib.suppress(Exception):
            await self._send(websocket, payload)

    def _hello_payload(self) -> dict:
        state = {}
        if self.state_provider is not None:
            try:
                state = self.state_provider() or {}
            except Exception as e:
                logger.error(f"State provider error: {e}")

        # The status snapshot is gateway business: when the
        # app-level state carries none, take the latest from the
        # event history.
        if "status" not in state:
            state["status"] = self._latest_status()

        return {
            "t": "hello",
            "seq": sink.seq,
            "content": state,
        }
