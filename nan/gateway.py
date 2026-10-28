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

Surfaces: a service POSTs its own panel spec as a heartbeat; the
gateway keeps a {service_id: (spec, last_seen)} table, broadcasts
surface_added/surface_removed to every connected desktop, replays
the table on each new connection, and drops entries whose
heartbeat stops. Registration is kept outside the event log;
panel urls are loopback.
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

from .config import settings
from .events import sink
from .utils.loopback import entry_allowed, is_local_asset, is_loopback_url


class Gateway:
    def __init__(
        self,
        *,
        on_input: Callable[[str, Any], None] | None = None,
        on_pause: Callable[[bool], None] | None = None,
        state_provider: Callable[[], dict] | None = None,
    ) -> None:
        # Endpoint and layout come from the process-wide settings;
        # callbacks are injected.
        self.on_input = on_input
        self.on_pause = on_pause
        self.state_provider = state_provider

        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None
        self._clients: set[WebSocket] = set()

        # service_id -> (surface spec, last heartbeat); entries whose
        # heartbeat stops expire after surface_ttl.
        self._surfaces: dict[str, tuple[dict[str, Any], float]] = {}

        # Input dedup by mid: seen mids are dropped; the cache is
        # bounded and evicts oldest entries first.
        self._seen_mids: dict[str, None] = {}

    # ------------------------------------------------------------------
    # Endpoint / layout (live reads of the global settings)
    # ------------------------------------------------------------------

    @property
    def host(self) -> str:
        return settings.gateway.host

    @property
    def port(self) -> int:
        return settings.gateway.port

    @property
    def dedup_cache_size(self) -> int:
        return settings.gateway.dedup_cache_size

    @property
    def surface_ttl(self) -> float:
        return settings.gateway.surface_ttl

    @property
    def frontend_dir(self) -> Path | None:
        """
        The built desktop at frontend/dist; None when index.html is missing.
        """
        built = settings.paths.frontend

        if (built / "index.html").is_file():
            return built

        logger.warning("frontend build not found ({}); run `npm run build` in frontend/", built)

        return None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def serve(self, open_browser: bool = True) -> None:
        """Start the HTTP + WebSocket server; open the browser when asked."""
        app = FastAPI()

        # ---------- Entry guard (loopback policy) ----------
        @app.middleware("http")
        async def entry_guard_http(request, call_next):
            if not entry_allowed(request.headers, self.port):
                return JSONResponse({"error": "forbidden"}, status_code=403)

            return await call_next(request)

        # ---------- WebSocket endpoint ----------
        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            if not entry_allowed(websocket.headers, self.port):
                await websocket.close(code=1008)
                return

            await websocket.accept()
            await self._ws_handler(websocket)

        # ---------- Surface registration ----------
        # POST is a heartbeat: the same spec refreshes last_seen, a
        # changed spec updates the panel in place, and a stopped
        # heartbeat expires it.
        @app.post("/surfaces")
        async def register_surface(body: dict[str, Any]) -> JSONResponse:
            spec, error = _validate_surface(body)

            if spec is None:
                return JSONResponse({"status": "rejected", "reason": error}, status_code=400)

            service_id = spec["service_id"]

            previous = self._surfaces.get(service_id)

            self._surfaces[service_id] = (spec, asyncio.get_running_loop().time())

            if previous is not None and previous[0] == spec:
                return JSONResponse({"status": "ok"})

            # A changed spec is an update of the same window, not a
            # withdrawal plus a new one: the desktop replaces the
            # spec under the same id, so the panel is not torn down
            # and rebuilt (or flashed out of its queue slot) just
            # because the service restarted on another port.
            logger.info("Surface registered: {} -> {}", service_id, spec["url"])

            await self._broadcast({"t": "surface_added", "content": spec})

            return JSONResponse({"status": "ok"})

        @app.delete("/surfaces/{service_id}")
        async def unregister_surface(service_id: str) -> JSONResponse:
            if self._surfaces.pop(service_id, None) is None:
                return JSONResponse({"status": "unknown"}, status_code=404)

            logger.info("Surface unregistered: {}", service_id)

            await self._broadcast({"t": "surface_removed", "content": {"service_id": service_id}})

            return JSONResponse({"status": "ok"})

        # ---------- Frontend static files ----------
        frontend_dir = self.frontend_dir

        if frontend_dir is not None:
            assets_dir = frontend_dir / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

            @app.get("/{full_path:path}")
            async def serve_spa(full_path: str):
                index_path = frontend_dir / "index.html"
                if index_path.exists():
                    return FileResponse(str(index_path))
                return {"error": "Frontend not built"}

            logger.info(f"Frontend static files served from {frontend_dir}")
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

        server_task = asyncio.create_task(self._server.serve())

        reaper = asyncio.create_task(self._reap_surfaces(), name="surface-reaper")

        try:
            if open_browser:
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

            await server_task

        finally:
            reaper.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await reaper

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

    async def _reap_surfaces(self) -> None:
        """Drop panels whose heartbeat stopped and withdraw them."""
        interval = max(1.0, self.surface_ttl / 3)

        while True:
            await asyncio.sleep(interval)

            cutoff = asyncio.get_running_loop().time() - self.surface_ttl

            expired = [sid for sid, (_, seen) in self._surfaces.items() if seen < cutoff]

            for service_id in expired:
                self._surfaces.pop(service_id, None)

                logger.info("Surface expired: {}", service_id)

                await self._broadcast(
                    {"t": "surface_removed", "content": {"service_id": service_id}}
                )

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
            # History is replayed after hello; date dividers are
            # rendered by the frontend from event timestamps.
            await self._send(websocket, self._hello_payload())

            # Replay the surface table to each new connection.
            for spec, _ in self._surfaces.values():
                await self._send(websocket, {"t": "surface_added", "content": spec})

            for event in sink.history():
                await self._send(websocket, event)

            forwarder = asyncio.create_task(
                self._forward(websocket, queue, client_id),
                name=f"forward-{client_id}",
            )

            monitor = asyncio.create_task(
                self._monitor_forwarder(websocket, forwarder, client_id),
                name=f"monitor-{client_id}",
            )

            try:
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
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                except TimeoutError:
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

    def _latest_paused(self) -> bool:
        """Pause state from the newest pause/resume event in history."""
        for event in reversed(sink.history()):
            if event.get("t") == "agent_paused":
                return True

            if event.get("t") == "agent_resumed":
                return False

        return False

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

    async def _broadcast(self, payload: dict) -> None:
        """Send to every connected desktop; a dead socket is dropped."""
        for websocket in list(self._clients):
            try:
                await self._send(websocket, payload)
            except Exception:
                self._clients.discard(websocket)

    def _hello_payload(self) -> dict:
        state = {}
        if self.state_provider is not None:
            try:
                state = self.state_provider() or {}
            except Exception as e:
                logger.error(f"State provider error: {e}")

        # paused is derived from event history when the app-level state omits it.
        if "paused" not in state:
            state["paused"] = self._latest_paused()

        return {
            "t": "hello",
            "seq": sink.seq,
            "content": state,
        }


def _validate_surface(body: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """
    A surface spec names a service_id, a loopback url and a title;
    icon is a local path or loopback url.
    """
    service_id = str(body.get("service_id") or "").strip()

    url = str(body.get("url") or "").strip()

    title = str(body.get("title") or "").strip()

    icon = str(body.get("icon") or "").strip()

    if not service_id or not url or not title:
        return None, "service_id, url and title required"

    if not is_loopback_url(url):
        return None, "url must be loopback"

    if icon and not is_local_asset(icon):
        return None, "icon must be a local path or loopback url"

    spec: dict[str, Any] = {"service_id": service_id, "url": url, "title": title}

    if icon:
        spec["icon"] = icon

    return spec, None
