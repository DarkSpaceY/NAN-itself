"""
WS gateway: bridge the EventBus to the UI.

    ws://127.0.0.1:8765/ws

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
from typing import Any, Callable

from loguru import logger
import websockets
from websockets.exceptions import ConnectionClosed

from src.nan_itself.events import EventBus, local_date_label


class Gateway:
    def __init__(
        self,
        *,
        bus: EventBus,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_input: Callable[[str, Any], None] | None = None,
        state_provider: Callable[[], dict] | None = None,
    ) -> None:
        self.bus = bus
        self.host = host
        self.port = port
        self.on_input = on_input
        self.state_provider = state_provider

        self._server: Any = None
        self._clients: set = set()

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def serve(self) -> None:
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
        )

        logger.info(
            "Gateway listening on ws://{}:{}/ws",
            self.host,
            self.port,
        )

    async def close(self) -> None:
        if self._server is None:
            return

        self._server.close()

        try:
            await self._server.wait_closed()
        except Exception:
            pass

        for ws in list(self._clients):
            try:
                await ws.close()
            except Exception:
                pass

        logger.info("Gateway closed")

    # ==================================================================
    # Connection handling
    # ==================================================================

    async def _handler(self, ws: Any) -> None:
        peer = getattr(ws, "remote_address", None)

        self._clients.add(ws)
        queue = self.bus.subscribe()

        logger.info("gateway: client connected {}", peer)

        try:
            await self._send(ws, self._hello_payload())
            await self._send(
                ws,
                {"t": "divider", "label": local_date_label()},
            )

            for event in self.bus.history():
                await self._send_with_date(ws, event)

            forwarder = asyncio.create_task(
                self._forward(ws, queue),
                name="gateway-forward",
            )

            try:
                async for raw in ws:
                    self._handle_message(ws, raw)
            finally:
                forwarder.cancel()

                try:
                    await forwarder
                except (asyncio.CancelledError, Exception):
                    pass

        except ConnectionClosed as exc:
            # Client side went away (tab closed, proxy drop, sleep).
            # A CF/RC frame here would tell us who closed first.
            logger.info(
                "gateway: client {} disconnected ({})",
                peer,
                getattr(exc, "rcvd", None) or getattr(exc, "sent", None) or "closed",
            )

        except Exception:
            logger.exception("Gateway connection error")

        finally:
            self.bus.unsubscribe(queue)
            self._clients.discard(ws)

            logger.info("gateway: client {} cleaned up", peer)

    async def _forward(self, ws: Any, queue: asyncio.Queue) -> None:
        last_date = local_date_label()

        while True:
            event = await queue.get()

            event_date = local_date_label(
                event.get("ts"),
            )

            if event_date != last_date:
                last_date = event_date

                await self._send(
                    ws,
                    {
                        "t": "divider",
                        "label": event_date,
                    },
                )

            await self._send(ws, event)

    def _handle_message(self, ws: Any, raw: Any) -> None:
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
                self._safe_send(ws, {"t": "pong"}),
                name="gateway-pong",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send(self, ws: Any, payload: dict) -> None:
        try:
            await ws.send(
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception:
            pass

    async def _safe_send(self, ws: Any, payload: dict) -> None:
        await self._send(ws, payload)

    async def _send_with_date(self, ws: Any, event: dict) -> None:
        # History replay: date dividers are derived from ts.
        await self._send(ws, event)

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
