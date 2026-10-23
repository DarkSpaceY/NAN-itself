"""
Injection: the out-of-process prompt-contributor protocol.

Services are independent local processes. A service registers
itself with the core over HTTP ({service_id, endpoint}), then
heartbeats. Before every prompt assembly the core POSTs the full
new Turn JSON (rendered excluded, keeping the payload small) to
every live service concurrently; each reply's text rides the
turn as an Injection. Failures are recorded in the Turn, which
is a complete runtime log.

The hub routes turns to endpoints and labels what came back.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI
from loguru import logger

from nan.turn import DeltaEvent, Injection, Turn

from .client import InjectionClient

DEFAULT_REQUEST_TIMEOUT = 2.0

DEFAULT_HEARTBEAT_TTL = 15.0


@dataclass
class ServiceEntry:
    service_id: str
    endpoint: str
    last_seen: float


class InjectionHub:
    """
    Registry (register / heartbeat / TTL eviction) + per-agent
    service-delta tracking + the per-turn concurrent collection.
    The HTTP server runs on its own port; collect() is called by
    every agent at every turn boundary.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8901,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        heartbeat_ttl: float = DEFAULT_HEARTBEAT_TTL,
    ) -> None:
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.heartbeat_ttl = heartbeat_ttl

        self._services: dict[str, ServiceEntry] = {}

        # agent_id -> live service ids seen at its last collection
        self._last_reported: dict[str, set[str]] = {}

        self._client = InjectionClient(request_timeout=request_timeout)
        self._server: uvicorn.Server | None = None

    # ------------------------------------------------------------------
    # Registry server
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/register")
        async def register(body: dict[str, Any]) -> dict[str, str]:
            service_id = str(body.get("service_id") or "").strip()

            endpoint = str(body.get("endpoint") or "").strip()

            if not service_id or not endpoint:
                return {"status": "rejected", "reason": "service_id and endpoint required"}

            self._services[service_id] = ServiceEntry(
                service_id=service_id, endpoint=endpoint, last_seen=time.time()
            )

            logger.info("Injection service registered: {} -> {}", service_id, endpoint)

            return {"status": "registered"}

        @app.post("/heartbeat")
        async def heartbeat(body: dict[str, Any]) -> dict[str, str]:
            service_id = str(body.get("service_id") or "")

            entry = self._services.get(service_id)

            if entry is None:
                return {"status": "unknown"}

            entry.last_seen = time.time()

            return {"status": "ok"}

        @app.post("/unregister")
        async def unregister(body: dict[str, Any]) -> dict[str, str]:
            service_id = str(body.get("service_id") or "")

            self._services.pop(service_id, None)

            logger.info("Injection service unregistered: {}", service_id)

            return {"status": "ok"}

        @app.get("/services")
        async def services() -> dict[str, Any]:
            return {
                "services": [
                    {
                        "service_id": e.service_id,
                        "endpoint": e.endpoint,
                        "last_seen": e.last_seen,
                    }
                    for e in self._services.values()
                ]
            }

        return app

    async def start(self) -> None:
        config = uvicorn.Config(
            self._build_app(),
            host=self.host,
            port=self.port,
            log_level="warning",
        )

        self._server = uvicorn.Server(config)

        asyncio.create_task(self._server.serve(), name="injection-hub")

        logger.info("Injection registry on http://{}:{}/register", self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

        await self._client.close()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _live_services(self) -> list[ServiceEntry]:
        now = time.time()

        return [e for e in self._services.values() if now - e.last_seen <= self.heartbeat_ttl]

    async def collect(self, turn: Turn) -> tuple[tuple[Injection, ...], list[DeltaEvent]]:
        """
        Ask every live service for this turn's prompt text, all
        concurrently. Every outcome lands on the record:
        contributions as ok=True, failures as ok=False. Returns
        (contributions, service_events): delta events for services
        that went online/offline since THIS agent's previous
        collection (tracked per agent, empty when nothing changed).
        """
        live = self._live_services()

        live_ids = {e.service_id for e in live}

        events = self._service_delta(turn.agent_id, live_ids)

        if not live:
            return (), events

        contributions = await self._client.collect(turn, live)

        return contributions, events

    def _service_delta(self, agent_id: str, live_ids: set[str]) -> list[DeltaEvent]:
        """
        Diff this agent's previously-seen live set against now.
        Per-agent (not global): every agent reports the full
        transition once from its own point of view.
        """
        prev = self._last_reported.get(agent_id)

        online = live_ids - (prev or set())
        offline = (prev or set()) - live_ids

        self._last_reported[agent_id] = live_ids

        if prev is None and not live_ids:
            return []  # no services registered yet

        events = [DeltaEvent("service", f"+{s} online") for s in sorted(online)]
        events += [DeltaEvent("service", f"-{s} offline") for s in sorted(offline)]

        return events
