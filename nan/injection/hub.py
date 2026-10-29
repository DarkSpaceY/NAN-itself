"""
Injection: the out-of-process prompt-contributor protocol.

Services are independent local processes. A service registers
itself with the core over HTTP ({service_id, endpoint, mode}),
then heartbeats. Two modes exist:

    inject    the turn arrives fresh before every prompt assembly;
              the service answers with prompt text
    archive   each turn arrives complete when it ends; the reply
              is an acknowledgement and is never recorded

Both deliveries carry one turn as a flat JSON body.
Both modes appear in the per-agent service delta.
Failures of injection are recorded in the Turn, which is a
complete runtime log.

The hub routes turns to endpoints and labels what came back.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

from nan.config import settings
from nan.turn import DeltaEvent, Injection, Turn
from nan.utils.loopback import entry_allowed, is_loopback_url

from .client import InjectionClient

ServiceMode = Literal["inject", "archive"]


@dataclass
class ServiceEntry:
    service_id: str
    endpoint: str
    mode: ServiceMode
    last_seen: float


class InjectionHub:
    """
    Registry (register / heartbeat / TTL eviction) + per-agent
    service-delta tracking + the two delivery channels. The HTTP
    server runs on its own port; inject() is called at every turn
    boundary, archive() once the turn ends.
    """

    def __init__(self) -> None:
        self._services: dict[str, ServiceEntry] = {}

        # agent_id -> live service ids seen at its last turn boundary
        self._last_reported: dict[str, set[str]] = {}

        self._client = InjectionClient()
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Endpoint / policy (live reads of the global settings)
    # ------------------------------------------------------------------

    @property
    def host(self) -> str:
        return settings.injection.host

    @property
    def port(self) -> int:
        return settings.injection.port

    @property
    def request_timeout(self) -> float:
        return settings.injection.timeout

    @property
    def heartbeat_ttl(self) -> float:
        return settings.injection.heartbeat_ttl

    # ------------------------------------------------------------------
    # Registry server
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        # The registry is loopback-only in both directions: the guard
        # rejects a foreign Host or Origin, and registration refuses
        # non-loopback endpoints.
        @app.middleware("http")
        async def entry_guard_http(request, call_next):
            if not entry_allowed(request.headers, self.port):
                return JSONResponse({"error": "forbidden"}, status_code=403)

            return await call_next(request)

        @app.post("/register")
        async def register(body: dict[str, Any]) -> dict[str, str]:
            service_id = str(body.get("service_id") or "").strip()

            endpoint = str(body.get("endpoint") or "").strip()

            mode_raw = str(body.get("mode") or "inject").strip()

            if not service_id or not endpoint:
                return {"status": "rejected", "reason": "service_id and endpoint required"}

            if mode_raw not in ("inject", "archive"):
                return {"status": "rejected", "reason": "mode must be inject or archive"}

            if not is_loopback_url(endpoint):
                logger.warning(
                    "Injection service rejected (endpoint not loopback): {} -> {}",
                    service_id,
                    endpoint,
                )

                return {"status": "rejected", "reason": "endpoint must be loopback"}

            mode: ServiceMode = "archive" if mode_raw == "archive" else "inject"

            self._services[service_id] = ServiceEntry(
                service_id=service_id,
                endpoint=endpoint,
                mode=mode,
                last_seen=time.time(),
            )

            logger.info("{} service registered: {} -> {}", mode, service_id, endpoint)

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
                        "mode": e.mode,
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

        self._server_task = asyncio.create_task(self._server.serve(), name="injection-hub")

        await self._wait_until_serving()

        logger.info("Injection registry on http://{}:{}/register", self.host, self.port)

    async def _wait_until_serving(self) -> None:
        """
        Return only after the socket is bound; a bind failure
        propagates out of startup.
        """
        assert self._server is not None
        assert self._server_task is not None

        for _ in range(300):
            if self._server.started:
                return

            if self._server_task.done():
                self._server_task.result()

                raise RuntimeError("injection registry exited before serving")

            await asyncio.sleep(0.01)

        raise RuntimeError("injection registry did not start within 3s")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

        if self._server_task is not None:
            server_task, self._server_task = self._server_task, None

            try:
                await server_task

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception("Injection registry termination failed")

        await self._client.close()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def forget(self, agent_id: str) -> None:
        """Drop per-agent tracking; called when the agent exits."""
        self._last_reported.pop(agent_id, None)

    def _live_services(self) -> list[ServiceEntry]:
        now = time.time()

        return [e for e in self._services.values() if now - e.last_seen <= self.heartbeat_ttl]

    async def inject(self, turn: Turn) -> tuple[tuple[Injection, ...], list[DeltaEvent]]:
        """
        Deliver the fresh turn to every live inject service at once.
        Returns (contributions, service_events): the delta covers
        services of either mode and is taken from the same snapshot,
        so a service reported online is a service that received this
        turn. Tracked per agent, empty when nothing changed.
        """
        live = self._live_services()

        events = self._service_delta(turn.agent_id, {e.service_id for e in live})

        contributions = await self._client.inject(turn, [e for e in live if e.mode == "inject"])

        return contributions, events

    async def archive(self, turn: Turn) -> None:
        """
        Deliver the completed turn to every live archive service.
        Nothing comes back: these services record, they do not
        contribute to a prompt.
        """
        await self._client.archive(turn, [e for e in self._live_services() if e.mode == "archive"])

    def _service_delta(self, agent_id: str, live_ids: set[str]) -> list[DeltaEvent]:
        """
        Diff this agent's previously-seen live set against now; the
        diff is per agent, each reporting the full transition from
        its own point of view.
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
