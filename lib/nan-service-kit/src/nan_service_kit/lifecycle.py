"""
A service's link to the core: register once when it comes up, prove
liveness while it runs, and unregister when it goes.

Every post is best effort. A service with no hub still serves its own
API and answers its own panel; it just receives no turns until the hub
is reachable, which the heartbeat retries.

Registration is what the hub needs to route turns; a heartbeat only
refreshes the TTL. A hub that restarted -- or started after the
service -- has forgotten the id, and says so: that reply re-registers.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import httpx
from loguru import logger

from .config import ServiceConfig

HEARTBEAT_EVERY = 5.0  # well under the hub's heartbeat_ttl (15s default)


class HubLink:
    """
    The hub/gateway side of one service process. Posts are awaited, so
    a slow or absent hub never blocks the event loop; failures are
    logged and dropped.
    """

    def __init__(
        self,
        config: ServiceConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config

        # Loopback traffic bypasses ambient proxies and never leaves
        # the machine through one.
        self._http = httpx.AsyncClient(
            timeout=5.0,
            trust_env=False,
            transport=transport,
        )

        self._task: asyncio.Task[None] | None = None

        self._surface_at = 0.0

    # ------------------------------------------------------------------
    # Posts
    # ------------------------------------------------------------------

    async def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = await self._http.post(url, json=body)

            response.raise_for_status()

            payload = response.json()

        except Exception as exc:
            logger.warning("{} failed: {}: {}", url, type(exc).__name__, exc)

            return None

        return payload if isinstance(payload, dict) else None

    async def _delete(self, url: str) -> None:
        try:
            response = await self._http.delete(url)

            response.raise_for_status()

        except Exception as exc:
            logger.warning("{} failed: {}: {}", url, type(exc).__name__, exc)

    # ------------------------------------------------------------------
    # Hub
    # ------------------------------------------------------------------

    async def register(self) -> None:
        """
        Announce this service's endpoint and mode. The hub refuses a
        non-loopback endpoint, so a rejection is a real misconfiguration
        and worth an error rather than a warning.
        """
        reply = await self._post(
            f"{self.config.hub_url}/register",
            {
                "service_id": self.config.service_id,
                "endpoint": self.config.endpoint,
                "mode": self.config.mode,
            },
        )

        if reply is not None and reply.get("status") == "rejected":
            logger.error(
                "hub rejected {}: {}", self.config.service_id, reply.get("reason", "no reason")
            )

    async def unregister(self) -> None:
        await self._post(
            f"{self.config.hub_url}/unregister", {"service_id": self.config.service_id}
        )

        if self.config.surface is not None:
            await self._delete(f"{self.config.gateway_url}/surfaces/{self.config.service_id}")

    # ------------------------------------------------------------------
    # Gateway panel
    # ------------------------------------------------------------------

    async def announce(self) -> None:
        """
        Declare the panel and prove this service is alive in one post:
        the gateway drops a surface not reposted within its TTL.
        """
        surface = self.config.surface

        if surface is None:
            return

        body: dict[str, Any] = {
            "service_id": self.config.service_id,
            "url": surface.url or f"http://{self.config.host}:{self.config.port}/",
            "title": surface.title,
        }

        if surface.icon:
            body["icon"] = surface.icon

        await self._post(f"{self.config.gateway_url}/surfaces", body)

    # ------------------------------------------------------------------
    # Background beat
    # ------------------------------------------------------------------

    async def beat(self) -> None:
        """
        One liveness step: heartbeat, register again when the hub has
        forgotten this id, and repost the panel when its TTL is due.
        """
        reply = await self._post(
            f"{self.config.hub_url}/heartbeat", {"service_id": self.config.service_id}
        )

        if reply is not None and reply.get("status") == "unknown":
            logger.info("hub lost {}, registering again", self.config.service_id)

            await self.register()

        if self.config.surface is not None and time.monotonic() >= self._surface_at:
            self._surface_at = time.monotonic() + self.config.surface.every

            await self.announce()

    async def _beat(self) -> None:
        while True:
            await self.beat()

            await asyncio.sleep(HEARTBEAT_EVERY)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> HubLink:
        await self.register()

        self._task = asyncio.create_task(self._beat(), name=f"{self.config.service_id}-hub")

        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def stop(self) -> None:
        """Cancel the beat, unregister, close the client."""
        if self._task is not None:
            task, self._task = self._task, None

            task.cancel()

            with contextlib.suppress(asyncio.CancelledError):
                await task

        await self.unregister()

        await self._http.aclose()
