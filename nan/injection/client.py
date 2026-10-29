"""
Delivery to local services, in two shapes.

Injection services receive the fresh turn and answer with prompt
text; archive services receive the completed turn and only
acknowledge it. Injection outcomes land on the turn record;
archive failures are logged and dropped, because the archive is
an observer and never fails a turn.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from nan.config import settings
from nan.turn import Injection, Turn

if TYPE_CHECKING:
    from nan.injection.hub import ServiceEntry


class InjectionClient:
    """
    One HTTP client per core process; loopback traffic bypasses
    ambient proxies.

    The reply protocol is one shape: a service answers with a
    JSON object ``{"text": "<prompt text>"}`` (empty text = no
    contribution). Anything else -- non-JSON, wrong shape, non-2xx
    -- is a failed delivery, recorded as ok=False.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=settings.injection.timeout, trust_env=False)

    async def close(self) -> None:
        await self._client.aclose()

    async def inject(
        self,
        turn: Turn,
        services: Sequence[ServiceEntry],
    ) -> tuple[Injection, ...]:
        """
        Concurrently ask every given service for this turn's
        prompt text. Empty contributions are dropped; errors stay
        on the record (ok=False).
        """
        if not services:
            return ()

        payload = turn.to_fresh_json()

        async def ask(entry: ServiceEntry) -> Injection:
            started = time.time()

            try:
                response = await self._client.post(entry.endpoint, json=payload)

                response.raise_for_status()

                reply = response.json()

                if not isinstance(reply, dict) or not isinstance(reply.get("text"), str):
                    raise ValueError('reply must be a JSON object {"text": <str>}')

                return Injection(
                    service_id=entry.service_id,
                    text=reply["text"],
                    started_at=started,
                    ended_at=time.time(),
                    ok=True,
                )

            except Exception as exc:
                return Injection(
                    service_id=entry.service_id,
                    text=f"{type(exc).__name__}: {exc}",
                    started_at=started,
                    ended_at=time.time(),
                    ok=False,
                )

        results = await asyncio.gather(*(ask(entry) for entry in services))

        return tuple(r for r in results if r.ok is False or r.text)

    async def archive(self, turn: Turn, sinks: Sequence[ServiceEntry]) -> None:
        """
        Post the completed turn to every archive service at once.
        A failed delivery is logged once and dropped.
        """
        if not sinks:
            return

        payload = turn.to_json()

        async def send(entry: ServiceEntry) -> None:
            try:
                response = await self._client.post(entry.endpoint, json=payload)

                response.raise_for_status()

            except Exception as exc:
                logger.warning(
                    "Archive delivery to {} failed: {}: {}",
                    entry.service_id,
                    type(exc).__name__,
                    exc,
                )

        await asyncio.gather(*(send(entry) for entry in sinks))
