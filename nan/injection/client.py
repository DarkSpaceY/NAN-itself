"""
The per-turn collector: POST the turn payload to every live
service concurrently, normalize every outcome onto the record.

Failures never block and never retry: a failed service shows up
as an Injection with ok=False, exactly like a silent one would
not.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx

from nan.turn import Injection, Turn

if TYPE_CHECKING:
    from nan.injection.hub import ServiceEntry


class InjectionClient:
    """
    One HTTP client per core process; loopback traffic bypasses
    ambient proxies.
    """

    def __init__(self, *, request_timeout: float = 2.0) -> None:
        self._client = httpx.AsyncClient(timeout=request_timeout, trust_env=False)

    async def close(self) -> None:
        await self._client.aclose()

    async def collect(
        self, turn: Turn, services: Sequence[ServiceEntry]
    ) -> tuple[Injection, ...]:
        """
        Concurrently ask every given service for this turn's
        prompt text. Empty contributions are dropped; errors stay
        on the record (ok=False).
        """
        if not services:
            return ()

        payload = {"turn": turn.to_json()}

        async def ask(entry: ServiceEntry) -> Injection:
            started = time.time()

            try:
                response = await self._client.post(entry.endpoint, json=payload)

                response.raise_for_status()

                text = _extract_text(response)

                return Injection(
                    service_id=entry.service_id,
                    text=text,
                    ok=True,
                    duration_s=time.time() - started,
                )

            except Exception as exc:
                return Injection(
                    service_id=entry.service_id,
                    text=f"{type(exc).__name__}: {exc}",
                    ok=False,
                    duration_s=time.time() - started,
                )

        results = await asyncio.gather(*(ask(entry) for entry in services))

        return tuple(r for r in results if r.ok is False or r.text)


def _extract_text(response: httpx.Response) -> str:
    """
    A service may answer plain text or JSON {"text": ...}.
    """
    content_type = response.headers.get("content-type", "")

    if "application/json" in content_type:
        body = response.json()

        if isinstance(body, dict):
            return str(body.get("text") or "")

        return str(body)

    return response.text
