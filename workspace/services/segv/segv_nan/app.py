"""NAN's two hub routes on top of the core app."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from loguru import logger

from segv.app import SegvService

from .protocol import inject_request, turn_record


def install(app: FastAPI, service: SegvService) -> None:
    """Mount /turn and /inject before the panel's catch-all."""

    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        # Durable before the ack: the hub drops an archive failure, so a
        # 2xx that lost the turn would lose memory silently.
        record = turn_record(body)

        if record is not None:
            await asyncio.to_thread(service.store.append, record)

        return {"status": "ok"}

    @app.post("/inject")
    async def inject(body: dict[str, Any]) -> dict[str, str]:
        # The hub treats a failure as "service unavailable" and moves on.
        # A degraded block is more useful than a 500, so this never raises.
        try:
            agent_id, epoch, query = inject_request(body)

            text = await asyncio.to_thread(
                service.cache.inject, agent_id, epoch, query
            )

        except Exception:
            logger.exception("segv-nan inject failed")

            text = ""

        return {"text": text}
