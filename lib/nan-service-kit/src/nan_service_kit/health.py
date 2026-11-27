"""
`GET /health`: the one route every service answers identically.

A probe wants a single shape it can compare across processes, so the
answer carries identity and nothing else. Liveness for the hub is the
heartbeat, not this route: `/health` says the listener is up, not that
the link is registered.

FastAPI is not imported here: every service has it, the kit does not,
so `app` is taken as-is and only asked to support `@app.get`.
"""

from __future__ import annotations

from typing import Any

from .config import ServiceConfig


def health_route(app: Any, config: ServiceConfig) -> None:
    """Attach `/health` to a service app."""

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service_id": config.service_id}
