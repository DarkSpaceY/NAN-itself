"""
Record service: the turn archive, outside the prompt.

It registers with the hub in `archive` mode, heartbeats, and
receives each complete turn once that turn ends -- reply, calls,
results, metering, rendered messages, and the death record when
the turn ended the agent. Every delivery is appended to a local
RocksDB store and served back as a trace UI.

Nothing flows from the archive to the agent, by design: record
observes, it does not speak.

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from nan_service_kit import HubLink, Mode, ServiceConfig, read_config, setup_logging

from store import TurnStore

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"

WEB_DIR = HERE / "web"

STORE_DIR = "turns"


class Config(ServiceConfig):
    """
    The archive keys: what to keep, and how the UI pages through it.
    """

    mode: Mode = "archive"

    max_bytes: int = 0

    prune_to: float = 0.95

    batch_size: int = 256

    batch_interval: float = 0.1

    queue_max: int = 10_000

    ledger_limit: int = 300

    activity_buckets: int = 1200


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    return read_config(Config, path)


class RecordService:
    """
    Owns the store, the hub lifecycle and the live fan-out.
    """

    def __init__(
        self,
        config: Config,
        *,
        store_dir: Path | None = None,
        link: HubLink | None = None,
    ) -> None:
        self.config = config

        self.link = link or HubLink(config)

        self._loop: asyncio.AbstractEventLoop | None = None

        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

        self.store = TurnStore(
            store_dir or config.home / STORE_DIR,
            max_bytes=config.max_bytes,
            prune_to=config.prune_to,
            batch_size=config.batch_size,
            batch_interval=config.batch_interval,
            queue_max=config.queue_max,
            notify=self._publish,
        )

    # -- live fan-out ----------------------------------------------------

    def _publish(self, summaries: list[dict[str, Any]]) -> None:
        """Called from the writer thread."""
        loop = self._loop

        if loop is None or loop.is_closed():
            return

        loop.call_soon_threadsafe(self._broadcast, summaries)

    def _broadcast(self, summaries: list[dict[str, Any]]) -> None:
        for subscriber in list(self._subscribers):
            for summary in summaries:
                try:
                    subscriber.put_nowait(summary)

                except asyncio.QueueFull:
                    pass

    async def _stream(self) -> AsyncIterator[str]:
        subscriber: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)

        self._subscribers.add(subscriber)

        try:
            yield "retry: 3000\n\n"

            while True:
                try:
                    summary = await asyncio.wait_for(subscriber.get(), timeout=15.0)

                    yield "event: turn\ndata: "
                    yield json.dumps(summary)
                    yield "\n\n"

                except TimeoutError:
                    yield ": keepalive\n\n"

        finally:
            self._subscribers.discard(subscriber)

    # -- turn endpoint ----------------------------------------------------

    def ingest(self, body: dict[str, Any]) -> None:
        """
        The archive channel only ever delivers a turn that ended,
        as a flat body.
        """
        self.store.ingest(body)


def build_app(service: RecordService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service._loop = asyncio.get_running_loop()

        service.store.start()

        async with service.link:
            logger.info(
                "record archive up on {}:{} (store {})",
                service.config.host,
                service.config.port,
                service.store.path,
            )

            yield

        service.store.stop()

    app = FastAPI(lifespan=lifespan)

    # The desktop hosts this page in a sandboxed iframe without
    # allow-same-origin, so the page runs under an opaque origin and
    # every fetch / EventSource reaches this service as `Origin: null`.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ---------- page freshness ----------
    @app.middleware("http")
    async def revalidate_static(request: Request, call_next: Any) -> Any:
        """
        The UI files are edited in place, so a browser must not reuse
        a copy without asking: heuristic freshness would keep showing
        the previous page.
        """
        response = await call_next(request)

        if not request.url.path.startswith(("/api", "/turn")):
            response.headers["Cache-Control"] = "no-cache"

        return response

    # ---------- archive endpoint ----------
    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        service.ingest(body)

        return {"ok": "true"}

    # ---------- API ----------
    @app.get("/api/agents")
    async def agents() -> dict[str, Any]:
        return {"agents": await asyncio.to_thread(service.store.agents)}

    @app.get("/api/agents/{agent_id}/turns")
    async def turns(agent_id: str, limit: int = 50, before: int | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(service.store.turns, agent_id, limit=limit, before=before)

    @app.get("/api/turns/{agent_id}/{seq}")
    async def one_turn(agent_id: str, seq: int) -> JSONResponse:
        payload = await asyncio.to_thread(service.store.turn, agent_id, seq)

        if payload is None:
            return JSONResponse({"error": "unknown turn"}, status_code=404)

        return JSONResponse(payload)

    @app.get("/api/activity")
    async def activity(
        start: float = 0.0,
        end: float | None = None,
        buckets: int | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            service.store.activity,
            start=start,
            end=end if end is not None else time.time(),
            buckets=max(1, min(buckets or service.config.activity_buckets, 2000)),
        )

    @app.get("/api/ledger")
    async def ledger(
        start: float = 0.0,
        end: float | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            service.store.ledger,
            start=start,
            end=end if end is not None else time.time(),
            limit=min(limit or service.config.ledger_limit, 1000),
            cursor=cursor,
        )

    @app.get("/api/stats")
    async def stats() -> dict[str, Any]:
        return await asyncio.to_thread(service.store.stats)

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        return StreamingResponse(service._stream(), media_type="text/event-stream")

    # ---------- static UI ----------
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


def main() -> None:
    config = load_config()

    setup_logging(config)

    service = RecordService(config)

    uvicorn.run(
        build_app(service),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
