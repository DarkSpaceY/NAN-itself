"""
Record service: the turn archive, outside the prompt.

An injection service in form only -- it registers with the hub,
heartbeats, and answers every turn request, but its reply is
always empty text (dropped by the collector, so it never reaches
the prompt). Its real job is the payload: every collect carries
`prev_turn` complete -- reply, calls, results, metering, rendered
messages -- exactly once per turn. That is appended to a local
RocksDB store and served back as a trace UI.

Nothing flows from the archive to the agent, by design: record
observes, it does not speak.

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from store import TurnStore

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"

WEB_DIR = HERE / "web"


class Config:
    """Typed view of config.yaml; unknown keys are ignored."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.hub_url: str = str(raw.get("hub_url", "http://127.0.0.1:8901")).rstrip("/")

        self.gateway_url: str = str(raw.get("gateway_url", "http://127.0.0.1:8765")).rstrip("/")

        self.service_id: str = str(raw.get("service_id", "record"))

        self.host: str = str(raw.get("host", "127.0.0.1"))

        self.port: int = int(raw.get("port", 9003))

        self.surface_every: float = float(raw.get("surface_every", 10.0))

        self.data_dir: Path = HERE / str(raw.get("data_dir", "data"))

        self.max_bytes: int = int(raw.get("max_bytes", 0))

        self.prune_to: float = float(raw.get("prune_to", 0.95))

        self.batch_size: int = int(raw.get("batch_size", 256))

        self.batch_interval: float = float(raw.get("batch_interval", 0.1))

        self.queue_max: int = int(raw.get("queue_max", 10_000))

        self.ledger_limit: int = int(raw.get("ledger_limit", 300))

        self.activity_buckets: int = int(raw.get("activity_buckets", 1200))


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return Config(raw)


class RecordService:
    """
    Owns the store, the hub lifecycle and the surface heartbeat.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        self._loop: asyncio.AbstractEventLoop | None = None

        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

        self.store = TurnStore(
            config.data_dir,
            max_bytes=config.max_bytes,
            prune_to=config.prune_to,
            batch_size=config.batch_size,
            batch_interval=config.batch_interval,
            queue_max=config.queue_max,
            notify=self._publish,
        )

        self._client = httpx.Client(timeout=5.0)

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

    # -- hub / gateway lifecycle -----------------------------------------

    def _post(self, url: str, body: dict[str, Any]) -> None:
        try:
            self._client.post(url, json=body)

        except Exception:
            logger.warning("{} unreachable", url)

    def register(self) -> None:
        self._post(
            f"{self.config.hub_url}/register",
            {
                "service_id": self.config.service_id,
                "endpoint": f"http://{self.config.host}:{self.config.port}/turn",
            },
        )

    def register_surface(self) -> None:
        self._post(
            f"{self.config.gateway_url}/surfaces",
            {
                "service_id": self.config.service_id,
                "url": f"http://{self.config.host}:{self.config.port}/",
                "title": "Record",
            },
        )

    async def _heartbeat_loop(self) -> None:
        surface_at = 0.0

        while True:
            self._post(
                f"{self.config.hub_url}/heartbeat", {"service_id": self.config.service_id}
            )

            now = time.monotonic()

            if now >= surface_at:
                surface_at = now + self.config.surface_every

                self.register_surface()

            await asyncio.sleep(5.0)

    def unregister(self) -> None:
        self._post(f"{self.config.hub_url}/unregister", {"service_id": self.config.service_id})

        try:
            self._client.delete(f"{self.config.gateway_url}/surfaces/{self.config.service_id}")

        except Exception:
            pass

    # -- turn endpoint ----------------------------------------------------

    def ingest(self, body: dict[str, Any]) -> None:
        """
        Only completed turns are archived. The fresh `turn` is a
        subset of what `prev_turn` will carry one boundary later, so
        recording it would duplicate every turn.
        """
        prev = body.get("prev_turn")

        if not isinstance(prev, dict) or prev.get("ended_at") is None:
            return

        self.store.ingest(prev)


def build_app(service: RecordService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service._loop = asyncio.get_running_loop()

        service.store.start()

        service.register()

        beater = asyncio.create_task(service._heartbeat_loop(), name="record-heartbeat")

        logger.info(
            "record service up on {}:{} (store {})",
            service.config.host,
            service.config.port,
            service.config.data_dir,
        )

        yield

        beater.cancel()

        service.unregister()

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

    # ---------- injection endpoint ----------
    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        service.ingest(body)

        return {"text": ""}

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

    config.data_dir.mkdir(parents=True, exist_ok=True)

    # The log file is owned by the boot script that launches this
    # process (stdout/stderr are redirected there); logging to the
    # same path here would put two writers on one rotating file.
    service = RecordService(config)

    uvicorn.run(
        build_app(service),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
