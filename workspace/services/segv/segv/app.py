"""HTTP surface: the trace API and the panel. Protocol routes plug in."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .cache.layer import CacheLayer
from .config import CoreConfig
from .fact.layer import ROUTES, FactLayer
from .store import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

SEARCH_MODES = (*ROUTES, "grep")

FIELD_KEYS = ("origin", "role", "epoch")


def _at(value: Any) -> float | None:
    """Accept epoch seconds or an ISO timestamp; empty means unset."""
    if value is None or value == "":
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    try:
        return float(text)

    except ValueError:
        pass

    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _int_or_none(value: Any) -> int | None:
    return None if value is None or value == "" else int(value)


def _lag(service: Any, agent_id: str) -> tuple[int, int, int]:
    """(newest archived seq, newest indexed seq, turns the index is behind)."""
    meta = service.store.meta(agent_id) or {}

    latest = int(meta.get("next_seq") or 0) - 1

    try:
        indexed = int(service.fact.max_seq(agent_id))

    except Exception:
        indexed = -1

    lag = max(0, latest - indexed) if indexed >= 0 else max(0, latest + 1)

    return latest, indexed, lag


def _in_range(
    hit: dict[str, Any],
    seq_from: int | None,
    seq_to: int | None,
    since: float | None,
    until: float | None,
) -> bool:
    seq = hit.get("seq")

    if seq_from is not None and (seq is None or int(seq) < seq_from):
        return False

    if seq_to is not None and (seq is None or int(seq) > seq_to):
        return False

    started = hit.get("started_at")

    if since is not None or until is not None:
        try:
            stamp = float(started)

        except (TypeError, ValueError):
            return False

        if since is not None and stamp < since:
            return False

        if until is not None and stamp > until:
            return False

    return True


def search_payload(service: Any, body: dict[str, Any]) -> JSONResponse:
    """One entry point for every retrieval mode: grep, lexical, vector, hybrid."""
    agent_id = str(body.get("agent_id") or "").strip()

    query = str(body.get("query") or "").strip()

    if not agent_id or not query:
        return JSONResponse(
            {"error": "agent_id and query are required"}, status_code=400
        )

    mode = str(body.get("mode") or "hybrid").strip().lower()

    if mode not in SEARCH_MODES:
        return JSONResponse(
            {"error": f"unknown search mode: {mode}"}, status_code=400
        )

    rerank = body.get("rerank")

    if rerank is not None and not isinstance(rerank, bool):
        return JSONResponse({"error": "rerank must be a boolean"}, status_code=400)

    try:
        limit = int(body.get("top_k") or body.get("limit") or 10)
        pool = _int_or_none(body.get("pool"))
        before = _int_or_none(body.get("before"))
        seq_from = _int_or_none(body.get("seq_from"))
        seq_to = _int_or_none(body.get("seq_to"))
        epoch = _int_or_none(body.get("epoch"))
        context = int(body.get("context") or 48)
        scan_timeout = float(body.get("scan_timeout") or 3.0)
        since = _at(body.get("since"))
        until = _at(body.get("until"))

    except (TypeError, ValueError):
        return JSONResponse({"error": "numeric filters are invalid"}, status_code=400)

    limit = max(1, min(limit, 100))
    context = min(max(context, 16), 400)
    scan_timeout = min(max(scan_timeout, 0.1), 60.0)

    origin = str(body.get("origin") or "").strip() or None
    role = str(body.get("role") or "").strip().lower() or None

    latest, indexed, lag = _lag(service, agent_id)

    def raw(note: str | None) -> JSONResponse:
        result = service.store.grep(
            agent_id,
            query,
            limit=limit,
            before=before,
            timeout=scan_timeout,
            context=context,
            seq_from=seq_from,
            seq_to=seq_to,
            since=since,
            until=until,
            origin=origin,
            epoch=epoch,
            role=role,
        )

        payload = {
            **result,
            "mode": "grep",
            "count": len(result.get("hits") or []),
            "lag_turns": lag,
            "latest_seq": latest,
            "indexed_seq": indexed,
        }

        if note:
            payload["note"] = note

        return JSONResponse(payload)

    fields = [
        key for key, value in (("origin", origin), ("role", role), ("epoch", epoch)) if value is not None
    ]

    if mode == "grep":
        return raw(None)

    if fields and mode != "grep":
        return raw(
            f"field filters ({', '.join(fields)}) need the raw scan; ran grep instead"
        )

    try:
        hits = service.fact.search(
            agent_id,
            query,
            top_k=limit,
            pool=pool,
            mode=mode,
            rerank=bool(rerank),
        )

    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    hits = [hit for hit in hits if _in_range(hit, seq_from, seq_to, since, until)]

    if not hits and lag > 0:
        return raw(f"index warming (lag={lag}); searched raw instead")

    payload = {
        "mode": mode,
        "hits": hits,
        "count": len(hits),
        "lag_turns": lag,
        "latest_seq": latest,
        "indexed_seq": indexed,
    }

    if lag > 0:
        payload["note"] = f"index warming (lag={lag}); results may miss recent turns"

    return JSONResponse(payload)


class SegvService:
    """Owns the archive store, the fact layer, the cache, and the SSE fan-out."""

    def __init__(
        self,
        config: CoreConfig,
        *,
        links: Any = None,
        fact: Any = None,
        cache: Any = None,
    ) -> None:
        self.config = config

        self.links = list(links or [])

        self.store = Store(
            Path(config.home) / config.data_dir,
            max_bytes=config.max_bytes,
            prune_to=config.prune_to,
            notify=self._on_turn,
        )

        self.fact = fact or FactLayer(self.store, config.fact)

        self.cache = cache or CacheLayer(self.store, self.fact, config.cache)

        self._queues: set[asyncio.Queue[None]] = set()

        self._loop: asyncio.AbstractEventLoop | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        self.fact.start()

        self.cache.start()

    def _on_turn(self, uid: str) -> None:
        """L0 committed a turn: wake the panel and queue derived work."""
        self.publish(uid)

        self.fact.notify(uid)

    # -- SSE ------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, _uid: str) -> None:
        """Called from the write thread; hops back to the event loop."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._fanout)

    def _fanout(self) -> None:
        for queue in list(self._queues):
            if not queue.full():
                queue.put_nowait(None)

    def subscribe(self) -> asyncio.Queue[None]:
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=8)

        self._queues.add(queue)

        return queue

    def unsubscribe(self, queue: asyncio.Queue[None]) -> None:
        self._queues.discard(queue)

    def close(self) -> None:
        self.cache.close()

        self.fact.close()

        self.store.close()


def build_app(
    service: SegvService,
    *,
    routes: Callable[[FastAPI], None] | None = None,
) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.bind_loop(asyncio.get_running_loop())

        service.start()

        logger.info(
            "segv up on {}:{} (store {})",
            service.config.host,
            service.config.port,
            service.store.path,
        )

        try:
            async with contextlib.AsyncExitStack() as stack:
                for link in service.links:
                    await stack.enter_async_context(link)

                yield

        finally:
            service.close()

    app = FastAPI(lifespan=lifespan)

    # The desktop embeds this panel in a sandboxed iframe, which has an
    # opaque origin: its module script and API calls are cross-origin.
    # Without these headers the page loads styled but its JS never runs.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # -- cache ----------------------------------------------------------

    @app.get("/api/cache/state")
    async def cache_state(agent_id: str = "") -> JSONResponse:
        if not agent_id.strip():
            return JSONResponse({"error": "agent_id is required"}, status_code=400)

        return JSONResponse(
            await asyncio.to_thread(service.cache.state, agent_id.strip())
        )

    @app.get("/api/cache/stats")
    async def cache_stats() -> dict[str, Any]:
        return await asyncio.to_thread(service.cache.stats)

    # -- trace API (the record UI contract) ------------------------------

    @app.get("/api/agents")
    async def agents() -> dict[str, Any]:
        return {"agents": await asyncio.to_thread(service.store.agents)}

    @app.get("/api/stats")
    async def stats() -> dict[str, Any]:
        return await asyncio.to_thread(service.store.stats)

    # -- fact ------------------------------------------------------------

    @app.get("/api/fact/stats")
    async def fact_stats() -> dict[str, Any]:
        return await asyncio.to_thread(service.fact.stats)

    @app.post("/api/search")
    async def search(body: dict[str, Any]) -> JSONResponse:
        return await asyncio.to_thread(search_payload, service, body)

    # Kept for scripts written against the recall-only endpoint.
    @app.post("/api/recall")
    async def recall(body: dict[str, Any]) -> JSONResponse:
        return await asyncio.to_thread(search_payload, service, body)

    @app.get("/api/activity")
    async def activity(
        start: float = 0.0,
        end: float | None = None,
        buckets: int | None = None,
    ) -> dict[str, Any]:
        buckets = max(1, min(buckets or service.config.activity_buckets, 2000))

        return await asyncio.to_thread(
            service.store.activity,
            start=start,
            end=end if end is not None else time.time(),
            buckets=buckets,
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

    @app.get("/api/turns/{agent_id}/{seq}")
    async def one_turn(agent_id: str, seq: int) -> JSONResponse:
        payload = await asyncio.to_thread(service.store.turn, agent_id, seq)

        if payload is None:
            return JSONResponse({"error": "unknown turn"}, status_code=404)

        return JSONResponse(payload)

    @app.get("/api/agents/{agent_id}/turns")
    async def agent_turns(
        agent_id: str, limit: int = 50, before: int | None = None
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            service.store.turns, agent_id, limit=min(limit, 500), before=before
        )

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        queue = service.subscribe()

        async def events() -> AsyncIterator[str]:
            try:
                yield ": connected\n\n"

                while True:
                    if await request.is_disconnected():
                        break

                    try:
                        await asyncio.wait_for(queue.get(), timeout=15.0)

                        yield "event: turn\ndata: {}\n\n"

                    except TimeoutError:
                        yield ": keepalive\n\n"

            finally:
                service.unsubscribe(queue)

        return StreamingResponse(events(), media_type="text/event-stream")

    if routes is not None:
        routes(app)

    # Static last: "/" is the fallback, never a shadowed route.
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app
