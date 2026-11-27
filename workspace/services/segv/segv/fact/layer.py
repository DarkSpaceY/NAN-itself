"""The fact layer: worker, ingestion, and four-route recall.

Ingestion is off the archive path. `/turn` writes L0 and returns; this
layer reads the message run from L0, builds chunks, cues and vectors, and
commits them with a marker. Recall fuses four routes with the frozen v2
weights, reranks the head with the cross-encoder, and fuses again.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ..store import Store, turn_uid
from .chunk import ChunkDraft, build_chunks
from .config import FactConfig
from .cue import CueExtractor
from .embed import Embedder
from .fusion import fuse_rrf
from .slm import Slm
from .store import ChunkRow, FactStore, quantize
from .text import event_doc, raw_doc

STOP = object()

ROUTES: dict[str, tuple[str, ...]] = {
    "hybrid": ("raw_bm25", "raw_dense", "event_dense", "event_bm25"),
    "raw": ("raw_bm25", "raw_dense"),
    "event": ("event_dense", "event_bm25"),
    "bm25": ("raw_bm25", "event_bm25"),
    "dense": ("raw_dense", "event_dense"),
    # The CLI's names for the same two route pairs.
    "lexical": ("raw_bm25", "event_bm25"),
    "vector": ("raw_dense", "event_dense"),
}


def _roles(chunk: ChunkDraft) -> str:
    return ", ".join(dict.fromkeys(chunk.roles))


class FactLayer:
    def __init__(
        self,
        l0: Store,
        config: FactConfig,
        *,
        slm: Any = None,
        embedder: Any = None,
        reranker: Any = None,
    ) -> None:
        self.l0 = l0
        self.config = config
        self.enabled = config.enabled

        if config.rerank.enabled and config.rerank.candidates > config.fusion.pool:
            raise ValueError(
                "rerank.candidates must not exceed fusion.pool: "
                f"{config.rerank.candidates} > {config.fusion.pool}"
            )

        self.store: FactStore | None = None
        self.slm: Any = slm
        self.embedder: Any = embedder
        self.cue: CueExtractor | None = None

        self._reranker = reranker
        self._reranker_failed = False
        self._reranker_lock = threading.Lock()

        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._backfiller: threading.Thread | None = None
        self._stop = threading.Event()

        if not self.enabled:
            return

        self.store = FactStore(
            Path(l0.path) / config.data_dir,
            dim=config.embed.dim,
            raw_bm25=config.raw_bm25,
            event_bm25=config.event_bm25,
            connectivity=config.hnsw_connectivity,
        )

        self.slm = slm or Slm(config.slm)
        self.embedder = embedder or Embedder(config.embed)
        self.cue = CueExtractor(config.cue, self.slm)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the ingest worker and queue L0 turns fact has not seen."""
        if not self.enabled or self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run, name="segv-fact", daemon=True)
        self._thread.start()

        # The scan is O(L0) and must never hold up service startup.
        self._backfiller = threading.Thread(
            target=self._backfill, name="segv-fact-backfill", daemon=True
        )
        self._backfiller.start()

        # The CE loads lazily; warm it now so the first search after boot
        # does not block on a multi-second model load.
        threading.Thread(target=self.warm, name="segv-fact-warm", daemon=True).start()

    def warm(self) -> None:
        """Load the reranker ahead of the first search."""
        if not self.enabled or not self.config.rerank.enabled:
            return

        started = time.monotonic()

        if self._ensure_reranker() is not None:
            logger.info(
                "segv-fact reranker warmed in {:.1f}s", time.monotonic() - started
            )

    def _backfill(self) -> None:
        try:
            for agent_id, seq in self.l0.iter_turns():
                if self._stop.is_set():
                    return

                if self.store is not None and not self.store.has_turn(agent_id, seq):
                    self.enqueue(agent_id, seq)

        except Exception:
            logger.exception("segv-fact backfill scan failed")

    def close(self) -> None:
        self._stop.set()

        if self._backfiller is not None:
            self._backfiller.join(timeout=2.0)
            self._backfiller = None

        if self._thread is not None:
            self._queue.put(STOP)
            self._thread.join(timeout=10.0)
            self._thread = None

        for client in (self.slm, self.embedder):
            close = getattr(client, "close", None)

            if callable(close):
                close()

        if self.store is not None:
            self.store.close()
            self.store = None

    def notify(self, uid: str) -> None:
        """L0 callback: queue the committed turn, never block the write."""
        if not self.enabled or self._thread is None:
            return

        found = self.l0.seq_of(uid)

        if found is None:
            return

        self.enqueue(*found)

    def enqueue(self, agent_id: str, seq: int) -> None:
        if self.enabled and not self._stop.is_set():
            self._queue.put((agent_id, seq))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)

            except queue.Empty:
                continue

            if item is STOP or item is None:
                break

            try:
                self.ingest(*item)

            except Exception:
                logger.exception("segv-fact ingest failed for {}/{}", *item)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, agent_id: str, seq: int) -> int:
        """Build and commit one turn's facts; idempotent by marker."""
        if self.store is None:
            return 0

        if self.store.has_turn(agent_id, seq):
            return 0

        payload = self.l0.turn(agent_id, seq)

        if payload is None:
            logger.warning("segv-fact skip {}/{}: turn was pruned or missing", agent_id, seq)

            return 0

        ts, chunks = build_chunks(payload, self.config.chunk)

        if not chunks:
            self.store.add_turn(agent_id, seq, [])

            return 0

        self._fill_cues(ts, chunks)

        events = [chunk.cue.strip() for chunk in chunks]

        raw_docs = []

        for index, chunk in enumerate(chunks):
            previous = chunks[index - 1].text if index > 0 else ""
            following = chunks[index + 1].text if index + 1 < len(chunks) else ""
            raw_docs.append(
                self.config.embed.doc_prefix
                + raw_doc(
                    "",
                    "",
                    chunk.text,
                    previous,
                    following,
                    self.config.chunk.ctx_chars,
                )
            )

        event_docs = [
            self.config.embed.doc_prefix
            + event_doc(
                events[index - 1] if index > 0 else "",
                event,
                events[index + 1] if index + 1 < len(events) else "",
                self.config.chunk.ctx_chars,
            )
            if event
            else ""
            for index, event in enumerate(events)
        ]

        pending = [document for document in event_docs if document]
        vectors = self.embedder.embed(raw_docs + pending)

        if len(vectors) != len(raw_docs) + len(pending):
            logger.warning(
                "segv-fact embed count {}/{} for {}/{}",
                len(vectors),
                len(raw_docs) + len(pending),
                agent_id,
                seq,
            )

            return 0

        raw_vectors = vectors[: len(chunks)]
        event_vectors = iter(vectors[len(chunks) :])

        started_at = float(payload.get("started_at") or 0.0)
        uid = str(payload.get("turn_id") or turn_uid(agent_id, started_at))

        rows: list[ChunkRow] = []

        for index, chunk in enumerate(chunks):
            event = events[index]

            rows.append(
                ChunkRow(
                    chunk_id=f"{uid}#c{index}",
                    ordinal=index,
                    started_at=started_at,
                    ts=ts,
                    text=chunk.text,
                    event=event,
                    raw_vec=quantize(raw_vectors[index]),
                    event_vec=quantize(next(event_vectors)) if event else None,
                )
            )

        self.store.add_turn(agent_id, seq, rows)

        return len(rows)

    def _fill_cues(self, ts: str, chunks: list[ChunkDraft]) -> None:
        """One cue per chunk: cue, chunk and retrieval stay 1:1."""
        if self.cue is None:
            return

        tasks = []

        for index, chunk in enumerate(chunks):
            previous = chunks[index - 1] if index > 0 else None
            following = chunks[index + 1] if index + 1 < len(chunks) else None

            tasks.append(
                (
                    ts,
                    _roles(chunk),
                    chunk.text,
                    (_roles(previous), previous.text) if previous else None,
                    (_roles(following), following.text) if following else None,
                )
            )

        for chunk, cue in zip(
            chunks,
            self.cue.extract_many(tasks, self.config.cue.workers),
            strict=True,
        ):
            chunk.cue = cue

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def search(
        self,
        agent_id: str,
        query: str,
        *,
        top_k: int = 20,
        pool: int | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
        exclude: set[tuple[int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse the routes `mode` selects, dropping already-injected chunks.

        `exclude` is keyed by (seq, ordinal): the cache layer's seen set.
        The pool is fixed, so an exhausted pool returns fewer hits rather
        than shifting the search window outward.
        """
        if self.store is None or not query.strip():
            return []

        routes = ROUTES.get(mode)

        if routes is None:
            raise ValueError(f"unknown search mode: {mode}")

        top_k = max(1, min(int(top_k), 100))
        pool = max(top_k, min(int(pool or self.config.fusion.pool), 1000))
        excluded = {tuple(item) for item in (exclude or ())}
        loc = self.store.loc_map(agent_id) if excluded else {}

        def visible(ids: list[str]) -> list[str]:
            return [chunk_id for chunk_id in ids if loc.get(chunk_id) not in excluded]

        query_vector: np.ndarray | None = None

        if "raw_dense" in routes or "event_dense" in routes:
            query_vector = quantize(
                self.embedder.embed_one(self.config.embed.query_prefix + query)
            )

        sources = {
            "raw_bm25": lambda: self.store.search_raw_bm25(agent_id, query, pool),
            "raw_dense": lambda: self.store.search_raw_dense(
                agent_id, query_vector, pool
            ),
            "event_dense": lambda: self.store.search_event_dense(
                agent_id, query_vector, pool
            ),
            "event_bm25": lambda: self.store.search_event_bm25(agent_id, query, pool),
        }

        rankings = [(name, visible(sources[name]())) for name in routes]

        seq = self.store.seq_map(agent_id)
        weights = dict(self.config.fusion.weights)
        order = [name for name in self.config.fusion.order if name in routes]
        k = self.config.fusion.k
        use_rerank = (
            self.config.rerank.enabled if rerank is None else bool(rerank)
        ) and self.config.rerank.enabled

        if use_rerank:
            pre = fuse_rrf(
                rankings,
                weights,
                k=k,
                seq=seq,
                top_k=self.config.rerank.candidates,
                order=order,
            )

            try:
                reranked = self._rerank(query, pre)

            except Exception:
                logger.exception("segv-fact rerank failed; fusing without the CE route")

                reranked = []

            if reranked:
                rankings.append(("ce", reranked))
                weights["ce"] = float(self.config.rerank.weight)
                order.append("ce")

        fused = fuse_rrf(rankings, weights, k=k, seq=seq, top_k=pool, order=order)

        if excluded:
            fused = [item for item in fused if loc.get(item[0]) not in excluded]

        fused = fused[:top_k]
        rows = self.store.get_chunks([chunk_id for chunk_id, _ in fused])

        hits: list[dict[str, Any]] = []

        for chunk_id, score in fused:
            row = rows.get(chunk_id)

            if row is None:
                continue

            hits.append(
                {
                    "chunk_id": chunk_id,
                    "agent_id": row["agent_id"],
                    "seq": row["seq"],
                    "ordinal": row["ordinal"],
                    "started_at": row["started_at"],
                    "ts": row["ts"],
                    "text": row["text"],
                    "event": row["event"],
                    "score": score,
                }
            )

        return hits

    def max_seq(self, agent_id: str) -> int:
        """The newest turn seq present in the index; -1 when it holds none."""
        if self.store is None:
            return -1

        return max(self.store.seq_map(agent_id).values(), default=-1)

    def _rerank(self, query: str, candidates: list[tuple[str, float]]) -> list[str]:
        """Order the RRF head with the cross-encoder, as a fifth route."""
        reranker = self._ensure_reranker()

        if reranker is None or not candidates:
            return []

        rows = self.store.get_chunks([chunk_id for chunk_id, _ in candidates]) if self.store else {}
        ids = [chunk_id for chunk_id, _ in candidates if chunk_id in rows]

        if not ids:
            return []

        scores = reranker.score(query, [rows[chunk_id]["text"] for chunk_id in ids])

        return [
            chunk_id
            for chunk_id, _ in sorted(
                zip(ids, scores, strict=True), key=lambda item: -item[1]
            )
        ]

    def _ensure_reranker(self) -> Any:
        if not self.config.rerank.enabled or self._reranker_failed:
            return None

        if self._reranker is not None:
            return self._reranker

        with self._reranker_lock:
            if self._reranker is not None or self._reranker_failed:
                return self._reranker

            try:
                from .rerank import Reranker

                self._reranker = Reranker(
                    self.config.rerank.model,
                    device=self.config.rerank.device,
                    fp16=self.config.rerank.fp16,
                    batch_size=self.config.rerank.batch_size,
                    max_length=self.config.rerank.max_length,
                )

            except Exception:
                logger.exception(
                    "segv-fact reranker unavailable; recall will fuse without it"
                )

                self._reranker_failed = True

        return self._reranker

    def stats(self) -> dict[str, Any]:
        if self.store is None:
            return {"enabled": False, "pending": 0}

        return {
            "enabled": True,
            **self.store.stats(),
            "pending": self._queue.qsize(),
            "reranker": "loaded"
            if self._reranker is not None
            else ("failed" if self._reranker_failed else "lazy"),
        }
