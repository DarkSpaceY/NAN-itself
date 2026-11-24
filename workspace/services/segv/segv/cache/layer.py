"""The cache layer: pending cues, background rewrites, and injection.

A rewrite runs off the inject path. The inject path only reads the last
committed snapshot, the tail after its pointer, and one fact recall; the
per-(agent, epoch) seen set makes sure a recall never repeats.
"""

from __future__ import annotations

import bisect
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ..fact.layer import FactLayer
from ..fact.slm import Slm
from ..store import Store
from .config import CacheConfig
from .prompts import GUIDE, rewrite_prompt
from .snapshot import Snapshot, SnapshotError, parse_snapshot
from .store import CacheState, CacheStore

STOP = object()

# The chunk text keeps the prompt envelope; it is structure, not a fact.
ENVELOPE = re.compile(r"</?(?:inbox|user)(?:\s[^>]*)?>", re.IGNORECASE)


@dataclass(frozen=True)
class CueRef:
    seq: int
    ordinal: int
    chunk_id: str
    ts: str
    text: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.seq, self.ordinal)


def render_refs(tag: str, refs: list[CueRef]) -> str:
    if not refs:
        return ""

    lines = []

    for ref in refs:
        # Cue text already starts with its timestamp; do not repeat it.
        stamp = f"{ref.ts} " if ref.ts and not ref.text.startswith(ref.ts) else ""
        lines.append(f"- [{ref.seq}:{ref.ordinal}] {stamp}{ref.text}")

    return f"<{tag}>\n" + "\n".join(lines) + f"\n</{tag}>"


def _tail(
    pending: list[CueRef],
    pointer: tuple[int, int] | None,
    max_cues: int,
    max_chars: int,
) -> list[CueRef]:
    """Cues after the pointer, keeping the newest ones inside both caps."""
    refs = [
        ref
        for ref in pending
        if pointer is None or (ref.seq, ref.ordinal) > pointer
    ]
    refs = refs[-max(0, max_cues) :] if max_cues else []
    out: list[CueRef] = []
    size = 0

    for ref in reversed(refs):
        if out and size + len(ref.text) > max_chars:
            break

        out.append(ref)
        size += len(ref.text)

    out.reverse()

    return out


def _epoch(value: Any) -> int:
    try:
        return int(value or 0)

    except (TypeError, ValueError):
        return 0


def _inbox_query(body: dict[str, Any]) -> str:
    parts: list[str] = []

    for message in body.get("inbox") or []:
        if not isinstance(message, dict):
            continue

        text = str(message.get("text") or "").strip()

        if text:
            parts.append(text)

    return "\n".join(parts)[-2000:].strip()


class CacheLayer:
    def __init__(
        self,
        l0: Store,
        fact: FactLayer,
        config: CacheConfig,
        *,
        slm: Any = None,
        store: CacheStore | None = None,
    ) -> None:
        self.l0 = l0
        self.fact = fact
        self.config = config

        self.enabled = bool(
            config.enabled
            and getattr(fact, "enabled", False)
            and getattr(fact, "store", None) is not None
        )

        self.store = (
            store or CacheStore(Path(l0.path) / config.data_dir)
            if self.enabled
            else None
        )
        self._owns_store = store is None

        self.slm = slm or (Slm(config.slm) if self.enabled else None)
        self._owns_slm = slm is None

        self._lock = threading.RLock()
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

        self._pending: dict[str, list[CueRef]] = {}
        self._have: dict[str, set[tuple[int, int]]] = {}
        self._queued: set[str] = set()

        self._queue: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run, name="segv-cache", daemon=True
        )
        self._thread.start()

        self._rebuild()
        self._trigger_all()

    def close(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._queue.put(STOP)
            self._thread.join(timeout=10.0)
            self._thread = None

        if self._owns_slm:
            close = getattr(self.slm, "close", None)

            if callable(close):
                close()

        if self._owns_store and self.store is not None:
            self.store.close()

    def _rebuild(self) -> None:
        store = getattr(self.fact, "store", None)

        if self.store is None or store is None:
            return

        try:
            agents = list(store.iter_agents())

        except Exception:
            logger.exception("segv-cache could not list agents")

            return

        for agent_id in agents:
            try:
                state = self.store.load(agent_id)
                pointer = state.pointer if state is not None else None
                refs: list[CueRef] = []

                for row in store.iter_chunks(agent_id, after=pointer):
                    text = str(row.get("event") or "").strip()

                    if not text:
                        text = str(row.get("text") or "").strip()[:240]

                    if text:
                        refs.append(
                            CueRef(
                                seq=int(row["seq"]),
                                ordinal=int(row["ordinal"]),
                                chunk_id=str(row["chunk_id"]),
                                ts=str(row.get("ts") or ""),
                                text=text,
                            )
                        )

                self._append(agent_id, refs)

            except Exception:
                logger.exception("segv-cache rebuild failed for {}", agent_id)

    def _trigger_all(self) -> None:
        with self._lock:
            agents = [
                agent_id
                for agent_id, pending in self._pending.items()
                if len(pending) >= self.config.trigger
            ]

        for agent_id in agents:
            self._enqueue(agent_id, False)

    # ------------------------------------------------------------------
    # Pending cues
    # ------------------------------------------------------------------

    def on_ingested(self, agent_id: str, seq: int, rows: list[Any]) -> None:
        """Fact callback: queue one ingested turn's cues."""
        if not self.enabled:
            return

        refs: list[CueRef] = []

        for row in rows:
            text = str(getattr(row, "event", "") or "").strip()

            if not text:
                text = str(getattr(row, "text", "") or "").strip()[:240]

            if text:
                refs.append(
                    CueRef(
                        seq=int(seq),
                        ordinal=int(row.ordinal),
                        chunk_id=str(row.chunk_id),
                        ts=str(getattr(row, "ts", "") or ""),
                        text=text,
                    )
                )

        if not refs:
            return

        count = self._append(agent_id, refs)

        if count >= self.config.trigger:
            self._enqueue(agent_id, False)

    def _append(self, agent_id: str, refs: list[CueRef]) -> int:
        with self._lock:
            pending = self._pending.setdefault(agent_id, [])
            have = self._have.setdefault(agent_id, set())

            for ref in refs:
                if ref.key in have:
                    continue

                have.add(ref.key)
                bisect.insort(pending, ref, key=lambda item: item.key)

            return len(pending)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _enqueue(self, agent_id: str, force: bool) -> None:
        with self._lock:
            if agent_id in self._queued and not force:
                return

            self._queued.add(agent_id)

        self._queue.put((agent_id, force))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)

            except queue.Empty:
                continue

            if item is STOP or item is None:
                break

            agent_id, force = item

            with self._lock:
                self._queued.discard(agent_id)

            try:
                self.compress(agent_id, force=force)

            except Exception:
                logger.exception("segv-cache rewrite failed for {}", agent_id)

    def compress(self, agent_id: str, *, force: bool = False) -> bool:
        """Rewrite one bounded batch of pending cues; keep state on failure."""
        if not self.enabled or self.store is None or self.slm is None:
            return False

        state = self.store.load(agent_id)
        previous = state.snapshot if state is not None else Snapshot()
        version = state.version if state is not None else 0

        with self._lock:
            pending = list(self._pending.get(agent_id, ()))

        if not pending or (not force and len(pending) < self.config.trigger):
            return False

        batch = pending[: max(1, self.config.batch_cues)]
        batch, evidence = self._evidence(batch)

        if not batch:
            return False

        snapshot = self._rewrite(previous, evidence)

        if snapshot is None:
            return False

        last = batch[-1]
        pointer = last.key

        # Commit and prune together: an epoch-start inject holds _lock while
        # it reads the pointer and the pending cues, so it sees one side of
        # this rewrite or the other, never a torn state.
        with self._lock:
            self.store.save_snapshot(
                agent_id,
                snapshot=snapshot,
                pointer=pointer,
                version=version + 1,
            )

            remaining = [
                ref
                for ref in self._pending.get(agent_id, ())
                if (ref.seq, ref.ordinal) > pointer
            ]
            self._pending[agent_id] = remaining
            have = self._have.get(agent_id)

            if have is not None:
                have.intersection_update({ref.key for ref in remaining})

        logger.info(
            "segv-cache {}/{}: {} chunks -> v{} ({} items, {} left)",
            agent_id,
            f"{pointer[0]}:{pointer[1]}",
            len(batch),
            version + 1,
            len(snapshot.unresolved_items)
            + len(snapshot.locked_decisions)
            + len(snapshot.forbidden_behaviors)
            + len(snapshot.entity_map),
            len(remaining),
        )

        if remaining and (force or len(remaining) >= self.config.trigger):
            self._enqueue(agent_id, force)

        return True

    def _evidence(
        self, refs: list[CueRef]
    ) -> tuple[list[CueRef], list[tuple[int, int, str, str]]]:
        """Raw chunk text for the batch, bounded by raw_chars.

        The cue is a fallback only: it is lossy by design, so compression reads
        the source whenever the fact store still has it.
        """
        store = getattr(self.fact, "store", None)
        rows: dict[str, Any] = {}

        if store is not None:
            try:
                rows = store.get_chunks([ref.chunk_id for ref in refs])

            except Exception:
                logger.exception("segv-cache could not read raw chunks")

        included: list[CueRef] = []
        evidence: list[tuple[int, int, str, str]] = []
        budget = max(1, int(self.config.raw_chars))
        used = 0

        for ref in refs:
            row = rows.get(ref.chunk_id) or {}
            text = str(row.get("text") or "").strip() or ref.text.strip()
            text = ENVELOPE.sub("", text).strip()

            if not text:
                continue

            if included and used + len(text) > budget:
                break

            if not included and len(text) > budget:
                text = text[:budget]

            included.append(ref)
            evidence.append((ref.seq, ref.ordinal, ref.ts, text))
            used += len(text)

        return included, evidence

    def _rewrite(
        self, previous: Snapshot, evidence: list[tuple[int, int, str, str]]
    ) -> Snapshot | None:
        system, user = rewrite_prompt(
            previous,
            evidence,
            max_items=self.config.max_items,
        )
        problem = ""

        for _ in range(2):
            prompt = user

            if problem:
                prompt += (
                    "\n\nYour previous reply was invalid: "
                    f"{problem}. Return only the JSON object."
                )

            try:
                raw = self.slm.chat(system, prompt)

                return parse_snapshot(
                    raw,
                    previous,
                    max_items=self.config.max_items,
                    min_retention=self.config.min_retention,
                )

            except SnapshotError as exc:
                problem = str(exc)

                logger.warning("segv-cache rewrite rejected: {}", exc)

            except Exception:
                logger.exception("segv-cache rewrite call failed")

                return None

        return None

    # ------------------------------------------------------------------
    # Inject
    # ------------------------------------------------------------------

    def inject(self, body: dict[str, Any]) -> str:
        """Return this turn's prompt text; never raise across the boundary."""
        if not self.enabled or self.store is None:
            return ""

        agent_id = str(body.get("agent_id") or "").strip()

        if not agent_id:
            return ""

        epoch = _epoch(body.get("epoch"))
        query = _inbox_query(body)

        with self._agent_lock(agent_id):
            with self._lock:
                try:
                    state, changed = self.store.begin_epoch(agent_id, epoch)

                except Exception:
                    logger.exception(
                        "segv-cache epoch begin failed for {}", agent_id
                    )

                    return ""

                pending = list(self._pending.get(agent_id, ()))

            blocks: list[str] = []
            tail: list[CueRef] = []
            seen: set[tuple[int, int]] = set()

            if changed:
                tail = _tail(
                    pending,
                    state.pointer,
                    self.config.tail_cues,
                    self.config.tail_chars,
                )
                # The guide opens every epoch (and init) so a freshly
                # compacted context relearns what these blocks are.
                blocks.append(GUIDE)

                snapshot_text = state.snapshot.render()

                if snapshot_text:
                    blocks.append(snapshot_text)

                tail_text = render_refs("recent_cues", tail)

                if tail_text:
                    blocks.append(tail_text)

                seen = {ref.key for ref in tail}

                if pending:
                    self._enqueue(agent_id, True)

            else:
                try:
                    seen = self.store.seen(agent_id, epoch)

                except Exception:
                    logger.exception(
                        "segv-cache seen read failed for {}/{}", agent_id, epoch
                    )

            recall = self._recall(agent_id, query, seen)

            if recall:
                blocks.append(render_refs("recall", recall))

            marks = [ref.key for ref in recall] + [ref.key for ref in tail]

            if marks:
                try:
                    self.store.mark_seen(agent_id, epoch, marks)

                except Exception:
                    logger.exception(
                        "segv-cache seen persist failed for {}/{}", agent_id, epoch
                    )

            return "\n\n".join(blocks)

    def _recall(
        self, agent_id: str, query: str, seen: set[tuple[int, int]]
    ) -> list[CueRef]:
        if not query:
            return []

        try:
            hits = self.fact.search(
                agent_id,
                query,
                top_k=self.config.recall_k,
                pool=self.config.recall_pool,
                rerank=False,
                exclude=seen,
            )

        except Exception:
            logger.exception("segv-cache recall failed for {}", agent_id)

            return []

        out: list[CueRef] = []

        for hit in hits:
            text = str(hit.get("event") or "").strip()

            if not text:
                text = str(hit.get("text") or "").strip()[:240]

            if not text:
                continue

            out.append(
                CueRef(
                    seq=int(hit.get("seq") or 0),
                    ordinal=int(hit.get("ordinal") or 0),
                    chunk_id=str(hit.get("chunk_id") or ""),
                    ts=str(hit.get("ts") or ""),
                    text=text,
                )
            )

            if len(out) >= self.config.recall_k:
                break

        return out

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def state(self, agent_id: str) -> dict[str, Any]:
        if not self.enabled or self.store is None:
            return {"enabled": False, "agent_id": agent_id}

        state: CacheState | None = self.store.load(agent_id)

        with self._lock:
            pending = len(self._pending.get(agent_id, ()))

        epoch = state.last_epoch if state is not None else None

        return {
            "enabled": True,
            "agent_id": agent_id,
            "pointer": (
                {"seq": state.pointer[0], "ordinal": state.pointer[1]}
                if state is not None and state.pointer is not None
                else None
            ),
            "last_epoch": epoch,
            "version": state.version if state is not None else 0,
            "updated_at": state.updated_at if state is not None else 0.0,
            "pending": pending,
            "seen": (
                self.store.seen_size(agent_id, epoch)
                if epoch is not None
                else 0
            ),
            "snapshot": (
                state.snapshot.to_json() if state is not None else Snapshot().to_json()
            ),
        }

    def tail(self, agent_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.enabled or self.store is None:
            return []

        state = self.store.load(agent_id)
        pointer = state.pointer if state is not None else None

        with self._lock:
            pending = list(self._pending.get(agent_id, ()))

        refs = _tail(
            pending,
            pointer,
            min(limit or self.config.tail_cues, self.config.tail_cues),
            self.config.tail_chars,
        )

        return [
            {
                "seq": ref.seq,
                "ordinal": ref.ordinal,
                "chunk_id": ref.chunk_id,
                "ts": ref.ts,
                "text": ref.text,
            }
            for ref in refs
        ]

    def stats(self) -> dict[str, Any]:
        if not self.enabled or self.store is None:
            return {"enabled": False}

        with self._lock:
            pending = sum(len(items) for items in self._pending.values())
            agents = len(self._pending)

        return {
            "enabled": True,
            **self.store.stats(),
            "pending": pending,
            "pending_agents": agents,
        }

    # ------------------------------------------------------------------

    def _agent_lock(self, agent_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(agent_id)

            if lock is None:
                lock = threading.Lock()
                self._locks[agent_id] = lock

            return lock
