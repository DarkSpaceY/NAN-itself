"""
The turn archive: RocksDB (rocksdict) plus a single writer thread.

Layout -- one column family, six key prefixes:

    T <agent_id>\\0 <seq:8B BE>        -> zstd(turn json)
    X <started_at:8B BE float>\\0 <agent>\\0 <seq:8B>
                                        -> ended_at, llm window, size
    M <agent_id>                       -> agent meta (json)
    R <unit>\\0 <bucket:8B BE>          -> turns recorded in that hour/day/week
    S \\0 total_bytes | watermark | first | last
                                        -> scalar stats

T is the agent timeline (prefix scan = paging); X is the global
time order, which is what the ledger, the activity histogram and
retention walk. Values are compressed on the way in, so
`total_bytes` tracks bytes actually stored and retention is a real
disk bound.

R holds hour / day / week rollups: a histogram over a year is a few
hundred key reads instead of a scan of every turn. Minute
resolution never needs a rollup -- a window that fine is small
enough to walk X directly.

Writes are queued and batched off the request path: the HTTP
endpoint must answer inside the injection timeout, so it only
enqueues. A full queue drops the oldest-arriving item and logs --
the archive is best-effort and must never fail a turn (a non-2xx
reply would land on the Turn record as action=ok=False).

Retention walks X from the watermark and deletes whole turns until
the store is back under `prune_to * max_bytes`.
"""

from __future__ import annotations

import json
import queue
import struct
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import zstandard
from loguru import logger
from rocksdict import DBCompressionType, Options, Rdict, WriteBatch

TURN_PREFIX = b"T"
TIME_PREFIX = b"X"
META_PREFIX = b"M"
ROLL_PREFIX = b"R"
STAT_PREFIX = b"S"

# Rollups that are actually written. Minute resolution is served by
# walking X, so it is deliberately not one of them.
ROLL_KEPT: tuple[tuple[str, int], ...] = (("h", 3600), ("d", 86400), ("w", 604800))

# Finest first. `buckets` picks the first unit coarse enough to fit;
# "m" is the sentinel for "walk the time index".
ROLL_SCAN: tuple[tuple[str, int], ...] = (("m", 60),) + ROLL_KEPT

# A window this short is drawn as individual turns instead of bars.
TURN_UNIT_SPAN = 3600.0
TURN_UNIT_LIMIT = 2000

KEY_OVERHEAD = 64

SEP = b"\x00"


def _turn_key(agent_id: str, seq: int) -> bytes:
    return TURN_PREFIX + agent_id.encode() + SEP + struct.pack(">Q", seq)


def _roll_key(unit: str, index: int) -> bytes:
    return ROLL_PREFIX + unit.encode() + SEP + struct.pack(">Q", index)


def _time_key(started_at: float, agent_id: str, seq: int) -> bytes:
    # >d keeps float order for positive timestamps, which is all
    # wall-clock time produces here.
    return TIME_PREFIX + struct.pack(">d", started_at) + SEP + agent_id.encode() + SEP + (
        struct.pack(">Q", seq)
    )


def _parse_time_key(key: bytes) -> tuple[float, str, int]:
    # Split by fixed width, not by separator: a seq of 0 is eight
    # NUL bytes and would collide with the separator.
    started_at = struct.unpack(">d", key[1:9])[0]
    return started_at, key[10:-9].decode(), struct.unpack(">Q", key[-8:])[0]


def _meta_key(agent_id: str) -> bytes:
    return META_PREFIX + agent_id.encode()


@dataclass
class AgentMeta:
    agent_id: str
    parent_id: str | None = None
    depth: int = 0
    count: int = 0
    first: float | None = None
    last: float | None = None
    last_death: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "count": self.count,
            "first": self.first,
            "last": self.last,
            "last_death": self.last_death,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> AgentMeta:
        return cls(
            agent_id=str(raw["agent_id"]),
            parent_id=raw.get("parent_id"),
            depth=int(raw.get("depth") or 0),
            count=int(raw.get("count") or 0),
            first=raw.get("first"),
            last=raw.get("last"),
            last_death=raw.get("last_death"),
        )


@dataclass
class _AgentCursor:
    """Per-agent write cursor."""

    seq: int = 0
    meta: AgentMeta = field(default_factory=lambda: AgentMeta(agent_id=""))


def _first_line(text: Any, limit: int = 160) -> str:
    """Collapse whitespace and clip; the ledger row has one line."""
    return " ".join(str(text or "").split())[:limit]


def headline(payload: dict[str, Any]) -> str:
    """
    One line of "what happened" for a ledger row: what the agent
    replied, else what it called, else what it was answering.
    """
    line = _first_line(payload.get("reply"))

    if line:
        return line

    names = [c.get("name") for c in payload.get("calls") or () if isinstance(c, dict) and c.get("name")]

    if names:
        return ", ".join(str(name) for name in names)

    for message in reversed(payload.get("inbox") or ()):
        line = _first_line(message.get("text") if isinstance(message, dict) else message)

        if line:
            return line

    return ""


def summarize(payload: dict[str, Any], seq: int) -> dict[str, Any]:
    """
    The list-view shape of one turn: enough for a ledger row and a
    histogram mark, small enough to ship over SSE.
    """
    llm = payload.get("llm") or {}

    results = payload.get("results") or []

    usage = llm.get("usage") or {}

    injections = payload.get("injections") or ()

    death = payload.get("death") if isinstance(payload.get("death"), dict) else {}

    verbs_start = min((r.get("started_at") for r in results), default=None)

    verbs_end = max((r.get("ended_at") for r in results), default=None)

    return {
        "agent_id": payload.get("agent_id"),
        "seq": seq,
        "started_at": payload.get("started_at"),
        "ended_at": payload.get("ended_at"),
        "origin": payload.get("origin"),
        "depth": payload.get("depth", 0),
        "parent_id": payload.get("parent_id"),
        "persona": payload.get("persona") is not None,
        "inbox": len(payload.get("inbox") or ()),
        "injections": len(injections),
        "delta": len(payload.get("delta") or ()),
        "calls": len(payload.get("calls") or ()),
        "reply_chars": len(payload.get("reply") or ""),
        "llm_started_at": llm.get("started_at"),
        "llm_ended_at": llm.get("ended_at"),
        "verbs_started_at": verbs_start,
        "verbs_ended_at": verbs_end,
        "retries": llm.get("retries") or 0,
        "tokens": int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0),
        "errors": sum(1 for i in injections if isinstance(i, dict) and i.get("ok") is False),
        "death": death.get("reason"),
        "summary": headline(payload),
    }


class TurnStore:
    """
    One instance per process. `ingest` is called from the HTTP
    thread; everything else (seq assignment, compression, batch
    write, retention, fan-out) runs on the writer thread.
    """

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 0,
        prune_to: float = 0.95,
        batch_size: int = 256,
        batch_interval: float = 0.1,
        queue_max: int = 10_000,
        notify: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.prune_to = prune_to
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self._notify = notify or (lambda _: None)

        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_max)

        self._db = self._open(path)

        self._compressor = zstandard.ZstdCompressor(level=3)
        self._decompressor = zstandard.ZstdDecompressor()

        # Writer-thread-owned state.
        self._cursors: dict[str, _AgentCursor] = {}
        self._total_bytes = 0
        self._turns = 0
        self._watermark: float | None = None
        self._first: float | None = None
        self._last: float | None = None
        # unit -> bucket index -> turn count. Small enough to hold:
        # a year is ~8.9k hours plus ~370 days plus ~53 weeks.
        self._roll: dict[str, dict[int, int]] = {unit: {} for unit, _ in ROLL_KEPT}

        self._load_state()

        self._thread = threading.Thread(target=self._run, name="record-writer", daemon=True)

        self.dropped = 0

    # ------------------------------------------------------------------
    # Open / state
    # ------------------------------------------------------------------

    @staticmethod
    def _open(path: Path) -> Rdict:
        path.mkdir(parents=True, exist_ok=True)

        options = Options()

        options.create_if_missing(True)

        # Values arrive already compressed; let RocksDB spend its
        # compression budget on nothing and keep CPU for the writer.
        options.set_compression_type(DBCompressionType.none())

        options.set_write_buffer_size(64 * 1024 * 1024)

        options.set_max_write_buffer_number(4)

        options.set_max_background_jobs(2)

        options.set_keep_log_file_num(2)

        return Rdict(str(path), options)

    def _load_state(self) -> None:
        with self._lock:
            for key, value in self._db.items(from_key=META_PREFIX):
                if not key.startswith(META_PREFIX):
                    break

                raw = json.loads(value)

                meta = AgentMeta.from_json(raw)

                cursor = _AgentCursor(seq=meta.count, meta=meta)

                self._cursors[meta.agent_id] = cursor

            total = self._db.get(STAT_PREFIX + SEP + b"total_bytes")

            turns = self._db.get(STAT_PREFIX + SEP + b"turns")

            watermark = self._db.get(STAT_PREFIX + SEP + b"watermark")

            first = self._db.get(STAT_PREFIX + SEP + b"first")

            last = self._db.get(STAT_PREFIX + SEP + b"last")

            self._total_bytes = int(total.decode()) if total else 0

            self._turns = int(turns.decode()) if turns else 0

            self._watermark = float(watermark.decode()) if watermark else None

            self._first = (float(first.decode()) or None) if first else None

            self._last = (float(last.decode()) or None) if last else None

            for unit, _ in ROLL_KEPT:
                prefix = ROLL_PREFIX + unit.encode() + SEP

                for key, value in self._db.items(from_key=prefix):
                    if not key.startswith(prefix):
                        break

                    self._roll[unit][struct.unpack(">Q", key[len(prefix) :])[0]] = int(value.decode())

            if self._turns and not self._roll["h"]:
                # Written before rollups existed, or rebuilt from a
                # directory whose rollups were lost: one walk of X
                # fixes it, and only ever happens once.
                self._backfill_rollups()

    def _bump(self, started_at: float, delta: int) -> list[tuple[bytes, bytes | None]]:
        """
        Move one turn through the rollups. Caller holds the lock;
        the caller also has to persist what comes back, where a
        None value means "this bucket is empty now, delete it".
        """
        touched: list[tuple[bytes, bytes | None]] = []

        for unit, seconds in ROLL_KEPT:
            index = int(started_at // seconds)

            bucket = self._roll[unit]

            count = bucket.get(index, 0) + delta

            if count > 0:
                bucket[index] = count

                touched.append((_roll_key(unit, index), str(count).encode()))
            else:
                bucket.pop(index, None)

                touched.append((_roll_key(unit, index), None))

        return touched

    @staticmethod
    def _put_rolls(write: WriteBatch, touched: list[tuple[bytes, bytes | None]]) -> None:
        for key, value in touched:
            if value is None:
                write.delete(key)
            else:
                write.put(key, value)

    def _backfill_rollups(self) -> None:
        """Caller holds the lock. One walk of X, one batch write."""
        write = WriteBatch()

        first: float | None = None

        last: float | None = None

        for key, _ in self._db.items(from_key=TIME_PREFIX):
            if not key.startswith(TIME_PREFIX):
                break

            started_at = struct.unpack(">d", key[1:9])[0]

            first = started_at if first is None else min(first, started_at)

            last = started_at if last is None else max(last, started_at)

            self._put_rolls(write, self._bump(started_at, 1))

        if first is None:
            return

        self._first = first

        self._last = last

        write.put(STAT_PREFIX + SEP + b"first", str(first).encode())

        write.put(STAT_PREFIX + SEP + b"last", str(last).encode())

        self._db.write(write)

        logger.info("record rebuilt rollups over {} turns", self._turns)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)

        except queue.Full:
            self._queue.get_nowait()

            self._queue.put_nowait(None)

        self._thread.join(timeout=10.0)

        with self._lock:
            self._db.flush()

            self._db.close()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def ingest(self, payload: dict[str, Any]) -> bool:
        """
        Hand one part of a collect payload to the writer. Returns
        False when the queue is full (dropped, logged, never
        surfaced as an error to the collector).
        """
        try:
            self._queue.put_nowait(payload)

            return True

        except queue.Full:
            self.dropped += 1

            if self.dropped % 100 == 1:
                logger.warning("record queue full; dropped {} turns", self.dropped)

            return False

    def _run(self) -> None:
        stop = False

        while not stop:
            first = self._queue.get()

            if first is None:
                break

            batch = [first]

            deadline = time.monotonic() + self.batch_interval

            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    break

                try:
                    item = self._queue.get(timeout=remaining)

                except queue.Empty:
                    break

                if item is None:
                    stop = True

                    break

                batch.append(item)

            try:
                self._write_batch(batch)

            except Exception:
                logger.exception("record batch write failed ({} turns)", len(batch))

    def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        summaries: list[dict[str, Any]] = []

        touched: dict[str, AgentMeta] = {}

        with self._lock:
            write = WriteBatch()

            for payload in batch:
                agent_id = str(payload.get("agent_id") or "")

                started_at = payload.get("started_at")

                if not agent_id or started_at is None:
                    continue

                cursor = self._cursors.get(agent_id)

                if cursor is None:
                    cursor = _AgentCursor(meta=AgentMeta(agent_id=agent_id))

                    self._cursors[agent_id] = cursor

                seq = cursor.seq

                cursor.seq += 1

                # The arrival is what the histogram counts and what
                # bounds the time axis.
                stamp = float(started_at)

                self._put_rolls(write, self._bump(stamp, 1))

                self._first = stamp if self._first is None else min(self._first, stamp)

                self._last = stamp if self._last is None else max(self._last, stamp)

                payload = {**payload, "seq": seq, "origin": payload.get("origin") or "normal"}

                data = self._compressor.compress(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
                )

                tkey = _turn_key(agent_id, seq)

                xkey = _time_key(float(started_at), agent_id, seq)

                llm = payload.get("llm") or {}

                write.put(tkey, data)

                write.put(
                    xkey,
                    struct.pack(
                        ">dddQ",
                        float(payload.get("ended_at") or started_at),
                        float(llm.get("started_at") or 0.0),
                        float(llm.get("ended_at") or 0.0),
                        len(data),
                    ),
                )

                self._total_bytes += len(data) + KEY_OVERHEAD

                self._turns += 1

                meta = cursor.meta

                meta.agent_id = agent_id

                meta.parent_id = payload.get("parent_id")

                meta.depth = int(payload.get("depth") or 0)

                meta.count = cursor.seq

                meta.first = meta.first if meta.first is not None else float(started_at)

                meta.last = float(started_at)

                death = payload.get("death")

                if isinstance(death, dict):
                    meta.last_death = str(death.get("reason") or "") or None

                touched[agent_id] = meta

                summaries.append(summarize(payload, seq))

            write.put(STAT_PREFIX + SEP + b"total_bytes", str(self._total_bytes).encode())

            write.put(STAT_PREFIX + SEP + b"turns", str(self._turns).encode())

            # Only write the axis bounds when they exist: a batch with
            # no usable payload must not persist 0.0 and make a later
            # reopen treat the archive as empty.
            if self._first is not None:
                write.put(STAT_PREFIX + SEP + b"first", str(self._first).encode())

            if self._last is not None:
                write.put(STAT_PREFIX + SEP + b"last", str(self._last).encode())

            for agent_id, meta in touched.items():
                write.put(_meta_key(agent_id), json.dumps(meta.to_json()).encode())

            self._db.write(write)

            self._prune_locked()

        if summaries:
            self._notify(summaries)

    def _prune_locked(self) -> None:
        """
        Delete oldest turns until under `prune_to * max_bytes`.
        Caller holds the lock; the DB write is separate so a large
        prune does not grow one batch unboundedly.
        """
        if self.max_bytes <= 0 or self._total_bytes <= self.max_bytes:
            return

        target = int(self.max_bytes * self.prune_to)

        deleted = 0

        write = WriteBatch()

        for xkey, size_raw in self._db.items(from_key=self._water_start()):
            if not xkey.startswith(TIME_PREFIX):
                break

            started_at, agent_id, seq = _parse_time_key(xkey)

            write.delete(xkey)

            write.delete(_turn_key(agent_id, seq))

            self._put_rolls(write, self._bump(started_at, -1))

            self._total_bytes -= struct.unpack(">dddQ", size_raw)[3] + KEY_OVERHEAD

            self._turns -= 1

            self._watermark = started_at

            deleted += 1

            if self._total_bytes <= target or deleted >= 50_000:
                break

        if not deleted:
            return

        write.delete(STAT_PREFIX + SEP + b"total_bytes")

        write.delete(STAT_PREFIX + SEP + b"turns")

        write.put(STAT_PREFIX + SEP + b"total_bytes", str(max(0, self._total_bytes)).encode())

        write.put(STAT_PREFIX + SEP + b"turns", str(max(0, self._turns)).encode())

        write.put(STAT_PREFIX + SEP + b"watermark", str(self._watermark or 0.0).encode())

        # Deleting from the old end is what moves the start of the axis.
        self._first = self._watermark

        write.put(STAT_PREFIX + SEP + b"first", str(self._watermark or 0.0).encode())

        self._db.write(write)

        logger.info("record retention pruned {} turns, {:.2f} GB left", deleted, self._total_bytes / 1e9)

    def _water_start(self) -> bytes:
        if self._watermark is None:
            return TIME_PREFIX

        # Start one tick below the watermark: the watermark turn itself
        # was already deleted.
        return TIME_PREFIX + struct.pack(">d", self._watermark) + SEP

    # ------------------------------------------------------------------
    # Read path (request threads; lock keeps iteration consistent)
    # ------------------------------------------------------------------

    def agents(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []

            for key, value in self._db.items(from_key=META_PREFIX):
                if not key.startswith(META_PREFIX):
                    break

                out.append(json.loads(value))

            return out

    def turns(self, agent_id: str, *, limit: int = 30, before: int | None = None) -> dict[str, Any]:
        """
        Newest-first page of complete turns; `before` is an exclusive
        seq cursor. Returns `next_before` for the next page (null
        when the agent has no older turns). Complete payloads are
        small (a turn is one observation + one reply + tool output),
        so a page ships without a second request per row.
        """
        prefix = TURN_PREFIX + agent_id.encode() + SEP

        upper = prefix + struct.pack(">Q", (before - 1) if before is not None else 2**64 - 1)

        out: list[dict[str, Any]] = []

        next_before: int | None = None

        with self._lock:
            for key, value in self._db.items(backwards=True, from_key=upper):
                if not key.startswith(prefix):
                    break

                try:
                    payload = json.loads(self._decompressor.decompress(value))

                except Exception:
                    logger.warning("record entry unreadable: {!r}", key)

                    continue

                out.append({**payload, "seq": struct.unpack(">Q", key[len(prefix) :])[0]})

                if len(out) >= limit:
                    next_before = out[-1]["seq"]

                    break

        return {"turns": out, "next_before": next_before}

    def turn(self, agent_id: str, seq: int) -> dict[str, Any] | None:
        with self._lock:
            value = self._db.get(_turn_key(agent_id, seq))

        if value is None:
            return None

        return json.loads(self._decompressor.decompress(value))

    def activity(self, *, start: float, end: float, buckets: int = 1200) -> dict[str, Any]:
        """
        A histogram of `[start, end)`. Resolution follows the span:
        individual turns for a short window, then minutes, hours,
        days, weeks -- whatever first fits in `buckets`. Weeks and
        above come from the rollups, so a year costs a few hundred
        key reads.
        """
        with self._lock:
            unit, series, total = self._series_locked(start, end, max(1, buckets))

        return {"unit": unit, "from": start, "to": end, "total": total, "series": series}

    def ledger(
        self,
        *,
        start: float,
        end: float,
        limit: int = 300,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """
        One page of the global time order, oldest first, so the UI
        can interleave every agent into a single timeline. `cursor`
        is the opaque hex of the previous page's last X key: paging
        neither re-reads a row nor skips one written between calls.

        The X value already carries the span, but state, parent and
        the headline only live in the compressed T value. Reading
        those costs one get per row (~300 a page) and saves the
        browser a request per row.
        """
        if cursor:
            from_key = bytes.fromhex(cursor) + SEP
        else:
            from_key = TIME_PREFIX + struct.pack(">d", max(0.0, start)) + SEP

        upper = TIME_PREFIX + struct.pack(">d", max(0.0, end))

        keys: list[bytes] = []

        spans: list[tuple[float, float, float]] = []

        with self._lock:
            for key, value in self._db.items(from_key=from_key):
                if not key.startswith(TIME_PREFIX) or key >= upper:
                    break

                ended_at, llm_start, llm_end, _ = struct.unpack(">dddQ", value)

                keys.append(key)

                spans.append((ended_at, llm_start, llm_end))

                if len(keys) >= limit:
                    break

            rows = [self._ledger_row(key, span) for key, span in zip(keys, spans)]

            return {
                "rows": rows,
                # The last row's key, whether or not the page was
                # full: the client resumes from it and treats a short
                # page as the end of the window.
                "cursor": keys[-1].hex() if keys else None,
                "total": None if cursor else self._series_locked(start, end, 512)[2],
                "first": self._first,
                "last": self._last,
            }

    def _ledger_row(self, key: bytes, span: tuple[float, float, float]) -> dict[str, Any]:
        """Caller holds the lock."""
        started_at, agent_id, seq = _parse_time_key(key)

        ended_at, llm_start, llm_end = span

        row: dict[str, Any] = {
            "agent_id": agent_id,
            "seq": seq,
            "started_at": started_at,
            "ended_at": ended_at,
            "llm_started_at": llm_start or None,
            "llm_ended_at": llm_end or None,
            "depth": 0,
            "parent_id": None,
            "tokens": 0,
            "errors": 0,
            "retries": 0,
            "calls": 0,
            "reply_chars": 0,
            "summary": "",
        }

        value = self._db.get(_turn_key(agent_id, seq))

        if value is None:
            return row

        try:
            payload = json.loads(self._decompressor.decompress(value))
        except Exception:
            logger.warning("record entry unreadable: {!r}", key)

            return row

        row.update(summarize(payload, seq))

        # The time index is the authority on when this turn ran.
        row["started_at"] = started_at

        row["ended_at"] = ended_at

        return row

    def _series_locked(
        self, start: float, end: float, buckets: int
    ) -> tuple[str, list[dict[str, Any]], int]:
        """
        Caller holds the lock. Returns (unit, series, total), where
        the series is sparse for "turn" (one point per turn) and
        dense for every other unit (one point per bucket).

        The window is clipped to what the archive actually holds, so a
        stray request for the epoch-to-century range cannot expand into
        thousands of empty buckets.
        """
        if self._first is None:
            return "m", [], 0

        start = max(float(start), self._first)

        end = min(float(end), self._last + 1.0) if self._last is not None else float(end)

        span = max(end - start, 0.0)

        if span <= 0:
            return "m", [], 0

        if span <= TURN_UNIT_SPAN:
            marks = self._marks_locked(start, end, TURN_UNIT_LIMIT)

            if marks is not None:
                return "turn", marks, len(marks)

        unit, seconds = ROLL_SCAN[-1]

        for candidate, width in ROLL_SCAN:
            if span / width <= buckets:
                unit, seconds = candidate, width

                break

        if unit == "m":
            return ("m", *self._minute_locked(start, end))

        return (unit, *self._rollup_locked(unit, seconds, start, end))

    def _marks_locked(self, start: float, end: float, limit: int) -> list[dict[str, Any]] | None:
        """Per-turn marks; None once the window is too busy to draw."""
        out: list[dict[str, Any]] = []

        for key in self._time_keys_locked(start, end):
            started_at, agent_id, seq = _parse_time_key(key)

            out.append({"t": started_at, "n": 1, "agent_id": agent_id, "seq": seq})

            if len(out) > limit:
                return None

        return out

    def _minute_locked(self, start: float, end: float) -> tuple[list[dict[str, Any]], int]:
        counts: dict[int, int] = {}

        for key in self._time_keys_locked(start, end):
            index = int(struct.unpack(">d", key[1:9])[0] // 60)

            counts[index] = counts.get(index, 0) + 1

        series = [{"t": index * 60.0, "n": count} for index, count in sorted(counts.items())]

        return series, sum(counts.values())

    def _rollup_locked(
        self, unit: str, seconds: int, start: float, end: float
    ) -> tuple[list[dict[str, Any]], int]:
        bucket = self._roll[unit]

        first = int(max(0.0, start) // seconds)

        last = int(float(end) // seconds)

        series: list[dict[str, Any]] = []

        total = 0

        for index in range(first, last + 1):
            count = bucket.get(index, 0)

            total += count

            series.append({"t": index * float(seconds), "n": count})

        return series, total

    def _time_keys_locked(self, start: float, end: float) -> Iterator[bytes]:
        """Caller holds the lock; walks the time index in [start, end)."""
        lower = TIME_PREFIX + struct.pack(">d", max(0.0, start)) + SEP

        upper = TIME_PREFIX + struct.pack(">d", max(0.0, end))

        for key, _ in self._db.items(from_key=lower):
            if not key.startswith(TIME_PREFIX) or key >= upper:
                break

            yield key

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "turns": max(0, self._turns),
                "bytes": max(0, self._total_bytes),
                "agents": len(self._cursors),
                "max_bytes": self.max_bytes,
                "watermark": self._watermark,
                "first": self._first,
                "last": self._last,
                "dropped": self.dropped,
                "path": str(self.path),
            }

    def iter_turns(self, agent_id: str) -> Iterator[dict[str, Any]]:
        """Test helper: every recorded payload for one agent, in seq order."""
        prefix = TURN_PREFIX + agent_id.encode() + SEP

        with self._lock:
            keys = []

            for key, _ in self._db.items(from_key=prefix):
                if not key.startswith(prefix):
                    break

                keys.append(key)

            for key in keys:
                value = self._db.get(key)

                if value is not None:
                    yield json.loads(self._decompressor.decompress(value))
