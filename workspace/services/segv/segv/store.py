"""
L0: the turn archive. RocksDB (rocksdict), one column family, prefixes.

    T <agent_id>\\0 <seq:8B BE>     -> zstd(complete turn json, rendered included)
    X <started_at:8B BE>\\0 <agent>\\0 <seq:8B>
                                    -> ended_at, llm window, stored size
    M <agent_id>                    -> agent meta (json)
    I <turn_id>                     -> agent_id\\0 seq (idempotency + stable id)
    R <unit>\\0 <bucket:8B BE>      -> turns recorded in that hour/day/week
    S \\0 <name>                    -> scalar counters

T is the per-agent timeline; X is the global time order the ledger,
the histogram and retention walk. Values are compressed on the way in,
so `total_bytes` is a real disk bound rather than a raw-input estimate.

A turn is written synchronously and atomically before `/turn` answers:
memory must not lose a turn the way a trace archive may. Everything
derived (fact / cache / exp) hangs off `notify`, never the reply path.
"""

from __future__ import annotations

import json
import struct
import threading
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
ID_PREFIX = b"I"
ROLL_PREFIX = b"R"
STAT_PREFIX = b"S"

ROLL_KEPT: tuple[tuple[str, int], ...] = (("h", 3600), ("d", 86400), ("w", 604800))

ROLL_SCAN: tuple[tuple[str, int], ...] = (("m", 60),) + ROLL_KEPT

TURN_UNIT_SPAN = 3600.0

TURN_UNIT_LIMIT = 2000

KEY_OVERHEAD = 64

SEP = b"\x00"

PRUNE_SLICE = 4096


def turn_uid(agent_id: str, started_at: float) -> str:
    """
    Stable identity for one turn. The hub may re-deliver a turn (it
    retries nothing, but a core restart can replay), so identity is
    content-derived, not a sequence number.
    """
    return f"{agent_id}:{started_at:.6f}"


def _turn_key(agent_id: str, seq: int) -> bytes:
    return TURN_PREFIX + agent_id.encode() + SEP + struct.pack(">Q", seq)


def _time_key(started_at: float, agent_id: str, seq: int) -> bytes:
    return (
        TIME_PREFIX
        + struct.pack(">d", started_at)
        + SEP
        + agent_id.encode()
        + SEP
        + struct.pack(">Q", seq)
    )


def _parse_time_key(key: bytes) -> tuple[float, str, int]:
    started_at = struct.unpack(">d", key[1:9])[0]

    rest = key[10:]

    return started_at, rest[:-9].decode(), struct.unpack(">Q", rest[-8:])[0]


def _meta_key(agent_id: str) -> bytes:
    return META_PREFIX + agent_id.encode()


def _id_key(uid: str) -> bytes:
    return ID_PREFIX + uid.encode()


def _roll_key(unit: str, index: int) -> bytes:
    return ROLL_PREFIX + unit.encode() + SEP + struct.pack(">Q", index)


def _stat_key(name: str) -> bytes:
    return STAT_PREFIX + SEP + name.encode()


@dataclass
class AgentMeta:
    agent_id: str
    parent_id: str | None = None
    depth: int = 0
    next_seq: int = 0
    count: int = 0
    pruned: int = 0
    first: float | None = None
    last: float | None = None
    last_death: str | None = None
    # epoch -> [first_ts, last_ts, turns]. The cache layer keys on it.
    epochs: dict[str, list[Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "next_seq": self.next_seq,
            "count": self.count,
            "pruned": self.pruned,
            "first": self.first,
            "last": self.last,
            "last_death": self.last_death,
            "epochs": self.epochs,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> AgentMeta:
        return cls(
            agent_id=str(raw.get("agent_id") or ""),
            parent_id=raw.get("parent_id"),
            depth=int(raw.get("depth") or 0),
            next_seq=int(raw.get("next_seq") or 0),
            count=int(raw.get("count") or 0),
            pruned=int(raw.get("pruned") or 0),
            first=raw.get("first"),
            last=raw.get("last"),
            last_death=raw.get("last_death"),
            epochs=dict(raw.get("epochs") or {}),
        )


def summarize(payload: dict[str, Any], seq: int) -> dict[str, Any]:
    """One ledger row: identity, span, and a headline, without a second read."""
    llm = payload.get("llm") or {}

    usage = llm.get("usage") or {}

    inbox = payload.get("inbox") or []

    results = payload.get("results") or []

    reply = payload.get("reply") or ""

    thinking = payload.get("thinking") or []

    headline = reply.strip().splitlines()[0] if reply.strip() else ""

    if not headline and inbox:
        headline = str(inbox[-1].get("text") or "")

    return {
        "agent_id": str(payload.get("agent_id") or ""),
        "seq": seq,
        "started_at": float(payload.get("started_at") or 0.0),
        "ended_at": payload.get("ended_at"),
        "depth": int(payload.get("depth") or 0),
        "parent_id": payload.get("parent_id"),
        "origin": str(payload.get("origin") or "normal"),
        "epoch": int(payload.get("epoch") or 0),
        "tokens": int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0),
        "retries": int(llm.get("retries") or 0),
        "calls": len(payload.get("calls") or []),
        "errors": sum(1 for r in results if not r.get("output")),
        "reply_chars": len(reply),
        "thinking_chars": sum(
            len(block.get("text") or "") for block in thinking if isinstance(block, dict)
        ),
        "summary": headline[:160],
    }


class Store:
    """One instance per process; all writes serialize on one lock."""

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 0,
        prune_to: float = 0.95,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path

        self.max_bytes = max_bytes

        self.prune_to = prune_to

        self._notify = notify or (lambda _: None)

        self._lock = threading.RLock()

        self._db = self._open(path)

        self._compressor = zstandard.ZstdCompressor(level=3)

        self._decompressor = zstandard.ZstdDecompressor()

        self._metas: dict[str, AgentMeta] = {}

        self._total_bytes = 0

        self._turns = 0

        self._watermark: float | None = None

        self._first: float | None = None

        self._last: float | None = None

        self._roll: dict[str, dict[int, int]] = {unit: {} for unit, _ in ROLL_KEPT}

        self._load_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _open(path: Path) -> Rdict:
        path.mkdir(parents=True, exist_ok=True)

        options = Options()

        options.create_if_missing(True)

        # Values arrive already compressed.
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

                try:
                    meta = AgentMeta.from_json(json.loads(value))

                except Exception:
                    logger.warning("segv meta unreadable: {!r}", key)

                    continue

                self._metas[meta.agent_id] = meta

            self._total_bytes = self._scalar_int("total_bytes")

            self._turns = self._scalar_int("turns")

            watermark = self._db.get(_stat_key("watermark"))

            first = self._db.get(_stat_key("first"))

            last = self._db.get(_stat_key("last"))

            self._watermark = float(watermark.decode()) if watermark else None

            self._first = float(first.decode()) if first else None

            self._last = float(last.decode()) if last else None

            for unit, _ in ROLL_KEPT:
                prefix = ROLL_PREFIX + unit.encode() + SEP

                for key, value in self._db.items(from_key=prefix):
                    if not key.startswith(prefix):
                        break

                    index = struct.unpack(">Q", key[len(prefix) :])[0]

                    self._roll[unit][index] = int(value.decode())

    def _scalar_int(self, name: str) -> int:
        raw = self._db.get(_stat_key(name))

        return int(raw.decode()) if raw else 0

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, payload: dict[str, Any]) -> str | None:
        """
        Store one completed turn. Returns its id, or None when the
        payload is unusable or this turn is already archived (the hub
        may re-deliver after a core restart).
        """
        agent_id = str(payload.get("agent_id") or "")

        started_at = payload.get("started_at")

        if not agent_id or started_at is None:
            logger.warning("segv drop: turn without agent_id/started_at")

            return None

        started = float(started_at)

        uid = turn_uid(agent_id, started)

        with self._lock:
            if self._db.get(_id_key(uid)) is not None:
                return None

            meta = self._metas.get(agent_id)

            if meta is None:
                meta = AgentMeta(agent_id=agent_id)

                self._metas[agent_id] = meta

            seq = meta.next_seq

            meta.next_seq = seq + 1

            meta.count += 1

            meta.parent_id = payload.get("parent_id")

            meta.depth = int(payload.get("depth") or 0)

            meta.first = started if meta.first is None else min(meta.first, started)

            meta.last = started if meta.last is None else max(meta.last, started)

            death = payload.get("death")

            if isinstance(death, dict):
                meta.last_death = str(death.get("reason") or "") or None

            epoch = int(payload.get("epoch") or 0)

            bucket = meta.epochs.setdefault(str(epoch), [started, started, 0])

            bucket[0] = min(bucket[0], started)

            bucket[1] = max(bucket[1], started)

            bucket[2] += 1

            record = {
                **payload,
                "seq": seq,
                "turn_id": uid,
                "origin": payload.get("origin") or "normal",
            }

            data = self._compressor.compress(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
            )

            llm = record.get("llm") or {}

            write = WriteBatch()

            write.put(_turn_key(agent_id, seq), data)

            write.put(_id_key(uid), agent_id.encode() + SEP + struct.pack(">Q", seq))

            write.put(
                _time_key(started, agent_id, seq),
                struct.pack(
                    ">dddQ",
                    float(record.get("ended_at") or started),
                    float(llm.get("started_at") or 0.0),
                    float(llm.get("ended_at") or 0.0),
                    len(data),
                ),
            )

            write.put(_meta_key(agent_id), json.dumps(meta.to_json()).encode())

            self._total_bytes += len(data) + KEY_OVERHEAD

            self._turns += 1

            self._first = started if self._first is None else min(self._first, started)

            self._last = started if self._last is None else max(self._last, started)

            self._put_rolls(write, self._bump(started, 1))

            write.put(_stat_key("total_bytes"), str(self._total_bytes).encode())

            write.put(_stat_key("turns"), str(self._turns).encode())

            write.put(_stat_key("first"), str(self._first).encode())

            write.put(_stat_key("last"), str(self._last).encode())

            self._db.write(write)

            self._prune_locked()

        self._notify(uid)

        return uid

    def _bump(self, started_at: float, delta: int) -> list[tuple[bytes, bytes | None]]:
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

    def _prune_locked(self) -> list[str]:
        if self.max_bytes <= 0 or self._total_bytes <= self.max_bytes:
            return []

        target = int(self.max_bytes * self.prune_to)

        removed: list[str] = []

        write = WriteBatch()

        start = TIME_PREFIX if self._watermark is None else _water_start(self._watermark)

        for xkey, _ in self._db.items(from_key=start):
            if not xkey.startswith(TIME_PREFIX) or self._total_bytes <= target:
                break

            started_at, agent_id, seq = _parse_time_key(xkey)

            tkey = _turn_key(agent_id, seq)

            raw = self._db.get(tkey)

            uid = ""

            if raw is not None:
                try:
                    uid = str(json.loads(self._decompressor.decompress(raw)).get("turn_id") or "")

                except Exception:
                    uid = ""

                self._total_bytes -= len(raw) + KEY_OVERHEAD

                self._turns -= 1

            write.delete(tkey)

            write.delete(xkey)

            if uid:
                write.delete(_id_key(uid))

            meta = self._metas.get(agent_id)

            if meta is not None:
                meta.count = max(0, meta.count - 1)

                meta.pruned += 1

                meta.first = started_at

                write.put(_meta_key(agent_id), json.dumps(meta.to_json()).encode())

            self._put_rolls(write, self._bump(started_at, -1))

            self._watermark = started_at

            removed.append(uid)

            if len(removed) >= PRUNE_SLICE:
                break

        write.put(_stat_key("total_bytes"), str(self._total_bytes).encode())

        write.put(_stat_key("turns"), str(self._turns).encode())

        write.put(_stat_key("watermark"), str(self._watermark).encode())

        self._db.write(write)

        if removed:
            logger.info("segv pruned {} turns ({} bytes left)", len(removed), self._total_bytes)

        return removed

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def agents(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []

            for key, value in self._db.items(from_key=META_PREFIX):
                if not key.startswith(META_PREFIX):
                    break

                try:
                    out.append(json.loads(value))

                except Exception:
                    continue

            return out

    def meta(self, agent_id: str) -> dict[str, Any] | None:
        with self._lock:
            raw = self._db.get(_meta_key(agent_id))

        if raw is None:
            return None

        try:
            return json.loads(raw)

        except Exception:
            return None

    def turns(self, agent_id: str, *, limit: int = 30, before: int | None = None) -> dict[str, Any]:
        prefix = TURN_PREFIX + agent_id.encode() + SEP

        upper = prefix + struct.pack(">Q", (before - 1) if before is not None else 2**64 - 1)

        out: list[dict[str, Any]] = []

        next_before: int | None = None

        with self._lock:
            for key, value in self._db.items(backwards=True, from_key=upper):
                if not key.startswith(prefix):
                    break

                payload = self._decode(value)

                if payload is None:
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

        return self._decode(value)

    def seq_of(self, uid: str) -> tuple[str, int] | None:
        """Resolve a turn id to its (agent, seq) pointer."""
        with self._lock:
            raw = self._db.get(_id_key(uid))

        if raw is None:
            return None

        agent_id, seq = raw.split(SEP, 1)

        return agent_id.decode(), struct.unpack(">Q", seq)[0]

    def iter_turns(self, batch: int = 1000) -> Iterator[tuple[str, int]]:
        """Walk the global time order without holding the lock across it."""
        cursor = TIME_PREFIX

        while True:
            keys: list[bytes] = []

            with self._lock:
                for key, _ in self._db.items(from_key=cursor):
                    if not key.startswith(TIME_PREFIX):
                        break

                    keys.append(key)

                    if len(keys) >= batch:
                        break

            if not keys:
                return

            for key in keys:
                _, agent_id, seq = _parse_time_key(key)

                yield agent_id, seq

            cursor = keys[-1] + SEP

    def ledger(
        self,
        *,
        start: float,
        end: float,
        limit: int = 300,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        from_key = bytes.fromhex(cursor) + SEP if cursor else _water_start(start)

        upper = TIME_PREFIX + struct.pack(">d", max(0.0, end))

        keys: list[bytes] = []

        with self._lock:
            for key, _ in self._db.items(from_key=from_key):
                if not key.startswith(TIME_PREFIX) or key >= upper:
                    break

                keys.append(key)

                if len(keys) >= limit:
                    break

            rows = [self._ledger_row(key) for key in keys]

            return {
                "rows": [row for row in rows if row is not None],
                "cursor": keys[-1].hex() if keys else None,
                "total": None if cursor else self._series_locked(start, end, 512)[2],
                "first": self._first,
                "last": self._last,
            }

    def _ledger_row(self, key: bytes) -> dict[str, Any] | None:
        started_at, agent_id, seq = _parse_time_key(key)

        raw = self._db.get(_time_key(started_at, agent_id, seq))

        ended_at = started_at

        llm_start = llm_end = 0.0

        if raw is not None and len(raw) == struct.calcsize(">dddQ"):
            ended_at, llm_start, llm_end, _ = struct.unpack(">dddQ", raw)

        value = self._db.get(_turn_key(agent_id, seq))

        if value is None:
            return None

        payload = self._decode(value)

        if payload is None:
            return None

        row = summarize(payload, seq)

        row.update(
            {
                "agent_id": agent_id,
                "started_at": started_at,
                "ended_at": ended_at,
                "llm_started_at": llm_start or None,
                "llm_ended_at": llm_end or None,
            }
        )

        return row

    def activity(self, *, start: float, end: float, buckets: int = 1200) -> dict[str, Any]:
        with self._lock:
            unit, series, total = self._series_locked(start, end, max(1, buckets))

        return {"unit": unit, "from": start, "to": end, "total": total, "series": series}

    def _series_locked(
        self, start: float, end: float, buckets: int
    ) -> tuple[str, list[dict[str, Any]], int]:
        span = end - start

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

    def _time_keys_locked(self, start: float, end: float) -> Iterator[bytes]:
        lower = _water_start(start)

        upper = TIME_PREFIX + struct.pack(">d", max(0.0, end))

        for key, _ in self._db.items(from_key=lower):
            if not key.startswith(TIME_PREFIX) or key >= upper:
                break

            yield key

    def _marks_locked(self, start: float, end: float, limit: int) -> list[dict[str, Any]] | None:
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

        series = [{"t": index * 60.0, "n": n} for index, n in sorted(counts.items())]

        return series, sum(counts.values())

    def _rollup_locked(
        self, unit: str, seconds: int, start: float, end: float
    ) -> tuple[list[dict[str, Any]], int]:
        bucket = self._roll[unit]

        first = int(max(0.0, start) // seconds)

        last = int(float(end) // seconds)

        series = [
            {"t": index * float(seconds), "n": bucket.get(index, 0)}
            for index in range(first, last + 1)
        ]

        return series, sum(point["n"] for point in series)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "turns": self._turns,
                "agents": len(self._metas),
                "bytes": self._total_bytes,
                "max_bytes": self.max_bytes,
                "first": self._first,
                "last": self._last,
                "pending": 0,
            }

    # ------------------------------------------------------------------

    def _decode(self, value: bytes) -> dict[str, Any] | None:
        try:
            return json.loads(self._decompressor.decompress(value))

        except Exception:
            logger.warning("segv turn unreadable")

            return None


def _water_start(start: float) -> bytes:
    return TIME_PREFIX + struct.pack(">d", max(0.0, start)) + SEP
