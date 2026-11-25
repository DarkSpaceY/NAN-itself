"""Cache durability: one state record per agent, one seen set per epoch.

    A <agent>\\0                 -> state json (snapshot + pointer + epoch)
    S <agent>\\0 <epoch:8B BE>   -> sorted (seq:8B, ordinal:4B) records

The state record is written whole, so the snapshot and its pointer can
never disagree after a crash. A seen set is only appended to; an epoch
boundary deletes every earlier set for that agent.
"""

from __future__ import annotations

import json
import struct
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from rocksdict import DBCompressionType, Options, Rdict, WriteBatch

from .snapshot import Snapshot

STATE_PREFIX = b"A"
SEEN_PREFIX = b"S"
SEP = b"\x00"

RECORD = struct.Struct(">QI")


def state_key(agent_id: str) -> bytes:
    return STATE_PREFIX + agent_id.encode() + SEP


def seen_key(agent_id: str, epoch: int) -> bytes:
    return SEEN_PREFIX + agent_id.encode() + SEP + struct.pack(">Q", epoch)


@dataclass
class CacheState:
    snapshot: Snapshot = field(default_factory=Snapshot)
    pointer: tuple[int, int] | None = None
    # None: a rewrite has produced a snapshot but no turn has consumed it
    # yet, so the next inject still owes the full epoch-start block.
    last_epoch: int | None = None
    # False until the epoch-start memory block has actually gone out; when
    # the epoch opens before memory is ready, a later turn retries once.
    delivered: bool = False
    version: int = 0
    updated_at: float = 0.0

    def to_json(self) -> dict:
        return {
            "snapshot": self.snapshot.to_json(),
            "pointer": (
                {"seq": self.pointer[0], "ordinal": self.pointer[1]}
                if self.pointer is not None
                else None
            ),
            "last_epoch": self.last_epoch,
            "delivered": self.delivered,
            "version": self.version,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, raw: dict) -> CacheState:
        pointer = raw.get("pointer")
        last_epoch = raw.get("last_epoch")

        return cls(
            snapshot=Snapshot.from_json(raw.get("snapshot")),
            pointer=(
                (int(pointer.get("seq") or 0), int(pointer.get("ordinal") or 0))
                if isinstance(pointer, dict)
                else None
            ),
            last_epoch=int(last_epoch) if last_epoch is not None else None,
            delivered=bool(raw.get("delivered")),
            version=int(raw.get("version") or 0),
            updated_at=float(raw.get("updated_at") or 0.0),
        )


class CacheStore:
    """One RocksDB per service; every call serializes on one lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._db = self._open(path)

    @staticmethod
    def _open(path: Path) -> Rdict:
        path.mkdir(parents=True, exist_ok=True)

        options = Options()

        options.create_if_missing(True)
        options.set_compression_type(DBCompressionType.zstd())
        options.set_write_buffer_size(16 * 1024 * 1024)
        options.set_max_write_buffer_number(2)
        options.set_max_background_jobs(1)
        options.set_keep_log_file_num(2)

        return Rdict(str(path), options)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def load(self, agent_id: str) -> CacheState | None:
        with self._lock:
            return self._load_locked(agent_id)

    def _load_locked(self, agent_id: str) -> CacheState | None:
        raw = self._db.get(state_key(agent_id))

        if raw is None:
            return None

        try:
            return CacheState.from_json(json.loads(raw))

        except Exception:
            logger.warning("segv-cache state unreadable for {}", agent_id)

            return None

    def _put_locked(self, agent_id: str, state: CacheState) -> None:
        self._db.put(
            state_key(agent_id),
            json.dumps(state.to_json(), ensure_ascii=False).encode(),
        )

    def begin_epoch(self, agent_id: str, epoch: int) -> tuple[CacheState, bool]:
        """Load the state and start a fresh seen set when the epoch changed."""
        with self._lock:
            state = self._load_locked(agent_id)
            created = state is None

            if created:
                state = CacheState()

            changed = created or state.last_epoch != epoch

            if changed:
                state.last_epoch = epoch
                state.delivered = False
                state.updated_at = time.time()
                self._put_locked(agent_id, state)
                self._clear_seen_locked(agent_id)

            return state, changed

    def set_delivered(self, agent_id: str, delivered: bool) -> None:
        """Record whether this epoch's memory block has gone out."""
        with self._lock:
            state = self._load_locked(agent_id)

            if state is None or state.delivered == delivered:
                return

            state.delivered = delivered
            state.updated_at = time.time()

            self._put_locked(agent_id, state)

    def save_snapshot(
        self,
        agent_id: str,
        *,
        snapshot: Snapshot,
        pointer: tuple[int, int],
        version: int,
    ) -> CacheState:
        """Write the snapshot and its pointer in one record."""
        with self._lock:
            state = self._load_locked(agent_id) or CacheState()

            state.snapshot = snapshot
            state.pointer = pointer
            state.version = version
            state.updated_at = time.time()

            self._put_locked(agent_id, state)

            return state

    def agents(self) -> list[str]:
        with self._lock:
            out: list[str] = []

            for key, _ in self._db.items(from_key=STATE_PREFIX):
                if not key.startswith(STATE_PREFIX):
                    break

                out.append(key[1:-1].decode())

            return sorted(out)

    # ------------------------------------------------------------------
    # Seen
    # ------------------------------------------------------------------

    def seen(self, agent_id: str, epoch: int) -> set[tuple[int, int]]:
        with self._lock:
            raw = self._db.get(seen_key(agent_id, epoch))

            return set(_decode_seen(raw)) if raw else set()

    def mark_seen(
        self, agent_id: str, epoch: int, items: Iterable[tuple[int, int]]
    ) -> None:
        new = {(int(seq), int(ordinal)) for seq, ordinal in items}

        if not new:
            return

        with self._lock:
            raw = self._db.get(seen_key(agent_id, epoch))
            current = set(_decode_seen(raw)) if raw else set()
            merged = current | new

            if merged == current:
                return

            self._db.put(
                seen_key(agent_id, epoch),
                b"".join(RECORD.pack(seq, ordinal) for seq, ordinal in sorted(merged)),
            )

    def _clear_seen_locked(self, agent_id: str) -> None:
        prefix = SEEN_PREFIX + agent_id.encode() + SEP
        keys: list[bytes] = []

        for key, _ in self._db.items(from_key=prefix):
            if not key.startswith(prefix):
                break

            keys.append(key)

        if not keys:
            return

        batch = WriteBatch()

        for key in keys:
            batch.delete(key)

        self._db.write(batch)

    def seen_size(self, agent_id: str, epoch: int) -> int:
        return len(self.seen(agent_id, epoch))

    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "path": str(self.path),
                "agents": len(self.agents()),
            }


def _decode_seen(raw: bytes) -> Iterator[tuple[int, int]]:
    for offset in range(0, len(raw) - RECORD.size + 1, RECORD.size):
        yield RECORD.unpack_from(raw, offset)
