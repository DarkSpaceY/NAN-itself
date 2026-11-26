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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from rocksdict import DBCompressionType, Options, Rdict

from .snapshot import Snapshot

STATE_PREFIX = b"A"
EPOCH_PREFIX = b"E"
SEEN_PREFIX = b"S"  # legacy: read once for migration, never written again
SEP = b"\x00"

RECORD = struct.Struct(">QI")


def state_key(agent_id: str) -> bytes:
    return STATE_PREFIX + agent_id.encode() + SEP


def epoch_key(agent_id: str, epoch: int) -> bytes:
    return EPOCH_PREFIX + agent_id.encode() + SEP + struct.pack(">Q", epoch)


def seen_key(agent_id: str, epoch: int) -> bytes:
    return SEEN_PREFIX + agent_id.encode() + SEP + struct.pack(">Q", epoch)


@dataclass
class CacheState:
    """What survives an epoch: the committed memory and where it ends."""

    snapshot: Snapshot = field(default_factory=Snapshot)
    pointer: tuple[int, int] | None = None
    last_epoch: int | None = None
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
            version=int(raw.get("version") or 0),
            updated_at=float(raw.get("updated_at") or 0.0),
        )


@dataclass
class EpochState:
    """What one epoch has already shown the model, and to which version."""

    epoch: int
    # A data block (memory or cues) has gone out. A status-only opening turn
    # does not count, so the first real block still gets the guide.
    opened: bool = False
    # The snapshot version whose memory block went out, None until one did.
    delivered_version: int | None = None
    refs: set[tuple[int, int]] = field(default_factory=set)
    status_key: str = ""
    updated_at: float = 0.0

    def to_json(self) -> dict:
        return {
            "epoch": self.epoch,
            "opened": self.opened,
            "delivered_version": self.delivered_version,
            "refs": [[seq, ordinal] for seq, ordinal in sorted(self.refs)],
            "status_key": self.status_key,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, raw: dict, epoch: int) -> EpochState:
        delivered = raw.get("delivered_version")

        return cls(
            epoch=int(raw.get("epoch", epoch)),
            opened=bool(raw.get("opened")),
            delivered_version=int(delivered) if delivered is not None else None,
            refs={
                (int(entry[0]), int(entry[1]))
                for entry in raw.get("refs") or []
                if isinstance(entry, (list, tuple)) and len(entry) == 2
            },
            status_key=str(raw.get("status_key") or ""),
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

    def load_epoch(self, agent_id: str, epoch: int) -> EpochState | None:
        with self._lock:
            return self._load_epoch_locked(agent_id, epoch)

    def _load_epoch_locked(self, agent_id: str, epoch: int) -> EpochState | None:
        raw = self._db.get(epoch_key(agent_id, epoch))

        if raw is None:
            return None

        try:
            return EpochState.from_json(json.loads(raw), epoch)

        except Exception:
            logger.warning("segv-cache epoch unreadable for {}/{}", agent_id, epoch)

            return None

    def save_epoch(self, agent_id: str, epoch_state: EpochState) -> None:
        with self._lock:
            self._save_epoch_locked(agent_id, epoch_state)

    def _save_epoch_locked(self, agent_id: str, epoch_state: EpochState) -> None:
        epoch_state.updated_at = time.time()
        self._db.put(
            epoch_key(agent_id, epoch_state.epoch),
            json.dumps(epoch_state.to_json(), ensure_ascii=False).encode(),
        )

    def begin_epoch(
        self, agent_id: str, epoch: int
    ) -> tuple[CacheState, EpochState, bool, bool]:
        """
        Load the durable snapshot and this epoch's delivery ledger.

        Returns (state, epoch_state, changed, stale). A request from an
        older epoch is stale: it keeps reading the newest epoch's ledger so
        a delayed turn cannot rewind delivery or clear a newer seen set.
        """
        with self._lock:
            state = self._load_locked(agent_id)

            created = state is None

            if created:
                state = CacheState()

            stale = state.last_epoch is not None and epoch < state.last_epoch

            current = state.last_epoch if stale else epoch

            changed = not stale and (created or state.last_epoch != epoch)

            if changed:
                state.last_epoch = epoch
                state.updated_at = time.time()

                self._put_locked(agent_id, state)

            epoch_state = self._load_epoch_locked(agent_id, current)

            if epoch_state is None:
                # One-time migration: keep the old seen set, drop the old
                # delivered bool (a single extra block is safer than a miss).
                legacy = self._db.get(seen_key(agent_id, current))
                epoch_state = EpochState(
                    epoch=current,
                    refs=set(_decode_seen(legacy)) if legacy else set(),
                )

                self._save_epoch_locked(agent_id, epoch_state)

            return state, epoch_state, changed, stale

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
            epoch_state = self._load_epoch_locked(agent_id, epoch)

            if epoch_state is not None:
                return set(epoch_state.refs)

            raw = self._db.get(seen_key(agent_id, epoch))

            return set(_decode_seen(raw)) if raw else set()

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
