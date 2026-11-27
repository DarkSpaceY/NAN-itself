"""Per-agent epoch marker: has this epoch already been opened?

One small record per agent:
    A<agent_id> -> {"last_epoch": N, "updated_at": t}

The state key and its `last_epoch` field predate this module, so an
upgrade keeps reading the epoch a running agent already opened. The old
snapshot / epoch-ledger / seen keys (prefixes E and S) are unreadable
now and are dropped on open.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from loguru import logger
from rocksdict import DBCompressionType, Options, Rdict, WriteBatch

STATE_PREFIX = b"A"
LEGACY_PREFIXES = (b"E", b"S")
SEP = b"\x00"


def state_key(agent_id: str) -> bytes:
    return STATE_PREFIX + agent_id.encode() + SEP


class CacheStore:
    """One RocksDB per service; every call serializes on one lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._db = self._open(path)

        self._purge_legacy()

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

    def _purge_legacy(self) -> None:
        """Drop keys the compression-era cache wrote; nothing reads them."""
        with self._lock:
            stale: list[bytes] = []

            for prefix in LEGACY_PREFIXES:
                for key, _ in self._db.items(from_key=prefix):
                    if not key.startswith(prefix):
                        break

                    stale.append(key)

            if not stale:
                return

            batch = WriteBatch()

            for key in stale:
                batch.delete(key)

            self._db.write(batch)

            logger.info("segv-cache dropped {} legacy keys", len(stale))

    # ------------------------------------------------------------------
    # Epoch marker
    # ------------------------------------------------------------------

    def last_epoch(self, agent_id: str) -> int | None:
        with self._lock:
            raw = self._db.get(state_key(agent_id))

        if raw is None:
            return None

        try:
            value = json.loads(raw).get("last_epoch")

        except Exception:
            logger.warning("segv-cache marker unreadable for {}", agent_id)

            return None

        return int(value) if value is not None else None

    def set_last_epoch(self, agent_id: str, epoch: int) -> None:
        with self._lock:
            self._db.put(
                state_key(agent_id),
                json.dumps(
                    {"last_epoch": int(epoch), "updated_at": time.time()}
                ).encode(),
            )

    def agents(self) -> list[str]:
        with self._lock:
            out: list[str] = []

            for key, _ in self._db.items(from_key=STATE_PREFIX):
                if not key.startswith(STATE_PREFIX):
                    break

                out.append(key[1:-1].decode())

            return sorted(out)

    def stats(self) -> dict:
        return {"path": str(self.path), "agents": len(self.agents())}
