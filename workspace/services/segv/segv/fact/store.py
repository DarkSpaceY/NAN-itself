"""Fact storage: RocksDB records plus in-memory BM25 and HNSW indexes.

One record per chunk, self-describing and compressed:

    C <agent>\\0 <seq:8B BE> <ordinal:4B BE>
        -> zstd( header | meta json | raw int8[dim] | event int8[dim]? )

    F <agent>\\0 <seq:8B BE>            -> ordinals committed (idempotency)
    S \\0 dim                           -> embedding dimension guard

The record is the fact: raw chunk text, cue text, two int8 vectors, and the
L0 pointer (agent_id, seq). There is no rendered JSON in this layer.

BM25 uses the frozen v2 parameters (k1=1.5, b=0) and the same idf
calculation as rank_bm25, so scores match the experiment. Dense search is
usearch HNSW over int8 cosine. Both indexes are rebuilt from this database
at startup; they are never the source of truth.
"""

from __future__ import annotations

import json
import math
import struct
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zstandard
from loguru import logger
from rocksdict import DBCompressionType, Options, Rdict, WriteBatch
from usearch.index import Index

from .config import BM25Config
from .text import tokenize

CHUNK_PREFIX = b"C"
MARK_PREFIX = b"F"
STAT_PREFIX = b"S"
SEP = b"\x00"
HEADER = struct.Struct(">IBI")
KEY_OVERHEAD = 64
EVENT_FLAG = 1


def chunk_key(agent_id: str, seq: int, ordinal: int) -> bytes:
    return CHUNK_PREFIX + agent_id.encode() + SEP + struct.pack(">QI", seq, ordinal)


def mark_key(agent_id: str, seq: int) -> bytes:
    return MARK_PREFIX + agent_id.encode() + SEP + struct.pack(">Q", seq)


def _stat_key(name: str) -> bytes:
    return STAT_PREFIX + SEP + name.encode()


def _parse_chunk_key(key: bytes) -> tuple[str, int, int]:
    agent, rest = key[1:].split(SEP, 1)
    seq, ordinal = struct.unpack(">QI", rest)

    return agent.decode(), seq, ordinal


def quantize(vector: Any) -> np.ndarray:
    """Normalized floats -> int8 at full scale; cosine survives rounding."""
    values = np.asarray(vector, dtype=np.float32)

    return np.rint(values * 127.0).clip(-127, 127).astype(np.int8)


@dataclass
class ChunkRow:
    chunk_id: str
    ordinal: int
    started_at: float
    ts: str
    text: str
    event: str
    raw_vec: np.ndarray
    event_vec: np.ndarray | None


class BM25Index:
    """BM25Okapi math with b=0, scoped to one agent."""

    def __init__(self, config: BM25Config) -> None:
        self.k1 = float(config.k1)
        self.b = float(config.b)
        self.tokenizer = config.tokenizer
        self._tf: dict[str, dict[str, int]] = {}
        self._df: dict[str, int] = {}
        self._dl: dict[str, int] = {}
        self._n = 0
        self._total_len = 0
        self._idf: dict[str, float] | None = None
        self._avgdl = 0.0

    def add(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text, self.tokenizer)
        counts = Counter(tokens)

        for term, freq in counts.items():
            self._tf.setdefault(term, {})[chunk_id] = freq
            self._df[term] = self._df.get(term, 0) + 1

        self._dl[chunk_id] = len(tokens)
        self._n += 1
        self._total_len += len(tokens)
        self._idf = None

    def _ensure_idf(self) -> None:
        if self._idf is not None:
            return

        raw = {
            term: math.log(self._n - df + 0.5) - math.log(df + 0.5)
            for term, df in self._df.items()
        }

        average = sum(raw.values()) / len(raw) if raw else 0.0
        epsilon = 0.25 * average

        self._idf = {term: (value if value > 0 else epsilon) for term, value in raw.items()}
        self._avgdl = self._total_len / self._n if self._n else 0.0

    def search(self, query: str, limit: int, seq: Mapping[str, int]) -> list[str]:
        if self._n == 0 or limit <= 0:
            return []

        self._ensure_idf()

        scores: dict[str, float] = {}

        for term in tokenize(query, self.tokenizer):
            postings = self._tf.get(term)

            if not postings:
                continue

            idf = self._idf.get(term, 0.0) if self._idf else 0.0

            if idf == 0.0:
                continue

            for chunk_id, freq in postings.items():
                length = self._dl[chunk_id]
                norm = 1.0 - self.b + (self.b * length / self._avgdl if self._avgdl else 0.0)
                score = idf * freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ordered = sorted(
            ((chunk_id, score) for chunk_id, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], seq.get(item[0], 0), item[0]),
        )

        return [chunk_id for chunk_id, _ in ordered[:limit]]


class DenseIndex:
    """usearch HNSW over int8 cosine vectors, scoped to one agent."""

    def __init__(self, dim: int, connectivity: int) -> None:
        self.index = Index(
            ndim=dim,
            metric="cos",
            dtype="i8",
            connectivity=connectivity,
        )
        self._next = 0
        self._ids: dict[int, str] = {}

    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        key = self._next
        self._next += 1

        self.index.add(key, np.asarray(vector, dtype=np.int8))
        self._ids[key] = chunk_id

    def search(self, vector: np.ndarray, limit: int) -> list[str]:
        if self.index.size == 0 or limit <= 0:
            return []

        matches = self.index.search(
            np.asarray(vector, dtype=np.int8),
            min(limit, self.index.size),
        )

        out: list[str] = []

        for key in matches.keys:
            chunk_id = self._ids.get(int(key))

            if chunk_id is not None:
                out.append(chunk_id)

        return out

    @property
    def size(self) -> int:
        return int(self.index.size)


class FactStore:
    """The fact database; every public call serializes on one lock."""

    def __init__(
        self,
        path: Path,
        *,
        dim: int,
        raw_bm25: BM25Config,
        event_bm25: BM25Config,
        connectivity: int = 16,
    ) -> None:
        self.path = path
        self.dim = dim
        self._raw_config = raw_bm25
        self._event_config = event_bm25
        self._connectivity = connectivity
        self._lock = threading.RLock()
        self._compressor = zstandard.ZstdCompressor(level=3)
        self._decompressor = zstandard.ZstdDecompressor()
        self._db = self._open(path)

        self._raw: dict[str, BM25Index] = {}
        self._event: dict[str, BM25Index] = {}
        self._raw_dense: dict[str, DenseIndex] = {}
        self._event_dense: dict[str, DenseIndex] = {}
        self._loc: dict[str, tuple[str, int, int]] = {}
        self._agent_loc: dict[str, dict[str, tuple[int, int]]] = {}
        self._agent_seq: dict[str, dict[str, int]] = {}
        self._agents: set[str] = set()
        self._count = 0
        self._bytes = 0

        try:
            self._load()

        except Exception:
            self._db.close()
            raise

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

    def _load(self) -> None:
        with self._lock:
            stored = self._db.get(_stat_key("dim"))

            if stored is not None and int(stored.decode()) != self.dim:
                raise ValueError(
                    f"fact store dim {int(stored.decode())} != configured {self.dim}: "
                    f"delete {self.path} to rebuild the derived layer"
                )

            self._db.put(_stat_key("dim"), str(self.dim).encode())

            for key, value in self._db.items(from_key=CHUNK_PREFIX):
                if not key.startswith(CHUNK_PREFIX):
                    break

                decoded = self._decode(value)

                if decoded is None:
                    continue

                meta, raw_vec, event_vec = decoded
                agent, seq, ordinal = _parse_chunk_key(key)
                chunk_id = str(meta.get("chunk_id") or "")

                if not chunk_id:
                    continue

                self._index_add(
                    agent,
                    chunk_id,
                    seq,
                    str(meta.get("text") or ""),
                    str(meta.get("event") or ""),
                    raw_vec,
                    event_vec,
                )

                self._loc[chunk_id] = (agent, seq, ordinal)
                self._agent_loc.setdefault(agent, {})[chunk_id] = (seq, ordinal)
                self._agent_seq.setdefault(agent, {})[chunk_id] = seq
                self._agents.add(agent)
                self._count += 1
                self._bytes += len(value) + KEY_OVERHEAD

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_turn(self, agent_id: str, seq: int, rows: list[ChunkRow]) -> bool:
        """Commit one turn's chunks; False when the marker already exists."""
        with self._lock:
            if self._db.get(mark_key(agent_id, seq)) is not None:
                return False

            encoded: list[tuple[bytes, bytes]] = []

            for row in rows:
                if len(row.raw_vec) != self.dim:
                    raise ValueError(f"raw vector dim {len(row.raw_vec)} != {self.dim}")

                if row.event_vec is not None and len(row.event_vec) != self.dim:
                    raise ValueError(f"event vector dim {len(row.event_vec)} != {self.dim}")

                encoded.append(
                    (
                        chunk_key(agent_id, seq, row.ordinal),
                        self._encode(row),
                    )
                )

            batch = WriteBatch()

            for key, value in encoded:
                batch.put(key, value)

            batch.put(mark_key(agent_id, seq), struct.pack(">I", len(rows)))

            self._db.write(batch)

            for row in rows:
                self._index_add(
                    agent_id,
                    row.chunk_id,
                    seq,
                    row.text,
                    row.event,
                    row.raw_vec,
                    row.event_vec,
                )

                self._loc[row.chunk_id] = (agent_id, seq, row.ordinal)
                self._agent_loc.setdefault(agent_id, {})[row.chunk_id] = (
                    seq,
                    row.ordinal,
                )
                self._agent_seq.setdefault(agent_id, {})[row.chunk_id] = seq
                self._agents.add(agent_id)
                self._count += 1

            self._bytes += sum(len(value) + KEY_OVERHEAD for _, value in encoded)

            return True

    def has_turn(self, agent_id: str, seq: int) -> bool:
        with self._lock:
            return self._db.get(mark_key(agent_id, seq)) is not None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}

        with self._lock:
            for chunk_id in chunk_ids:
                location = self._loc.get(chunk_id)

                if location is None:
                    continue

                agent_id, seq, ordinal = location
                raw = self._db.get(chunk_key(agent_id, seq, ordinal))

                if raw is None:
                    continue

                decoded = self._decode(raw)

                if decoded is None:
                    continue

                meta, _, _ = decoded
                out[chunk_id] = {**meta, "agent_id": agent_id, "seq": seq}

        return out

    def seq_map(self, agent_id: str) -> dict[str, int]:
        with self._lock:
            return dict(self._agent_seq.get(agent_id, {}))

    def loc_map(self, agent_id: str) -> dict[str, tuple[int, int]]:
        with self._lock:
            return dict(self._agent_loc.get(agent_id, {}))

    def iter_chunks(
        self,
        agent_id: str,
        after: tuple[int, int] | None = None,
        *,
        batch: int = 512,
    ) -> Iterator[dict[str, Any]]:
        """Walk one agent's chunks in (seq, ordinal) order, after a pointer."""
        prefix = CHUNK_PREFIX + agent_id.encode() + SEP

        if after is None:
            cursor = prefix

        else:
            seq, ordinal = after
            cursor = prefix + struct.pack(">QI", seq, ordinal + 1)

        while True:
            rows: list[tuple[bytes, bytes]] = []

            with self._lock:
                for key, value in self._db.items(from_key=cursor):
                    if not key.startswith(prefix):
                        break

                    rows.append((key, value))

                    if len(rows) >= batch:
                        break

            if not rows:
                return

            for key, value in rows:
                decoded = self._decode(value)

                if decoded is None:
                    continue

                meta, _, _ = decoded
                _, seq, ordinal = _parse_chunk_key(key)

                yield {
                    "chunk_id": str(meta.get("chunk_id") or ""),
                    "seq": seq,
                    "ordinal": ordinal,
                    "started_at": float(meta.get("started_at") or 0.0),
                    "ts": str(meta.get("ts") or ""),
                    "text": str(meta.get("text") or ""),
                    "event": str(meta.get("event") or ""),
                }

            if len(rows) < batch:
                return

            cursor = rows[-1][0] + SEP

    def search_raw_bm25(self, agent_id: str, query: str, limit: int) -> list[str]:
        with self._lock:
            index = self._raw.get(agent_id)

            return index.search(query, limit, self._agent_seq.get(agent_id, {})) if index else []

    def search_event_bm25(self, agent_id: str, query: str, limit: int) -> list[str]:
        with self._lock:
            index = self._event.get(agent_id)

            return index.search(query, limit, self._agent_seq.get(agent_id, {})) if index else []

    def search_raw_dense(self, agent_id: str, vector: np.ndarray, limit: int) -> list[str]:
        with self._lock:
            index = self._raw_dense.get(agent_id)

            return index.search(vector, limit) if index else []

    def search_event_dense(self, agent_id: str, vector: np.ndarray, limit: int) -> list[str]:
        with self._lock:
            index = self._event_dense.get(agent_id)

            return index.search(vector, limit) if index else []

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chunks": self._count,
                "agents": len(self._agents),
                "bytes": self._bytes,
                "dim": self.dim,
                "raw_vectors": sum(index.size for index in self._raw_dense.values()),
                "event_vectors": sum(index.size for index in self._event_dense.values()),
                "path": str(self.path),
            }

    def iter_agents(self) -> Iterator[str]:
        with self._lock:
            return iter(sorted(self._agents))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _meta(row: ChunkRow) -> dict[str, Any]:
        return {
            "chunk_id": row.chunk_id,
            "ordinal": row.ordinal,
            "started_at": row.started_at,
            "ts": row.ts,
            "text": row.text,
            "event": row.event,
        }

    def _encode(self, row: ChunkRow) -> bytes:
        meta = json.dumps(
            self._meta(row), ensure_ascii=False, separators=(",", ":")
        ).encode()
        flags = EVENT_FLAG if row.event_vec is not None else 0
        raw = HEADER.pack(len(meta), flags, len(row.raw_vec)) + meta
        raw += row.raw_vec.tobytes()

        if row.event_vec is not None:
            raw += row.event_vec.tobytes()

        return self._compressor.compress(raw)

    def _decode(self, value: bytes) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None] | None:
        try:
            raw = self._decompressor.decompress(value)
            meta_len, flags, dim = HEADER.unpack_from(raw, 0)
            offset = HEADER.size
            meta = json.loads(raw[offset : offset + meta_len])
            offset += meta_len
            raw_vec = np.frombuffer(raw, dtype=np.int8, count=dim, offset=offset).copy()
            offset += dim
            event_vec = None

            if flags & EVENT_FLAG:
                event_vec = np.frombuffer(
                    raw, dtype=np.int8, count=dim, offset=offset
                ).copy()

            return meta, raw_vec, event_vec

        except Exception:
            logger.warning("segv-fact chunk unreadable")

            return None

    def _index_add(
        self,
        agent_id: str,
        chunk_id: str,
        seq: int,
        text: str,
        event: str,
        raw_vec: np.ndarray,
        event_vec: np.ndarray | None,
    ) -> None:
        self._raw.setdefault(agent_id, BM25Index(self._raw_config)).add(chunk_id, text)

        if event:
            self._event.setdefault(agent_id, BM25Index(self._event_config)).add(
                chunk_id, event
            )

        self._raw_dense.setdefault(
            agent_id, DenseIndex(self.dim, self._connectivity)
        ).add(chunk_id, raw_vec)

        if event_vec is not None:
            self._event_dense.setdefault(
                agent_id, DenseIndex(self.dim, self._connectivity)
            ).add(chunk_id, event_vec)
