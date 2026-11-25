from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from segv.app import SegvService, build_app
from segv.config import CoreConfig
from segv.fact.chunk import build_chunks
from segv.fact.config import ChunkConfig, CueConfig, EmbedConfig, FactConfig
from segv.fact.layer import FactLayer
from segv.fact.store import HEADER, FactStore
from segv.record import TurnRecord
from segv.store import Store


class FakeSlm:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, system: str, user: str) -> str:
        self.calls.append(user)

        return "cue"

    def close(self) -> None:
        pass


class FakeEmbedder:
    """Deterministic bag-of-characters vector; enough for routing tests."""

    dim = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []

        for text in texts:
            vector = np.zeros(self.dim, dtype=np.float32)

            for index, character in enumerate(text):
                vector[index % self.dim] += (ord(character) % 17) / 17.0

            norm = float(np.linalg.norm(vector)) or 1.0

            out.append((vector / norm).tolist())

        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        pass


class FakeReranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        words = set(query.lower().split())

        return [float(len(words & set(text.lower().split()))) for text in texts]


def _record(ts: float, rendered: list[dict[str, Any]], reply: str = "ok") -> TurnRecord:
    return TurnRecord(
        agent_id="root",
        started_at=ts,
        ended_at=ts + 1.0,
        messages=rendered,
        summary=reply,
        stats={
            "tokens": 10,
            "retries": 0,
            "calls": 0,
            "errors": 0,
            "reply_chars": len(reply),
            "thinking_chars": 0,
        },
        llm_started_at=ts,
        llm_ended_at=ts + 1.0,
        raw={"reply": reply},
    )


def _stored(record: TurnRecord) -> dict[str, Any]:
    return record.to_json(
        seq=0, turn_id=f"{record.agent_id}:{record.started_at:.6f}"
    )


def _rendered() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": (
                "<delta>\n<state>boot</state>\n</delta>\n\n"
                '<inbox>\n<user id="u">\nhello alpha\n</user>\n</inbox>'
            ),
        },
        {"role": "assistant", "content": "reply beta", "thinking": [], "tool_calls": []},
        {"role": "tool", "tool_call_id": "c1", "content": "tool gamma"},
    ]


def _fact_config(**chunk: Any) -> FactConfig:
    return FactConfig(
        enabled=True,
        embed=EmbedConfig(dim=FakeEmbedder.dim, batch_size=4),
        cue=CueConfig(workers=1),
        chunk=ChunkConfig(**chunk) if chunk else ChunkConfig(),
    )


def _layer(l0: Store, config: FactConfig, *, slm: FakeSlm | None = None) -> FactLayer:
    return FactLayer(
        l0,
        config,
        slm=slm or FakeSlm(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )


def test_whole_turn_becomes_one_chunk() -> None:
    ts, chunks = build_chunks(_stored(_record(1_700_000_000.0, _rendered())), ChunkConfig())

    assert ts == "2023-11-14T22:13:20Z"
    assert len(chunks) == 1

    text = chunks[0].text

    assert "hello alpha" in text
    assert "reply beta" in text
    assert "tool gamma" in text
    assert "boot" in text
    assert text.count("[2023-11-14T22:13:20Z] user: ") == 1


def test_long_turn_splits_respecting_the_limit_and_order() -> None:
    words = " ".join(f"token{i}" for i in range(120))

    rendered = [
        {"role": "user", "content": f"<inbox>\n<user id=\"u\">\n{words}\n</user>\n</inbox>"},
        {"role": "assistant", "content": "after the split"},
    ]

    _, chunks = build_chunks(_stored(_record(1_700_000_000.0, rendered)), ChunkConfig(max_chars=80))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    assert all(chunk.text.startswith("[2023-11-14T22:13:20Z]") for chunk in chunks)

    flat = " ".join(chunk.text for chunk in chunks)

    assert flat.index("token0") < flat.index("after the split")


def test_one_cue_per_chunk_covers_every_message(tmp_path: Path) -> None:
    l0 = Store(tmp_path / "data")

    l0.append(_record(1_700_000_000.0, _rendered()))

    slm = FakeSlm()

    layer = _layer(l0, _fact_config(), slm=slm)

    assert layer.ingest("root", 0) == 1
    assert len(slm.calls) == 1

    prompt = slm.calls[0]

    assert "Current chunk [2023-11-14T22:13:20Z] (user, assistant, tool):" in prompt
    assert "hello alpha" in prompt
    assert "reply beta" in prompt
    assert "tool gamma" in prompt

    layer.close()
    l0.close()


def test_ingest_stores_two_texts_two_vectors_and_pointer(tmp_path: Path) -> None:
    l0 = Store(tmp_path / "data")

    l0.append(_record(1_700_000_000.0, _rendered()))

    layer = _layer(l0, _fact_config())

    assert layer.ingest("root", 0) == 1
    assert layer.ingest("root", 0) == 0

    assert layer.store is not None
    assert layer.store.stats()["chunks"] == 1

    hits = layer.search("root", "alpha", top_k=5, pool=10)

    assert hits
    assert hits[0]["seq"] == 0
    assert "hello alpha" in hits[0]["text"]
    assert "MEMORY_SECRET" not in hits[0]["text"]
    assert hits[0]["event"]
    assert hits[0]["chunk_id"].endswith("#c0")

    layer.close()
    l0.close()


def test_stored_vectors_are_int8_and_reload_from_disk(tmp_path: Path) -> None:
    l0 = Store(tmp_path / "data")

    l0.append(_record(1_700_000_000.0, _rendered()))

    layer = _layer(l0, _fact_config())

    layer.ingest("root", 0)

    chunk_id = layer.search("root", "alpha", top_k=1, pool=5)[0]["chunk_id"]

    layer.close()

    store = FactStore(
        Path(l0.path) / "fact",
        dim=FakeEmbedder.dim,
        raw_bm25=_fact_config().raw_bm25,
        event_bm25=_fact_config().event_bm25,
    )

    row = store.get_chunks([chunk_id])[chunk_id]

    assert "hello alpha" in row["text"]
    assert store.stats()["chunks"] == 1
    assert store.stats()["raw_vectors"] == 1

    store.close()
    l0.close()


def test_decode_rejects_a_mismatched_vector_dim(tmp_path: Path) -> None:
    store = FactStore(
        tmp_path / "fact",
        dim=FakeEmbedder.dim,
        raw_bm25=_fact_config().raw_bm25,
        event_bm25=_fact_config().event_bm25,
    )

    meta = json.dumps({"chunk_id": "root#c0"}, separators=(",", ":")).encode()
    raw = HEADER.pack(len(meta), 0, FakeEmbedder.dim + 1) + meta
    raw += np.zeros(FakeEmbedder.dim + 1, dtype=np.int8).tobytes()

    assert store._decode(store._compressor.compress(raw)) is None

    store.close()


def test_multi_chunk_turn_indexes_every_chunk(tmp_path: Path) -> None:
    words = " ".join(f"token{i}" for i in range(120))

    rendered = [
        {"role": "user", "content": f"<inbox>\n<user id=\"u\">\n{words}\n</user>\n</inbox>"},
        {"role": "assistant", "content": "after the split"},
    ]

    l0 = Store(tmp_path / "data")

    l0.append(_record(1_700_000_000.0, rendered))

    slm = FakeSlm()

    layer = _layer(l0, _fact_config(max_chars=80), slm=slm)

    assert layer.ingest("root", 0) > 1

    stats = layer.stats()

    assert stats["chunks"] == stats["raw_vectors"] == stats["event_vectors"]
    assert len(slm.calls) == stats["chunks"]

    hits = layer.search("root", "token119", top_k=3, pool=10)

    assert hits
    assert hits[0]["seq"] == 0

    layer.close()
    l0.close()


def test_backfill_ingests_existing_l0_turns(tmp_path: Path) -> None:
    l0 = Store(tmp_path / "data")

    l0.append(_record(1_700_000_000.0, _rendered()))

    layer = _layer(l0, _fact_config())

    layer.start()

    deadline = time.time() + 5.0
    stats: dict[str, Any] = {}

    while time.time() < deadline:
        stats = layer.stats()

        if stats.get("chunks"):
            break

        time.sleep(0.05)

    assert stats["chunks"] == 1

    layer.close()
    l0.close()


def test_dim_mismatch_is_rejected(tmp_path: Path) -> None:
    l0 = Store(tmp_path / "data")

    l0.append(_record(1_700_000_000.0, _rendered()))

    layer = _layer(l0, _fact_config())

    layer.ingest("root", 0)
    layer.close()

    with pytest.raises(ValueError, match="delete"):
        FactStore(
            Path(l0.path) / "fact",
            dim=99,
            raw_bm25=_fact_config().raw_bm25,
            event_bm25=_fact_config().event_bm25,
        )

    l0.close()


def test_service_recall_endpoint(config: CoreConfig) -> None:
    service = SegvService(config)

    fact = _layer(service.store, _fact_config())

    service.fact = fact

    with TestClient(build_app(service)) as client:
        timestamp = 1_700_000_000.0

        assert service.store.append(_record(timestamp, _rendered()))

        deadline = time.time() + 5.0
        stats: dict[str, Any] = {}

        while time.time() < deadline:
            stats = client.get("/api/fact/stats").json()

            if stats.get("chunks"):
                break

            time.sleep(0.05)

        assert stats["enabled"] is True
        assert stats["chunks"] == 1

        response = client.post(
            "/api/recall", json={"agent_id": "root", "query": "alpha", "top_k": 3}
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload["count"] >= 1
        assert "hello alpha" in payload["hits"][0]["text"]

        assert client.post("/api/recall", json={"query": "alpha"}).status_code == 400
        assert client.post("/api/recall", json={"agent_id": "root"}).status_code == 400
