from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from segv import cli
from segv.app import SegvService, build_app
from segv.cache.config import CacheConfig
from segv.cache.layer import CacheLayer, CueRef, render_refs
from segv.cache.snapshot import (
    Entity,
    Item,
    Snapshot,
    SnapshotError,
    parse_snapshot,
)
from segv.config import Config
from segv.fact.config import FactConfig, SlmConfig
from segv.fact.layer import FactLayer
from segv.store import Store
from test_fact import FakeEmbedder, FakeReranker, FakeSlm, _fact_config, _payload


class FakeCompressor:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[str] = []

    def chat(self, system: str, user: str) -> str:
        self.calls.append(user)

        return self.replies.pop(0) if self.replies else "{}"

    def close(self) -> None:
        pass


def _reply(
    goal: str = "build segv cache",
    *,
    unresolved: list[dict[str, str]] | None = None,
    decisions: list[dict[str, str]] | None = None,
    forbidden: list[dict[str, str]] | None = None,
    entities: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps(
        {
            "goal": goal,
            "unresolved_items": unresolved or [],
            "locked_decisions": decisions or [],
            "forbidden_behaviors": forbidden or [],
            "entity_map": entities or [],
        },
        ensure_ascii=False,
    )


def _setup(
    tmp_path: Path,
    *,
    trigger: int = 5,
    tail_cues: int = 0,
    recall_k: int = 1,
    batch_cues: int = 10,
    raw_chars: int = 4000,
    replies: list[str] | None = None,
) -> tuple[Store, FactLayer, CacheLayer, FakeCompressor]:
    l0 = Store(tmp_path / "data")
    fact = FactLayer(
        l0,
        _fact_config(),
        slm=FakeSlm(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )
    config = CacheConfig(
        enabled=True,
        trigger=trigger,
        tail_cues=tail_cues,
        tail_chars=2000,
        batch_cues=batch_cues,
        raw_chars=raw_chars,
        recall_k=recall_k,
        recall_pool=20,
        max_items=20,
        min_retention=0.0,
        slm=SlmConfig(max_tokens=1200),
    )
    compressor = FakeCompressor(replies)
    cache = CacheLayer(l0, fact, config, slm=compressor)
    fact.on_ingested = cache.on_ingested

    return l0, fact, cache, compressor


def _ingest(l0: Store, fact: FactLayer, ts: float, label: str) -> tuple[str, int]:
    payload = _payload(ts, [{"role": "user", "content": f"alpha {label}"}])
    uid = l0.append(payload)

    assert uid

    agent_id, seq = l0.seq_of(uid)

    assert fact.ingest(agent_id, seq) == 1

    return agent_id, seq


def _locs(text: str) -> list[tuple[int, int]]:
    return [(int(seq), int(ordinal)) for seq, ordinal in re.findall(r"\[(\d+):(\d+)\]", text)]


# ----------------------------------------------------------------------
# Snapshot parsing
# ----------------------------------------------------------------------


def test_parse_keeps_ids_and_assigns_the_next_free_one() -> None:
    previous = Snapshot(
        goal="old",
        unresolved_items=[Item("u2", "keep me")],
        locked_decisions=[Item("d1", "X — because Y")],
        entity_map=[Entity("e1", "SEGV", "memory service")],
    )
    raw = (
        "```json\n"
        + json.dumps(
            {
                "goal": "new goal",
                "unresolved_items": [
                    {"id": "u2", "text": "keep me"},
                    {"id": "", "text": "new item"},
                ],
                "locked_decisions": [{"id": "d1", "text": "X — because Y"}],
                "forbidden_behaviors": [{"id": "u2", "text": "duplicate id"}],
                "entity_map": [{"id": "", "name": "cache", "meaning": "snapshot"}],
            }
        )
        + "\n```"
    )

    snapshot = parse_snapshot(raw, previous)

    assert snapshot.goal == "new goal"
    assert [item.id for item in snapshot.unresolved_items] == ["u2", "u3"]
    assert [item.id for item in snapshot.locked_decisions] == ["d1"]
    assert [item.id for item in snapshot.forbidden_behaviors] == ["f1"]
    assert [entity.id for entity in snapshot.entity_map] == ["e2"]


def test_parse_rejects_an_empty_goal_and_a_drop() -> None:
    with pytest.raises(SnapshotError, match="goal"):
        parse_snapshot('{"goal": "", "unresolved_items": []}', Snapshot())

    previous = Snapshot(
        goal="g",
        unresolved_items=[Item("u1", "a"), Item("u2", "b")],
        locked_decisions=[Item("d1", "c")],
        entity_map=[Entity("e1", "n", "m")],
    )

    with pytest.raises(SnapshotError, match="drop"):
        parse_snapshot(
            _reply("g", unresolved=[{"id": "u1", "text": "a"}]),
            previous,
        )


# ----------------------------------------------------------------------
# Rewrites
# ----------------------------------------------------------------------


def test_failed_rewrite_keeps_the_pointer_and_retry_advances_it(tmp_path: Path) -> None:
    l0, fact, cache, compressor = _setup(
        tmp_path, trigger=2, replies=["not json", "still not json"]
    )

    _ingest(l0, fact, 1000.0, "one")
    _ingest(l0, fact, 1001.0, "two")

    assert cache.compress("root") is False
    assert cache.store is not None
    assert cache.store.load("root") is None
    assert cache.state("root")["pending"] == 2

    compressor.replies.append(_reply("goal one"))

    assert cache.compress("root") is True

    state = cache.store.load("root")

    assert state.pointer == (1, 0)
    assert state.snapshot.goal == "goal one"
    assert cache.state("root")["pending"] == 0

    cache.close()
    fact.close()
    l0.close()


def test_worker_rewrites_at_the_threshold(tmp_path: Path) -> None:
    l0, fact, cache, _ = _setup(
        tmp_path, trigger=1, replies=[_reply("worker goal")]
    )

    cache.start()

    try:
        _ingest(l0, fact, 1000.0, "one")

        deadline = time.time() + 5.0

        while time.time() < deadline and cache.state("root")["version"] == 0:
            time.sleep(0.05)

        assert cache.state("root")["version"] == 1
        assert cache.state("root")["snapshot"]["goal"] == "worker goal"

    finally:
        cache.close()
        fact.close()
        l0.close()


def test_rewrite_reads_raw_text_and_stops_at_the_budget(tmp_path: Path) -> None:
    l0, fact, cache, compressor = _setup(
        tmp_path,
        trigger=1,
        raw_chars=80,
        replies=[_reply("raw goal"), _reply("raw goal two")],
    )

    _ingest(l0, fact, 1000.0, "one " + "x" * 60)
    _ingest(l0, fact, 1001.0, "two " + "y" * 60)

    assert cache.compress("root", force=True) is True

    prompt = compressor.calls[0]

    assert "NEW RAW CHUNKS" in prompt
    assert "alpha one" in prompt
    assert "alpha two" not in prompt
    assert "cue" not in prompt

    state = cache.state("root")

    assert state["pointer"] == {"seq": 0, "ordinal": 0}
    assert state["pending"] == 1

    assert cache.compress("root", force=True) is True
    assert "alpha two" in compressor.calls[1]
    assert cache.state("root")["pending"] == 0

    cache.close()
    fact.close()
    l0.close()


def test_rewrite_drops_prompt_envelope_tags(tmp_path: Path) -> None:
    l0, fact, cache, compressor = _setup(
        tmp_path, trigger=1, replies=[_reply("tag goal")]
    )
    payload = _payload(
        1000.0,
        [
            {
                "role": "user",
                "content": '<inbox>\n<user id="u">\nalpha one\n</user>\n</inbox>',
            }
        ],
    )
    uid = l0.append(payload)

    assert uid

    agent_id, seq = l0.seq_of(uid)

    assert fact.ingest(agent_id, seq) == 1
    assert cache.compress("root", force=True) is True

    prompt = compressor.calls[0]

    assert "alpha one" in prompt
    assert "<inbox>" not in prompt
    assert "<user" not in prompt

    cache.close()
    fact.close()
    l0.close()


# ----------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------


def test_new_epoch_injects_snapshot_and_tail_once(tmp_path: Path) -> None:
    l0, fact, cache, _ = _setup(
        tmp_path,
        tail_cues=5,
        recall_k=1,
        replies=[_reply("stable goal")],
    )

    _ingest(l0, fact, 1000.0, "one")
    _ingest(l0, fact, 1001.0, "two")
    assert cache.compress("root", force=True) is True

    _ingest(l0, fact, 1002.0, "three")

    first = cache.inject(
        {"agent_id": "root", "epoch": 1, "inbox": [{"text": "alpha"}]}
    )

    assert "<segv-guide>" in first
    assert "segv help" in first
    assert "<memory>" in first
    assert "<goal>\nstable goal\n</goal>" in first
    assert "<recent_cues>" in first
    assert "[2:0]" in first
    assert (2, 0) in cache.store.seen("root", 1)

    second = cache.inject(
        {"agent_id": "root", "epoch": 1, "inbox": [{"text": "alpha"}]}
    )

    assert "<memory>" not in second
    assert "<recent_cues>" not in second
    assert "[2:0]" not in second

    cache.close()
    fact.close()
    l0.close()


def test_recall_never_repeats_and_survives_a_restart(tmp_path: Path) -> None:
    l0, fact, cache, _ = _setup(tmp_path, trigger=99, tail_cues=0, recall_k=1)

    _ingest(l0, fact, 1000.0, "one")
    _ingest(l0, fact, 1001.0, "two")
    _ingest(l0, fact, 1002.0, "three")

    body = {"agent_id": "root", "epoch": 0, "inbox": [{"text": "alpha"}]}

    first = _locs(cache.inject(body))
    second = _locs(cache.inject(body))

    assert len(first) == 1
    assert len(second) == 1
    assert first[0] != second[0]

    cache.close()

    restarted = CacheLayer(l0, fact, cache.config, slm=FakeCompressor())

    try:
        third = _locs(restarted.inject(body))
    finally:
        restarted.close()

    assert len(third) == 1

    seen = first + second + third

    assert len(set(seen)) == 3

    fact.close()
    l0.close()


def test_inject_endpoint_returns_the_epoch_start_block(tmp_path: Path) -> None:
    config = Config(
        service_id="segv",
        port=9004,
        hub_url="http://127.0.0.1:1",
        data_dir="data",
        max_bytes=0,
        fact=FactConfig(enabled=False),
    )
    config._home = tmp_path

    service = SegvService(config, links=[])
    fact = FactLayer(
        service.store,
        _fact_config(),
        slm=FakeSlm(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )
    cache_config = CacheConfig(
        enabled=True,
        trigger=5,
        tail_cues=5,
        recall_k=1,
        max_items=20,
        min_retention=0.0,
        slm=SlmConfig(max_tokens=1200),
    )
    cache = CacheLayer(
        service.store,
        fact,
        cache_config,
        slm=FakeCompressor([_reply("endpoint goal")]),
    )
    fact.on_ingested = cache.on_ingested

    service.fact = fact
    service.cache = cache

    _ingest(service.store, fact, 1000.0, "one")

    assert cache.compress("root", force=True) is True

    with TestClient(build_app(service)) as client:
        response = client.post(
            "/inject",
            json={"agent_id": "root", "epoch": 0, "inbox": [{"text": "alpha"}]},
        )

        assert response.status_code == 200

        text = response.json()["text"]

        assert "<segv-guide>" in text
        assert "segv help" in text
        assert "<memory>" in text
        assert "endpoint goal" in text

        state = client.get(
            "/api/cache/state", params={"agent_id": "root"}
        ).json()

        assert state["last_epoch"] == 0
        assert state["pointer"] == {"seq": 0, "ordinal": 0}

        second = client.post(
            "/inject",
            json={"agent_id": "root", "epoch": 0, "inbox": [{"text": "alpha"}]},
        ).json()["text"]

        assert "<memory>" not in second


def test_inject_endpoint_never_raises(tmp_path: Path) -> None:
    config = Config(
        service_id="segv",
        port=9004,
        hub_url="http://127.0.0.1:1",
        data_dir="data",
        max_bytes=0,
        fact=FactConfig(enabled=False),
    )
    config._home = tmp_path

    service = SegvService(config, links=[])

    def boom(_body: dict[str, Any]) -> str:
        raise RuntimeError("boom")

    service.cache.inject = boom

    with TestClient(build_app(service)) as client:
        response = client.post("/inject", json={"agent_id": "root"})

        assert response.status_code == 200
        assert response.json() == {"text": ""}

        invalid = client.post(
            "/api/recall",
            json={"agent_id": "root", "query": "x", "mode": "nope"},
        )

        assert invalid.status_code == 400


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_cli_search_reads_the_shell_agent(monkeypatch, capsys) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data or b"{}")

        return _FakeResponse(
            {
                "hits": [
                    {
                        "seq": 3,
                        "ordinal": 1,
                        "started_at": 1700000000.0,
                        "event": "a cue",
                        "text": "a full chunk",
                        "score": 0.5,
                    }
                ],
                "count": 1,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("NAN_AGENT_ID", "root")

    assert cli.main(["search", "alpha", "--json", "--no-ce", "--mode", "bm25"]) == 0

    assert captured["url"].endswith("/api/recall")
    assert captured["body"]["agent_id"] == "root"
    assert captured["body"]["rerank"] is False
    assert captured["body"]["mode"] == "bm25"

    payload = json.loads(capsys.readouterr().out)

    assert payload["hits"][0]["event"] == "a cue"


def test_cli_help_and_missing_agent(monkeypatch, capsys) -> None:
    assert cli.main(["help"]) == 0

    out = capsys.readouterr().out

    assert "search" in out
    assert "state" in out
    assert "tail" in out

    monkeypatch.delenv("NAN_AGENT_ID", raising=False)

    with pytest.raises(SystemExit):
        cli.main(["state"])


def test_render_refs_does_not_repeat_a_leading_timestamp() -> None:
    ts = "2026-11-24T09:58:30Z"
    refs = [
        CueRef(seq=2, ordinal=4, chunk_id="c", ts=ts, text=f"{ts} assistant replied"),
        CueRef(seq=0, ordinal=1, chunk_id="d", ts=ts, text="user asked"),
    ]

    rendered = render_refs("recall", refs)

    assert rendered.count(ts) == 2
    assert f"{ts} {ts}" not in rendered
    assert f"- [2:4] {ts} assistant replied" in rendered
    assert f"- [0:1] {ts} user asked" in rendered
