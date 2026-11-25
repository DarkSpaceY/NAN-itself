from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from segv.app import SegvService, build_app
from segv.cache.config import CacheConfig
from segv.cache.layer import CacheLayer
from segv.fact.config import SlmConfig
from segv.fact.layer import FactLayer
from segv_nan.app import install
from segv_nan.protocol import inject_request, strip_injections, turn_record
from test_cache import FakeCompressor, _ingest, _reply
from test_fact import FakeEmbedder, FakeReranker, FakeSlm, _fact_config


def _turn(ts: float = 1000.0, reply: str = "done") -> dict[str, Any]:
    return {
        "agent_id": "root",
        "started_at": ts,
        "ended_at": ts + 1.0,
        "epoch": 1,
        "depth": 2,
        "parent_id": "boss",
        "origin": "subagent",
        "inbox": [{"text": "alpha"}],
        "rendered": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": (
                    "<injection>\n<service id=\"segv\">\nMEMORY_SECRET\n</service>\n"
                    "</injection>\n\nhello"
                ),
            },
        ],
        "reply": reply,
        "thinking": [{"text": "think"}],
        "calls": [{"name": "x", "arguments": {}}],
        "results": [{"output": "ok"}, {"output": None}],
        "llm": {
            "model": "m",
            "started_at": ts,
            "ended_at": ts + 1.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "retries": 1,
        },
        "death": {"reason": "budget"},
    }


def _app(service: SegvService):
    return build_app(service, routes=lambda app: install(app, service))


def test_nan_turn_becomes_a_neutral_record() -> None:
    record = turn_record(_turn())

    assert record is not None
    assert record.summary == "done"
    assert record.epoch == 1
    assert record.depth == 2
    assert record.parent_id == "boss"
    assert record.origin == "subagent"
    assert record.death == "budget"
    assert record.llm_started_at == 1000.0
    assert record.messages[1]["content"] == "hello"
    assert record.stats == {
        "tokens": 15,
        "retries": 1,
        "calls": 1,
        "errors": 1,
        "reply_chars": 4,
        "thinking_chars": 5,
    }
    # The payload is archived verbatim for the panel; the core never reads it.
    assert record.raw["reply"] == "done"


def test_nan_turn_without_identity_is_dropped() -> None:
    assert turn_record({"reply": "x"}) is None
    assert turn_record({"agent_id": "root", "started_at": "later"}) is None


def test_inject_request_uses_the_inbox_text() -> None:
    agent_id, epoch, query = inject_request(
        {
            "agent_id": " root ",
            "epoch": "2",
            "inbox": [{"text": "  alpha  "}, {"text": "beta"}, {"text": ""}],
        }
    )

    assert (agent_id, epoch, query) == ("root", 2, "alpha\nbeta")


def test_inject_request_keeps_head_and_tail_of_a_long_inbox() -> None:
    long = "HEAD" + "x" * 3000 + "TAIL"

    _, _, query = inject_request({"inbox": [{"text": long}]})

    assert query.startswith("HEAD")
    assert query.endswith("TAIL")
    assert "[...]" in query
    assert len(query) <= 2000


def test_strip_injections_removes_every_service() -> None:
    content = (
        "<injection>\n<service id=\"segv\">\nmine\n</service>\n\n"
        '<service id="other">\ntheirs\n</service>\n</injection>\n'
        '<inbox>\n<user id="u">\nkeep me\n</user>\n</inbox>'
    )

    cleaned = strip_injections(content)

    assert "mine" not in cleaned
    assert "theirs" not in cleaned
    assert "keep me" in cleaned


def test_turn_endpoint_archives_clean_messages_and_the_raw_payload(config) -> None:
    service = SegvService(config, links=[])

    with TestClient(_app(service)) as client:
        assert client.post("/turn", json=_turn()).json() == {"status": "ok"}

        # The hub may replay a turn after a core restart.
        assert client.post("/turn", json=_turn()).json() == {"status": "ok"}

        # An unusable turn is dropped, but the hub still gets its ack.
        assert client.post("/turn", json={"reply": "x"}).json() == {"status": "ok"}

        turns = client.get("/api/agents/root/turns").json()["turns"]

        assert len(turns) == 1

        turn = turns[0]

        assert turn["raw"]["reply"] == "done"
        assert "MEMORY_SECRET" not in turn["messages"][1]["content"]
        assert turn["messages"][1]["content"] == "hello"


def test_inject_endpoint_returns_the_epoch_start_block(config) -> None:
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

    body = {"agent_id": "root", "epoch": 0, "inbox": [{"text": "alpha"}]}

    with TestClient(_app(service)) as client:
        response = client.post("/inject", json=body)

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

        second = client.post("/inject", json=body).json()["text"]

        assert "<memory>" not in second


def test_inject_endpoint_never_raises(config) -> None:
    service = SegvService(config, links=[])

    def boom(_agent: str, _epoch: int, _query: str) -> str:
        raise RuntimeError("boom")

    service.cache.inject = boom

    with TestClient(_app(service)) as client:
        response = client.post("/inject", json={"agent_id": "root"})

        assert response.status_code == 200
        assert response.json() == {"text": ""}
