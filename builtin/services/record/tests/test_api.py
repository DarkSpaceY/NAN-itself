"""
API tests: the archive endpoint acknowledges and stores, nothing
ever reaches the prompt; the read API serves pages and the
timeline out of the same store.
"""

from __future__ import annotations

import time

import main as record
from fastapi.testclient import TestClient


def _turn(agent_id: str, started_at: float):
    return {
        "agent_id": agent_id,
        "parent_id": None,
        "depth": 0,
        "origin": "normal",
        "persona": None,
        "inbox": [],
        "injections": [],
        "delta": [],
        "reply": "ok",
        "calls": [],
        "results": [],
        "llm": {
            "model": "m",
            "started_at": started_at + 0.1,
            "ended_at": started_at + 0.4,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "finish_reason": "stop",
            "retries": 0,
        },
        "rendered": [{"role": "user", "content": "x"}],
        "started_at": started_at,
        "ended_at": started_at + 0.5,
        "death": None,
    }


def _service(tmp_path) -> tuple[record.RecordService, TestClient]:
    config = record.load_config()

    config.batch_interval = 0.01

    config.batch_size = 16

    service = record.RecordService(config, store_dir=tmp_path / "turns")

    service.store.start()

    return service, TestClient(record.build_app(service))


def _settle(service, count: int, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if service.store.stats()["turns"] >= count:
            return
        time.sleep(0.01)
    raise AssertionError("turn never landed")


def test_turn_endpoint_archives(tmp_path):
    service, client = _service(tmp_path)

    try:
        base = time.time()

        reply = client.post("/turn", json=_turn("a1", base))

        assert reply.status_code == 200
        assert reply.json() == {"ok": "true"}

        _settle(service, 1)

        agents = client.get("/api/agents").json()["agents"]

        assert [a["agent_id"] for a in agents] == ["a1"]
        assert agents[0]["count"] == 1

        page = client.get("/api/agents/a1/turns").json()

        assert page["turns"][0]["seq"] == 0
        assert page["turns"][0]["reply"] == "ok"
        assert page["next_before"] is None

        one = client.get("/api/turns/a1/0").json()

        assert one["rendered"] == [{"role": "user", "content": "x"}]

        assert client.get("/api/turns/a1/7").status_code == 404

        spans = client.get(f"/api/ledger?start=0&end={base + 60}").json()["rows"]

        assert len(spans) == 1
        assert spans[0]["agent_id"] == "a1"
        assert spans[0]["summary"] == "ok"

        hist = client.get(f"/api/activity?start={base - 1}&end={base + 60}").json()

        assert hist["unit"] == "turn"
        assert hist["total"] == 1

        stats = client.get("/api/stats").json()

        assert stats["turns"] == 1
        assert stats["bytes"] > 0
        assert stats["first"] == base

    finally:
        service.store.stop()


def test_health_of_the_static_ui(tmp_path):
    service, client = _service(tmp_path)

    try:
        index = client.get("/")

        assert index.status_code == 200
        assert "record" in index.text

        assert client.get("/app.js").status_code == 200

    finally:
        service.store.stop()
