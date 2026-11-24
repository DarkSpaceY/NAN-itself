from __future__ import annotations

import time

from fastapi.testclient import TestClient
from segv.app import SegvService, build_app


def _client(service: SegvService) -> TestClient:
    return TestClient(build_app(service))


def _turn(ts: float, reply: str = "hi") -> dict:
    return {
        "agent_id": "root",
        "parent_id": None,
        "depth": 0,
        "epoch": 0,
        "origin": "normal",
        "started_at": ts,
        "ended_at": ts + 1.0,
        "inbox": [
            {
                "kind": "user",
                "source": "user",
                "text": "hello",
                "event": "send",
                "received_at": ts,
            }
        ],
        "injections": [],
        "delta": [],
        "reply": reply,
        "thinking": [],
        "calls": [],
        "results": [],
        "llm": {
            "model": "m",
            "started_at": ts,
            "ended_at": ts + 1.0,
            "usage": {"prompt_tokens": 8, "completion_tokens": 2},
            "finish_reason": "stop",
            "retries": 0,
        },
        "rendered": [{"role": "user", "content": "hello"}],
        "death": None,
    }


def test_turn_endpoint_and_trace_api(config) -> None:
    service = SegvService(config, links=[])

    with _client(service) as client:
        base = time.time() - 60.0

        assert client.post("/turn", json=_turn(base, "first")).status_code == 200
        assert client.post("/turn", json=_turn(base + 1, "second")).status_code == 200

        # Redelivery is accepted and ignored.
        assert client.post("/turn", json=_turn(base, "first")).status_code == 200

        stats = client.get("/api/stats").json()

        assert stats["turns"] == 2
        assert stats["agents"] == 1

        agents = client.get("/api/agents").json()["agents"]

        assert agents[0]["agent_id"] == "root"
        assert agents[0]["count"] == 2

        one = client.get("/api/turns/root/0").json()

        assert one["reply"] == "first"
        assert one["rendered"] == [{"role": "user", "content": "hello"}]

        assert client.get("/api/turns/root/99").status_code == 404

        ledger = client.get("/api/ledger", params={"start": 0, "limit": 10}).json()

        assert [row["summary"] for row in ledger["rows"]] == ["first", "second"]

        activity = client.get(
            "/api/activity", params={"start": base - 10, "end": base + 10, "buckets": 200}
        ).json()

        assert activity["unit"] == "turn"
        assert activity["total"] == 2

        page = client.get("/api/agents/root/turns", params={"limit": 1}).json()

        assert [t["seq"] for t in page["turns"]] == [1]
        assert page["next_before"] == 1


def test_ui_is_served(config) -> None:
    service = SegvService(config, links=[])

    with _client(service) as client:
        response = client.get("/")

        assert response.status_code == 200
        assert "<title>segv</title>" in response.text
