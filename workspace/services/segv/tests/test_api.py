from __future__ import annotations

import time

from fastapi.testclient import TestClient
from segv.app import SegvService, build_app
from segv.record import TurnRecord


def _client(service: SegvService) -> TestClient:
    return TestClient(build_app(service))


def _record(ts: float, reply: str = "hi") -> TurnRecord:
    return TurnRecord(
        agent_id="root",
        started_at=ts,
        ended_at=ts + 1.0,
        messages=[{"role": "user", "content": "hello"}],
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
        raw={
            "reply": reply,
            "rendered": [{"role": "user", "content": "hello"}],
        },
    )


def test_web_panel_is_readable_from_a_sandboxed_iframe(config) -> None:
    service = SegvService(config, links=[])

    with _client(service) as client:
        for path in ("/app.js", "/api/stats"):
            response = client.get(path, headers={"Origin": "null"})

            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == "*"


def test_trace_api(config) -> None:
    service = SegvService(config, links=[])

    with _client(service) as client:
        base = time.time() - 60.0

        assert service.store.append(_record(base, "first"))
        assert service.store.append(_record(base + 1, "second"))

        # Redelivery is idempotent at the store.
        assert service.store.append(_record(base, "first")) is None

        stats = client.get("/api/stats").json()

        assert stats["turns"] == 2
        assert stats["agents"] == 1

        agents = client.get("/api/agents").json()["agents"]

        assert agents[0]["agent_id"] == "root"
        assert agents[0]["count"] == 2

        one = client.get("/api/turns/root/0").json()

        assert one["raw"]["reply"] == "first"
        assert one["messages"] == [{"role": "user", "content": "hello"}]

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

        grep = client.post(
            "/api/search",
            json={"agent_id": "root", "query": "FIRST", "mode": "grep"},
        ).json()

        assert grep["mode"] == "grep"
        assert [hit["seq"] for hit in grep["hits"]] == [0]
        assert "first" in grep["hits"][0]["snippet"]
        assert grep["scanned"] == 2

        assert (
            client.post(
                "/api/search", json={"agent_id": "root", "query": "  "}
            ).status_code
            == 400
        )

        assert client.post(
            "/api/search",
            json={"agent_id": "nobody", "query": "x", "mode": "grep"},
        ).json()["hits"] == []

        invalid = client.post(
            "/api/recall",
            json={"agent_id": "root", "query": "x", "mode": "nope"},
        )

        assert invalid.status_code == 400


def test_ui_is_served(config) -> None:
    service = SegvService(config, links=[])

    with _client(service) as client:
        response = client.get("/")

        assert response.status_code == 200
        assert "<title>segv</title>" in response.text
