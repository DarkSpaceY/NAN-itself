"""
The doubles a service's tests run against: a hub that records instead
of routing, and the two replies the protocol fixes.

A service test builds the real app, points its link at StubHub, and
then checks the same three things every service promises: what it
registers, what it answers on a turn, and that it unregisters.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx

from .config import ServiceConfig

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

FRESH_TURN: dict[str, Any] = {
    "agent_id": "test",
    "parent_id": None,
    "depth": 0,
    "origin": "first",
    "persona": None,
    "inbox": [],
    "injections": [],
    "delta": [],
    "started_at": 0.0,
}


class StubHub:
    """Stands in for both the hub and the gateway; records every call."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []

        self.deletes: list[str] = []

        # What the hub answers a heartbeat; "unknown" means it forgot
        # this service, as after a restart.
        self.heartbeat = "ok"

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if request.method == "DELETE":
            self.deletes.append(path)

            return httpx.Response(200, json={})

        if request.method == "GET":
            return httpx.Response(200, json={"services": []})

        body = json.loads(request.content or b"{}")

        self.posts.append((path, body))

        if path == "/heartbeat":
            return httpx.Response(200, json={"status": self.heartbeat})

        return httpx.Response(200, json={"status": "registered"})

    def bodies(self, path: str) -> list[dict[str, Any]]:
        """Every body posted to one path, in order."""
        return [body for posted, body in self.posts if posted == path]


def assert_registered(stub: StubHub, config: ServiceConfig) -> None:
    """Whatever else a service does at boot, it registers first, in one shape."""
    assert stub.posts, "the service never reached the hub"

    path, body = stub.posts[0]

    assert path == "/register", path

    assert body == {
        "service_id": config.service_id,
        "endpoint": config.endpoint,
        "mode": config.mode,
    }


def assert_turn_text(client: TestClient, text: str = "") -> None:
    """An inject service answers one shape: {"text": <str>}, and nothing else."""
    reply = client.post("/turn", json=FRESH_TURN)

    assert reply.status_code == 200

    assert reply.json() == {"text": text}


def assert_turn_ack(client: TestClient) -> None:
    """An archive service only has to answer 2xx; the body is ignored."""
    reply = client.post("/turn", json={**FRESH_TURN, "reply": "", "ended_at": 1.0})

    assert 200 <= reply.status_code < 300
