"""Smoke test for the injection protocol (ported from
paper_search/nan_core/tests/smoke_injection.py): registration
payload, ok/fail records, per-agent service delta."""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request

from nan.injection import InjectionHub
from nan.turn import Turn

pytestmark = pytest.mark.asyncio


def fake_service(service_id: str, endpoint: str) -> Any:
    return type("E", (), {"service_id": service_id, "endpoint": endpoint, "last_seen": 9e15})()


async def _free_port() -> int:
    sock = socket.socket()

    sock.bind(("127.0.0.1", 0))

    port = int(sock.getsockname()[1])

    sock.close()

    return port


async def _serve_stub() -> tuple[uvicorn.Server, asyncio.Task[None], int]:
    app = FastAPI()

    @app.post("/inject")
    async def inject(request: Request) -> dict[str, str]:
        body = await request.json()

        assert body["turn"]["agent_id"] == "a1"

        return {"text": "SERVICE SAYS HI"}

    config = uvicorn.Config(app, host="127.0.0.1", port=await _free_port(), log_level="critical")

    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())

    for _ in range(100):
        if server.started:
            break

        await asyncio.sleep(0.05)

    assert server.started

    return server, task, int(config.port or 0)


async def test_injection_collect_and_delta(no_sink: None) -> None:
    server, serve_task, port = await _serve_stub()

    try:
        hub = InjectionHub(request_timeout=2.0)

        turn = Turn(agent_id="a1", parent_id="root", depth=1)
        turn_b = Turn(agent_id="b2", parent_id="root", depth=1)

        # 1. no services -> no contributions, no delta
        r, delta = await hub.collect(turn)
        assert r == () and delta == []

        # 2. dead service appears -> error recorded + online delta
        hub._services["dead"] = fake_service("dead", "http://127.0.0.1:1/inject")
        r, delta = await hub.collect(turn)
        assert len(r) == 1 and r[0].ok is False
        assert [(e.section, e.line) for e in delta] == [("service", "+dead online")]

        # 3. delta is per-agent: a fresh agent sees the same transitions once
        _, delta_b = await hub.collect(turn_b)
        assert [(e.section, e.line) for e in delta_b] == [("service", "+dead online")]

        # 4. unchanged -> no delta
        _, delta = await hub.collect(turn)
        assert delta == []

        # 5. live stub joins -> contribution + online delta
        hub._services["live"] = fake_service("live", f"http://127.0.0.1:{port}/inject")
        r, delta = await hub.collect(turn)
        assert any(i.ok and i.text == "SERVICE SAYS HI" for i in r)
        assert [(e.section, e.line) for e in delta] == [("service", "+live online")]

        # 6. service disappears -> offline delta
        del hub._services["dead"]
        _, delta = await hub.collect(turn)
        assert [(e.section, e.line) for e in delta] == [("service", "-dead offline")]
    finally:
        server.should_exit = True

        await serve_task


async def test_heartbeat_ttl_expiry(no_sink: None) -> None:
    hub = InjectionHub(request_timeout=2.0, heartbeat_ttl=0.05)

    hub._services["ghost"] = fake_service("ghost", "http://127.0.0.1:1/inject")

    # fresh registration: alive
    hub._services["ghost"].last_seen = time.time()

    live = hub._live_services()
    assert [e.service_id for e in live] == ["ghost"]

    # no heartbeat inside the TTL window: silently expired
    await asyncio.sleep(0.1)

    assert hub._live_services() == []
