"""Smoke test for the injection protocol (ported from
paper_search/nan_core/tests/smoke_injection.py): registration
payload, ok/fail records, per-agent service delta."""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import replace
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request

from nan.config import settings
from nan.injection import InjectionHub
from nan.injection.client import InjectionClient
from nan.turn import Turn, freeze_turn

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
        hub = InjectionHub()

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
    settings.injection.heartbeat_ttl = 0.05

    hub = InjectionHub()

    hub._services["ghost"] = fake_service("ghost", "http://127.0.0.1:1/inject")

    # fresh registration: alive
    hub._services["ghost"].last_seen = time.time()

    live = hub._live_services()
    assert [e.service_id for e in live] == ["ghost"]

    # no heartbeat inside the TTL window: silently expired
    await asyncio.sleep(0.1)

    assert hub._live_services() == []


async def _serve_recorder() -> tuple[uvicorn.Server, asyncio.Task[None], int, list[dict]]:
    """Stub service that records every payload body it receives."""
    bodies: list[dict] = []

    app = FastAPI()

    @app.post("/inject")
    async def inject(request: Request) -> dict[str, str]:
        bodies.append(await request.json())

        return {"text": "SERVICE SAYS HI"}

    config = uvicorn.Config(app, host="127.0.0.1", port=await _free_port(), log_level="critical")

    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())

    for _ in range(100):
        if server.started:
            break

        await asyncio.sleep(0.05)

    assert server.started

    return server, task, int(config.port or 0), bodies


async def _stop(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    server.should_exit = True

    await task


async def test_payload_fresh_turn_and_complete_prev(no_sink: None) -> None:
    server, task, port, bodies = await _serve_recorder()

    try:
        hub = InjectionHub()

        hub._services["live"] = fake_service("live", f"http://127.0.0.1:{port}/inject")

        t1 = Turn(agent_id="a1", parent_id=None, depth=0, state="start")
        await hub.collect(t1)

        t1_done = freeze_turn(replace(t1, reply="R1"))

        t2 = Turn(agent_id="a1", parent_id=None, depth=0, state="running")
        await hub.collect(t2, prev_turn=t1_done)

        assert len(bodies) == 2

        first, second = bodies

        # fresh shape: static pre-call fields only
        assert first["turn"]["agent_id"] == "a1"
        assert first["turn"]["state"] == "start"
        assert first["prev_turn"] is None
        for absent in ("reply", "calls", "results", "llm", "ended_at", "rendered"):
            assert absent not in first["turn"]

        # prev_turn rides complete: reply + frozen rendered included
        assert second["prev_turn"]["reply"] == "R1"
        assert second["prev_turn"]["rendered"] is not None
        assert second["prev_turn"]["state"] == "start"
        assert "rendered" not in second["turn"]
        assert second["turn"]["state"] == "running"
    finally:
        await _stop(server, task)


async def test_exit_delivery_payload(no_sink: None) -> None:
    server, task, port, bodies = await _serve_recorder()

    try:
        hub = InjectionHub()

        hub._services["live"] = fake_service("live", f"http://127.0.0.1:{port}/inject")

        t1 = freeze_turn(replace(Turn(agent_id="a1", parent_id=None, depth=0), reply="LAST"))

        # the exit is the same collect call with no arriving turn
        await hub.collect(None, prev_turn=replace(t1, state="end"))

        assert len(bodies) == 1

        assert bodies[0]["turn"] is None
        assert bodies[0]["prev_turn"]["state"] == "end"
        assert bodies[0]["prev_turn"]["reply"] == "LAST"
    finally:
        await _stop(server, task)


async def test_reply_protocol_is_single_shape() -> None:
    """The reply contract is {"text": str}; any other shape fails."""
    app = FastAPI()

    @app.post("/inject")
    async def inject() -> dict[str, int]:
        return {"text": 5}

    port = await _free_port()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")

    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())

    for _ in range(100):
        if server.started:
            break

        await asyncio.sleep(0.05)

    client = InjectionClient()

    try:
        contributions = await client.collect(
            Turn(agent_id="a1", parent_id=None, depth=0),
            [fake_service("plain", f"http://127.0.0.1:{port}/inject")],
        )
    finally:
        await client.close()

        await _stop(server, task)

    assert len(contributions) == 1
    assert contributions[0].ok is False
    assert '{"text": <str>}' in contributions[0].text
