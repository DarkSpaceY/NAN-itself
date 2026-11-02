"""Smoke test for the injection protocol (ported from
paper_search/nan_core/tests/smoke_injection.py): registration
payload, ok/fail records, per-agent service delta, and the two
delivery shapes (fresh turn, complete turn)."""

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
from nan.turn import Death, Turn, freeze_turn

pytestmark = pytest.mark.asyncio


def fake_service(service_id: str, endpoint: str, mode: str = "inject") -> Any:
    return type(
        "E",
        (),
        {"service_id": service_id, "endpoint": endpoint, "mode": mode, "last_seen": 9e15},
    )()


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

        assert body["agent_id"] == "a1"

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


async def test_injection_and_delta(no_sink: None) -> None:
    server, serve_task, port = await _serve_stub()

    try:
        hub = InjectionHub()

        turn = Turn(agent_id="a1", parent_id="root", depth=1)
        turn_b = Turn(agent_id="b2", parent_id="root", depth=1)

        # 1. no services -> no contributions, no delta
        r, delta = await hub.inject(turn)
        assert r == () and delta == []

        # 2. dead service appears -> error recorded + online delta
        hub._services["dead"] = fake_service("dead", "http://127.0.0.1:1/inject")
        r, delta = await hub.inject(turn)
        assert len(r) == 1 and r[0].ok is False
        assert [(e.section, e.line) for e in delta] == [("service", "+dead online")]

        # 3. delta is per-agent: a fresh agent sees the same transitions once
        _, delta_b = await hub.inject(turn_b)
        assert [(e.section, e.line) for e in delta_b] == [("service", "+dead online")]

        # 4. unchanged -> no delta
        _, delta = await hub.inject(turn)
        assert delta == []

        # 5. live stub joins -> contribution + online delta
        hub._services["live"] = fake_service("live", f"http://127.0.0.1:{port}/inject")
        r, delta = await hub.inject(turn)
        assert any(i.ok and i.text == "SERVICE SAYS HI" for i in r)
        assert [(e.section, e.line) for e in delta] == [("service", "+live online")]

        # 6. service disappears -> offline delta
        del hub._services["dead"]
        _, delta = await hub.inject(turn)
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


async def _serve_recorder() -> tuple[
    uvicorn.Server, asyncio.Task[None], int, list[dict], list[dict]
]:
    """Stub service that records every payload, split by channel."""
    injected: list[dict] = []
    archived: list[dict] = []

    app = FastAPI()

    @app.post("/inject")
    async def inject(request: Request) -> dict[str, str]:
        injected.append(await request.json())

        return {"text": "SERVICE SAYS HI"}

    @app.post("/archive")
    async def archive(request: Request) -> dict[str, bool]:
        archived.append(await request.json())

        return {"ok": True}

    config = uvicorn.Config(app, host="127.0.0.1", port=await _free_port(), log_level="critical")

    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())

    for _ in range(100):
        if server.started:
            break

        await asyncio.sleep(0.05)

    assert server.started

    return server, task, int(config.port or 0), injected, archived


async def _stop(server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    server.should_exit = True

    await task


async def test_delivery_shapes(no_sink: None) -> None:
    server, task, port, injected, archived = await _serve_recorder()

    try:
        hub = InjectionHub()

        hub._services["live"] = fake_service("live", f"http://127.0.0.1:{port}/inject")

        hub._services["sink"] = fake_service(
            "sink", f"http://127.0.0.1:{port}/archive", mode="archive"
        )

        t1 = Turn(agent_id="a1", parent_id=None, depth=0, origin="first")

        await hub.inject(t1)

        done = freeze_turn(replace(t1, reply="R1"))

        await hub.archive(done)

        assert len(injected) == 1 and len(archived) == 1

        fresh = injected[0]

        assert fresh["agent_id"] == "a1"
        assert fresh["origin"] == "first"

        for absent in ("reply", "calls", "results", "llm", "death", "ended_at", "rendered"):
            assert absent not in fresh

        # The archive sees the same turn complete: reply and frozen
        # rendered included, still one flat body.
        complete = archived[0]

        assert complete["agent_id"] == "a1"
        assert complete["origin"] == "first"
        assert complete["reply"] == "R1"
        assert complete["rendered"] is not None
    finally:
        await _stop(server, task)


async def test_archive_carries_the_death(no_sink: None) -> None:
    server, task, port, _, archived = await _serve_recorder()

    try:
        hub = InjectionHub()

        hub._services["sink"] = fake_service(
            "sink", f"http://127.0.0.1:{port}/archive", mode="archive"
        )

        last = freeze_turn(
            replace(
                Turn(agent_id="a1", parent_id=None, depth=0),
                reply="LAST",
                death=Death(reason="finished", report="done"),
            )
        )

        await hub.archive(last)

        assert len(archived) == 1
        assert archived[0]["reply"] == "LAST"
        assert archived[0]["death"] == {"reason": "finished", "report": "done"}
    finally:
        await _stop(server, task)


async def test_archive_failure_is_dropped(no_sink: None) -> None:
    hub = InjectionHub()

    hub._services["gone"] = fake_service("gone", "http://127.0.0.1:1/archive", mode="archive")

    # A dead archive is an observer that missed a turn, never a
    # reason to fail the turn that was just committed.
    await hub.archive(Turn(agent_id="a1", parent_id=None, depth=0))


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
        contributions = await client.inject(
            Turn(agent_id="a1", parent_id=None, depth=0),
            [fake_service("plain", f"http://127.0.0.1:{port}/inject")],
        )
    finally:
        await client.close()

        await _stop(server, task)

    assert len(contributions) == 1
    assert contributions[0].ok is False
    assert '{"text": <str>}' in contributions[0].text
