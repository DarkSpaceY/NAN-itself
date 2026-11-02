"""The gateway WS protocol: hello/history, input routing with mid
dedup, pause/resume and ping-pong."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from pathlib import Path
from typing import Any

import pytest
import websockets

from nan.config import settings
from nan.events import sink
from nan.gateway import Gateway

pytestmark = pytest.mark.asyncio


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_port(port: int) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0

    while loop.time() < deadline:
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        except OSError:
            await asyncio.sleep(0.05)

    raise TimeoutError("gateway did not come up")


async def _until(ws: Any, kind: str) -> dict:
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))

        if message.get("t") == kind:
            return message


async def _serve(gateway: Gateway, port: int) -> tuple[asyncio.Task[None], str]:
    settings.paths.frontend = Path("/nonexistent/nan-frontend")
    task = asyncio.create_task(gateway.serve(open_browser=False))

    await _wait_port(port)

    return task, f"ws://127.0.0.1:{port}/ws"


async def test_hello_carries_state_and_history_replays(no_sink: Any) -> None:
    port = _free_port()
    settings.gateway.port = port

    sink.reset(boot_id="test-boot")
    sink.emit("agent_started", content={"agent_id": "r0"})

    gateway = Gateway(
        state_provider=lambda: {"boot": "test-boot", "model": "fake", "paused": False}
    )
    task, url = await _serve(gateway, port)

    try:
        async with websockets.connect(url, proxy=None) as ws:
            hello = await _until(ws, "hello")

            assert hello["content"] == {"boot": "test-boot", "model": "fake", "paused": False}
            assert hello["seq"] == sink.seq

            replayed = await _until(ws, "agent_started")

            assert replayed["content"] == {"agent_id": "r0"}
    finally:
        await gateway.close()

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)


async def test_input_carries_the_text_and_the_mid(no_sink: Any) -> None:
    port = _free_port()
    settings.gateway.port = port

    received: list[tuple[str, str | None]] = []

    gateway = Gateway(on_input=lambda text, mid: received.append((text, mid)))
    task, url = await _serve(gateway, port)

    try:
        async with websockets.connect(url, proxy=None) as ws:
            await _until(ws, "hello")

            await ws.send(json.dumps({"t": "input", "text": "hello", "mid": "m1"}))

            await asyncio.sleep(0.05)

            assert received == [("hello", "m1")]

            # the same mid is dropped: a reconnect must not double-send
            await ws.send(json.dumps({"t": "input", "text": "hello", "mid": "m1"}))

            await asyncio.sleep(0.05)

            assert received == [("hello", "m1")]

            # a new mid goes through; text is trimmed; empty text is ignored
            await ws.send(json.dumps({"t": "input", "text": " next ", "mid": "m2"}))
            await ws.send(json.dumps({"t": "input", "text": "   ", "mid": "m3"}))

            await asyncio.sleep(0.05)

            assert received == [("hello", "m1"), ("next", "m2")]
    finally:
        await gateway.close()

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)


async def test_ping_pong_and_pause_resume(no_sink: Any) -> None:
    port = _free_port()
    settings.gateway.port = port

    paused: list[bool] = []

    gateway = Gateway(on_pause=paused.append)
    task, url = await _serve(gateway, port)

    try:
        async with websockets.connect(url, proxy=None) as ws:
            await _until(ws, "hello")

            await ws.send(json.dumps({"t": "ping"}))

            assert await _until(ws, "pong") == {"t": "pong", "content": {}}

            await ws.send(json.dumps({"t": "pause"}))
            await ws.send(json.dumps({"t": "resume"}))

            await asyncio.sleep(0.05)

            assert paused == [True, False]

            # malformed input is ignored, and the connection stays live
            await ws.send("not json")

            await ws.send(json.dumps({"t": "ping"}))

            assert await _until(ws, "pong") == {"t": "pong", "content": {}}
    finally:
        await gateway.close()

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)


async def test_unknown_message_type_is_ignored(no_sink: Any) -> None:
    port = _free_port()
    settings.gateway.port = port

    called: list[Any] = []

    gateway = Gateway(on_input=lambda text, mid: called.append(text), on_pause=called.append)
    task, url = await _serve(gateway, port)

    try:
        async with websockets.connect(url, proxy=None) as ws:
            await _until(ws, "hello")

            await ws.send(json.dumps({"t": "unknown"}))

            await asyncio.sleep(0.05)

            assert called == []
    finally:
        await gateway.close()

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)
