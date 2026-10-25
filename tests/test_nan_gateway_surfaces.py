"""Gateway surfaces: a service registers and withdraws its own
panel; the table broadcasts live and replays on every connection."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from typing import Any

import httpx
import pytest
import websockets

from nan.config import settings
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
    """Read past unrelated stream events until this type arrives."""
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))

        if message.get("t") == kind:
            return message


async def test_surface_registration_broadcast_and_replay(no_sink: None) -> None:
    port = _free_port()
    settings.gateway.port = port
    gateway = Gateway()
    task = asyncio.create_task(gateway.serve(open_browser=False))

    try:
        await _wait_port(port)

        base = f"http://127.0.0.1:{port}"
        ws_url = f"ws://127.0.0.1:{port}/ws"

        spec = {"service_id": "record", "url": f"http://127.0.0.1:{port}/ui", "title": "record"}

        async with websockets.connect(ws_url, proxy=None) as first:
            assert (await _until(first, "hello"))["t"] == "hello"

            async with httpx.AsyncClient(trust_env=False) as client:
                added = await client.post(f"{base}/surfaces", json=spec)
                assert added.status_code == 200
                assert added.json()["status"] == "ok"

                # internet panel urls are refused
                bad = await client.post(
                    f"{base}/surfaces",
                    json={"service_id": "evil", "url": "https://evil.com/ui", "title": "evil"},
                )
                assert bad.status_code == 400
                assert bad.json()["status"] == "rejected"

            assert await _until(first, "surface_added") == {
                "t": "surface_added",
                "content": spec,
            }

            # a desktop connecting later gets the table replayed
            async with websockets.connect(ws_url, proxy=None) as second:
                assert (await _until(second, "hello"))["t"] == "hello"
                assert (await _until(second, "surface_added"))["content"] == spec

            async with httpx.AsyncClient(trust_env=False) as client:
                removed = await client.delete(f"{base}/surfaces/record")
                assert removed.status_code == 200

                unknown = await client.delete(f"{base}/surfaces/record")
                assert unknown.status_code == 404

            assert await _until(first, "surface_removed") == {
                "t": "surface_removed",
                "content": {"service_id": "record"},
            }
    finally:
        await gateway.close()

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)
