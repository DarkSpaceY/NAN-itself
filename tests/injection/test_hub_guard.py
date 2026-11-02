"""Injection hub guard: the registry rejects non-loopback
endpoints and hostile browser-borne requests."""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from nan.config import settings
from nan.injection import InjectionHub

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

    raise TimeoutError("hub did not come up")


async def test_hub_rejects_non_loopback_endpoint_and_hostile_entry(no_sink: None) -> None:
    port = _free_port()

    settings.injection.port = port

    hub = InjectionHub()

    await hub.start()

    try:
        await _wait_port(port)

        base = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(trust_env=False) as client:
            # an internet endpoint would ship every turn payload out
            out = await client.post(
                f"{base}/register",
                json={"service_id": "evil", "endpoint": "https://evil.com/turn"},
            )

            assert out.json()["status"] == "rejected"
            assert "evil" not in hub._services

            local = await client.post(
                f"{base}/register",
                json={"service_id": "record", "endpoint": f"{base}/turn"},
            )

            assert local.json()["status"] == "registered"
            assert "record" in hub._services

            # DNS rebinding: foreign Host on the loopback socket
            forged = await client.get(f"{base}/services", headers={"Host": f"evil.com:{port}"})
            assert forged.status_code == 403

            # hostile page: foreign Origin
            hostile = await client.get(f"{base}/services", headers={"Origin": "http://evil.com"})
            assert hostile.status_code == 403
    finally:
        await hub.stop()
