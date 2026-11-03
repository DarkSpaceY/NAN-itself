"""The injection registry endpoints: register, heartbeat,
unregister and listing."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx
import pytest

from nan.config import Settings
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


async def test_registry_round_trip(no_sink: Any) -> None:
    port = _free_port()

    cfg = Settings()
    cfg.injection.port = port

    hub = InjectionHub(cfg)

    await hub.start()

    try:
        await _wait_port(port)

        base = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(trust_env=False) as client:
            missing = await client.post(f"{base}/register", json={"service_id": "x"})

            assert missing.json() == {
                "status": "rejected",
                "reason": "service_id and endpoint required",
            }

            bad_mode = await client.post(
                f"{base}/register",
                json={"service_id": "x", "endpoint": f"{base}/turn", "mode": "sink"},
            )

            assert bad_mode.json() == {
                "status": "rejected",
                "reason": "mode must be inject or archive",
            }

            registered = await client.post(
                f"{base}/register",
                json={"service_id": "clock", "endpoint": f"{base}/turn", "mode": "archive"},
            )

            assert registered.json() == {"status": "registered"}

            services = await client.get(f"{base}/services")

            listed = services.json()["services"]

            assert [(s["service_id"], s["mode"], s["endpoint"]) for s in listed] == [
                ("clock", "archive", f"{base}/turn")
            ]

            heartbeat = await client.post(f"{base}/heartbeat", json={"service_id": "clock"})

            assert heartbeat.json() == {"status": "ok"}

            unknown = await client.post(f"{base}/heartbeat", json={"service_id": "ghost"})

            assert unknown.json() == {"status": "unknown"}

            unregistered = await client.post(f"{base}/unregister", json={"service_id": "clock"})

            assert unregistered.json() == {"status": "ok"}
            assert hub._services == {}
    finally:
        await hub.stop()


async def test_default_mode_is_inject(no_sink: Any) -> None:
    port = _free_port()

    cfg = Settings()
    cfg.injection.port = port

    hub = InjectionHub(cfg)

    await hub.start()

    try:
        await _wait_port(port)

        base = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(trust_env=False) as client:
            await client.post(
                f"{base}/register",
                json={"service_id": "plain", "endpoint": f"{base}/turn"},
            )

        assert hub._services["plain"].mode == "inject"
    finally:
        await hub.stop()
