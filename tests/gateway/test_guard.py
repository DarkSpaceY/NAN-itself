"""Gateway entry guard: forged Host (DNS rebinding) and hostile
Origin are rejected; loopback requests without Origin pass."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from pathlib import Path

import httpx
import pytest

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


async def test_entry_guard_rejects_forged_host_and_origin() -> None:
    port = _free_port()
    settings.gateway.port = port

    settings.paths.frontend = Path("/nonexistent/nan-frontend")
    gateway = Gateway()
    task = asyncio.create_task(gateway.serve(open_browser=False))

    try:
        await _wait_port(port)
        base = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient(trust_env=False) as client:
            # DNS rebinding: attacker's domain resolved to loopback
            # arrives with a foreign Host header.
            r = await client.get(f"{base}/", headers={"Host": f"evil.com:{port}"})
            assert r.status_code == 403

            # Browser-borne request from a hostile page: loopback
            # Host, non-loopback Origin.
            r = await client.get(f"{base}/", headers={"Origin": "http://evil.com"})
            assert r.status_code == 403

            # Native client (no Origin) on a loopback Host passes
            # the guard; without a frontend there is no route, so
            # a 404 means the guard let it through.
            r = await client.get(f"{base}/")
            assert r.status_code == 404

            r = await client.get(f"{base}/", headers={"Origin": base})
            assert r.status_code == 404
    finally:
        await gateway.close()

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)
