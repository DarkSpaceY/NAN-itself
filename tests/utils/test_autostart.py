"""Autostart: the core owns the lifetime of what the boot scripts
start. The group id it keeps at shutdown must stay anchored to a live
process, or a reaped pid could be recycled and killpg would reach a
stranger; and the whole group has to be gone once the core exits."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from nan.utils.autostart import Autostart

pytestmark = pytest.mark.asyncio


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)

    except ProcessLookupError:
        return False

    except PermissionError:
        return True

    return True


async def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()

    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if not _exists(pid):
            return True

        await asyncio.sleep(0.05)

    return False


def _boot_script(tmp_path: Path) -> tuple[Path, Path]:
    """A script that leaves one background process behind and returns."""
    script = tmp_path / "autostart.sh"

    marker = tmp_path / "child.pid"

    script.write_text(
        f'#!/usr/bin/env bash\nsleep 300 & echo $! > "{marker}"\n',
        encoding="utf-8",
    )

    return script, marker


def _boot(tmp_path: Path, script: Path) -> Autostart:
    autostart = Autostart(
        (("user", script, False), ("agent", tmp_path / "missing.sh", True)),
        env=dict(os.environ),
        wrap=lambda command: command,
    )

    autostart.start()

    return autostart


async def test_autostart_group_stays_anchored_and_is_stopped(tmp_path: Path) -> None:
    script, marker = _boot_script(tmp_path)

    autostart = _boot(tmp_path, script)

    await asyncio.gather(*autostart._tasks)

    assert len(autostart._pgids) == 1

    pgid = autostart._pgids[0]

    for _ in range(100):
        if marker.is_file() and marker.read_text().strip():
            break

        await asyncio.sleep(0.05)

    child = int(marker.read_text().strip())

    # The script has returned (it wrote the pid on its way out) while
    # its background process runs on: the session leader is still the
    # process we recorded, so the group id is not a recycled pid.
    assert os.getpgid(pgid) == pgid

    assert _exists(child)

    await autostart.stop()

    assert await _wait_gone(child)


async def test_autostart_without_scripts_starts_nothing(tmp_path: Path) -> None:
    autostart = _boot(tmp_path, tmp_path / "missing-user.sh")

    assert autostart._tasks == []

    await autostart.stop()


async def _wait_for(path: Path, timeout: float = 5.0) -> str:
    loop = asyncio.get_running_loop()

    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if path.is_file():
            return path.read_text()

        await asyncio.sleep(0.05)

    return ""


async def test_sandbox_env_reaches_only_the_sandboxed_script(tmp_path: Path) -> None:
    # The user's boot script runs with the real environment; the
    # agent's runs inside the sandbox and gets the extra values
    # (the workspace home) without leaking into the user's script.
    user_script = tmp_path / "user.sh"
    agent_script = tmp_path / "agent.sh"
    user_marker = tmp_path / "user.txt"
    agent_marker = tmp_path / "agent.txt"

    for script, marker in ((user_script, user_marker), (agent_script, agent_marker)):
        script.write_text(
            f'#!/usr/bin/env bash\nprintf "%s" "${{NAN_TEST_HOME:-unset}}" > "{marker}"\n',
            encoding="utf-8",
        )

    autostart = Autostart(
        (("user", user_script, False), ("agent", agent_script, True)),
        env=dict(os.environ),
        wrap=lambda command: command,
        sandbox_env={"NAN_TEST_HOME": "/agent/home"},
    )

    autostart.start()

    assert await _wait_for(user_marker) == "unset"
    assert await _wait_for(agent_marker) == "/agent/home"

    await autostart.stop()
