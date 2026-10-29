"""
Boot scripts: the two shells the core launches once at startup, and
the lifetime it owns afterwards.

    builtin/autostart.sh    maintained by the user, runs outside the
                            sandbox with the user's full rights.
    workspace/autostart.sh  writable by agents, runs inside the
                            execution sandbox.

Each script is launched in its own session, so everything it starts
shares one process group. The core keeps that group and stops it on
exit: a boot never inherits a listener from the boot before it.

The group leader has to stay alive for as long as the core does (see
ANCHOR_SECONDS), or its pid -- which is the group id the core signals
at shutdown -- could be reaped and recycled into an unrelated process.

This module knows nothing about settings: the core hands it the
scripts, the environment and the sandbox wrapper it should use.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from loguru import logger

# Seconds the autostart shell lives on past its script. A pid whose
# process has been reaped can be handed to someone else, and the pid
# of a boot script is the group id the core signals at shutdown: the
# leader has to stay alive, or killpg could reach a stranger. The
# value is a portable maximum (BSD and GNU sleep both take it).
ANCHOR_SECONDS = 2_147_483_647

# label, script, run inside the sandbox?
Script = tuple[str, Path, bool]


class Autostart:
    """Launch the boot scripts, then stop what they started."""

    def __init__(
        self,
        scripts: Iterable[Script],
        *,
        env: Mapping[str, str],
        wrap: Callable[[str], str],
    ) -> None:
        self._scripts = tuple(scripts)
        self._env = dict(env)
        self._wrap = wrap

        # Process-group ids, in launch order. Each is the pid of its
        # session leader, which also carries the group id.
        self._pids: list[int] = []
        self._tasks: list[asyncio.Task[None]] = []

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Schedule every script that exists; failures never block the core."""
        loop = asyncio.get_running_loop()

        for label, script, wrapped in self._scripts:
            if not script.is_file():
                continue

            self._tasks.append(
                loop.create_task(self._launch(script, wrapped=wrapped), name=f"autostart:{label}")
            )

    async def _launch(self, script: Path, *, wrapped: bool) -> None:
        try:
            # The script runs normally and the shell then sleeps to
            # keep the group anchored: a boot script returns in
            # milliseconds while the services it started live on, and
            # the group needs a live process for as long as they do
            # (see ANCHOR_SECONDS).
            command = f"bash {shlex.quote(str(script))}; sleep {ANCHOR_SECONDS}"

            if wrapped:
                command = self._wrap(command)

            proc = await asyncio.create_subprocess_shell(
                command,
                env=self._env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )

        except Exception:
            logger.exception("Autostart failed to launch: {}", script)

            return

        # With start_new_session the pid is also the group id. The
        # anchor sleep keeps that leader alive for the core's whole
        # life, so the id cannot be recycled under us.
        self._pids.append(proc.pid)

        logger.info("Autostart launched: {} (pid {})", script, proc.pid)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Stop everything the boot scripts started: gently, then firmly."""
        if self._tasks:
            _, pending = await asyncio.wait(self._tasks, timeout=2.0)

            for task in pending:
                task.cancel()

            await asyncio.gather(*self._tasks, return_exceptions=True)

        if not self._pids:
            return

        for pgid in self._pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGTERM)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0

        while loop.time() < deadline and any(_group_alive(pgid) for pgid in self._pids):
            await asyncio.sleep(0.1)

        for pgid in self._pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)

    except ProcessLookupError:
        return False

    except PermissionError:
        return True

    return True
