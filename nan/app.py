"""
Composition root: core process = agents + injection registry +
gateway + execution sandbox.

Every component reads the process-wide `settings`; this module wires
input and pause and owns the process lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import uuid
from pathlib import Path

from loguru import logger

from nan.agent import Agent
from nan.agent.handle import Book
from nan.agent.sandbox import Sandbox
from nan.config import settings
from nan.events import sink
from nan.gateway import Gateway
from nan.injection import InjectionHub
from nan.llm import LLMProvider

_PROXY_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


async def run_core() -> None:
    # Strip ambient proxies.
    for key in _PROXY_KEYS:
        os.environ.pop(key, None)

    llm = LLMProvider()

    book = Book()

    hub = InjectionHub()

    settings.paths.workspace.mkdir(parents=True, exist_ok=True)

    sandbox = Sandbox()

    root = Agent(book=book, llm=llm, injections=hub, sandbox=sandbox)

    # ------------------------------------------------------------------
    # Gateway: subscribes to the `sink` singleton on its own.
    # ------------------------------------------------------------------

    BOOT_ID = uuid.uuid4().hex[:12]

    sink.reset(boot_id=BOOT_ID)

    def ingest(text: str, mid: str | None = None) -> None:
        # User input enters the root handle directly.
        asyncio.get_running_loop().create_task(root.receive_up(text, "user"))

        sink.emit(
            "user_input",
            content={"text": text} | ({"mid": mid} if mid else {}),
        )

    def on_pause(paused: bool) -> None:
        # Pause cancels the in-flight turn (discarded whole);
        # resume opens a clean turn.
        if paused:
            root.pause()
        else:
            root.resume()

    gateway = Gateway(
        on_input=ingest,
        on_pause=on_pause,
        state_provider=lambda: {
            "boot": BOOT_ID,
            "model": llm.model,
            "paused": root.paused,
        },
    )

    gateway_task: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[bool] | None = None

    loop = asyncio.get_running_loop()

    try:
        await hub.start()

        if settings.autostart.enabled:
            _run_autostart(sandbox)

        root.start()

        gateway_task = asyncio.create_task(gateway.serve(open_browser=True), name="gateway")

        stop = asyncio.Event()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: _shutdown(root, stop))

        logger.info(
            "nan running: gateway http://{}:{}, injection registry :{}, workspace {}, sandbox {}",
            settings.gateway.host,
            settings.gateway.port,
            settings.injection.port,
            settings.paths.workspace,
            sandbox.available and "on" or "degraded",
        )

        stop_task = asyncio.create_task(stop.wait(), name="stop")

        # The core lives until a stop signal or until the gateway dies.
        done, _ = await asyncio.wait({stop_task, gateway_task}, return_when=asyncio.FIRST_COMPLETED)

        if gateway_task in done:
            # Propagate the gateway's exception.
            gateway_task.result()

    finally:
        _shutdown(root)

        if stop_task is not None and not stop_task.done():
            stop_task.cancel()

            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stop_task

        if root._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await root._task

        try:
            await gateway.close()

        except BaseException:
            logger.exception("Gateway shutdown failed")

        if gateway_task is not None and not gateway_task.done():
            gateway_task.cancel()

            with contextlib.suppress(asyncio.CancelledError, Exception):
                await gateway_task

        await hub.stop()

        await llm.close()

        logger.info("nan stopped")


def _run_autostart(sandbox: Sandbox) -> None:
    """
    Launch the two boot scripts once at boot, each with its own
    authority:

        builtin/autostart.sh    maintained by the user, runs
                                outside the sandbox with the user's
                                full rights.
        workspace/autostart.sh  writable by agents and runs inside
                                the execution sandbox.
    """
    scripts = (
        ("user", settings.paths.autostart, False),
        ("agent", settings.paths.agent_autostart, True),
    )

    env = {
        **os.environ,
        "NAN_CORE_HOME": str(settings.paths.package),
        "NAN_WORKSPACE": str(settings.paths.workspace),
        "NAN_INJECTION_PORT": str(settings.injection.port),
        "NAN_CORE_PID": str(os.getpid()),
    }

    async def launch(script: Path, *, wrapped: bool) -> None:
        try:
            command = f"bash {shlex.quote(str(script))}"

            if wrapped:
                command = sandbox.wrap(command)

            proc = await asyncio.create_subprocess_shell(
                command,
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

        except Exception:
            logger.exception("Autostart failed to launch: {}", script)

            return

        logger.info("Autostart launched: {} (pid {})", script, proc.pid)

    for label, script, wrapped in scripts:
        if not script.is_file():
            continue

        asyncio.get_running_loop().create_task(
            launch(script, wrapped=wrapped), name=f"autostart:{label}"
        )


def _shutdown(root: Agent, stop_event: asyncio.Event | None = None) -> None:
    logger.info("Stop signal received")

    root.request_death("shutdown")

    if stop_event is not None:
        stop_event.set()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_core())


if __name__ == "__main__":
    main()
