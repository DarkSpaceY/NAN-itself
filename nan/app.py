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
import signal
import uuid

from loguru import logger

from nan.agent import Agent
from nan.agent.sandbox import Sandbox
from nan.config import settings
from nan.events import sink
from nan.gateway import Gateway
from nan.injection import InjectionHub
from nan.llm import LLMProvider
from nan.recovery import SnapshotStore, restore
from nan.turn import InboxMessage
from nan.utils.autostart import Autostart


async def run_core() -> None:
    llm = LLMProvider()

    hub = InjectionHub()

    settings.paths.workspace.mkdir(parents=True, exist_ok=True)

    sandbox = Sandbox()

    # ------------------------------------------------------------------
    # Gateway: subscribes to the `sink` singleton on its own.
    # ------------------------------------------------------------------

    BOOT_ID = uuid.uuid4().hex[:12]

    # Reset before restore: its events must belong to this boot.
    sink.reset(boot_id=BOOT_ID)

    snapshots = SnapshotStore(settings.paths.snapshots)

    root = await restore(
        llm=llm,
        injections=hub,
        sandbox=sandbox,
        snapshots=snapshots,
    )

    if root is None:
        root = Agent(
            llm=llm,
            injections=hub,
            sandbox=sandbox,
            snapshots=snapshots,
        )

    def ingest(text: str, mid: str | None = None) -> None:
        # User input enters the root's inbox directly.
        root.deliver(InboxMessage.from_user(text))

        sink.emit(
            "user_input",
            content={"text": text} | ({"mid": mid} if mid else {}),
        )

    def on_pause(paused: bool) -> None:
        # Pause cancels the in-flight turn where it stands and saves
        # it; resume continues that same turn.
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

    autostart: Autostart | None = None

    loop = asyncio.get_running_loop()

    try:
        await hub.start()

        if settings.autostart.enabled:
            autostart = Autostart(
                (
                    ("user", settings.paths.autostart, False),
                    ("agent", settings.paths.agent_autostart, True),
                ),
                env={
                    **os.environ,
                    "NAN_CORE_HOME": str(settings.paths.package),
                    "NAN_WORKSPACE": str(settings.paths.workspace),
                    "NAN_INJECTION_PORT": str(settings.injection.port),
                    "NAN_CORE_PID": str(os.getpid()),
                },
                wrap=sandbox.wrap,
            )

            autostart.start()

        # Restored agents were started by restore; a fresh root
        # starts here.
        if root.task is None:
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

        # The core lives until a stop signal. A dead gateway is
        # reported and the core keeps running: its failure must not
        # kill the agents or their snapshots.
        done, _ = await asyncio.wait({stop_task, gateway_task}, return_when=asyncio.FIRST_COMPLETED)

        if gateway_task in done:
            error = gateway_task.exception()

            if error is not None:
                logger.error("Gateway stopped; core continues: {}", error)

            await stop_task

    finally:
        _shutdown(root)

        if stop_task is not None and not stop_task.done():
            stop_task.cancel()

            with contextlib.suppress(BaseException):
                await stop_task

        if root._task is not None:
            with contextlib.suppress(BaseException):
                await root._task

        # The boot scripts' services are children of this core: stop
        # them while the gateway and the hub are still up, so they can
        # unregister and close their stores, and so the next boot finds
        # the ports free.
        if autostart is not None:
            await autostart.stop()

        try:
            await gateway.close()

        except BaseException:
            logger.exception("Gateway shutdown failed")

        if gateway_task is not None and not gateway_task.done():
            gateway_task.cancel()

            with contextlib.suppress(BaseException):
                await gateway_task

        await hub.stop()

        await llm.close()

        logger.info("nan stopped")


def _shutdown(root: Agent, stop_event: asyncio.Event | None = None) -> None:
    logger.info("Stop signal received")

    # Shutdown is persistence, not death: the snapshot stays and
    # the next boot restores from it.
    root.suspend()

    if stop_event is not None:
        stop_event.set()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_core())


if __name__ == "__main__":
    main()
