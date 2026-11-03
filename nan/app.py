"""
Composition root: core process = agents + injection registry +
gateway + execution sandbox.

This module builds the one `Settings` instance and hands it to every
component; it also wires input and pause and owns the process
lifecycle.

One instance drives one process; `run` may be called once. Shutdown
persists: `root.suspend()` saves the snapshot and the next boot
restores from it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from enum import Enum

from loguru import logger

from nan.agent import Agent
from nan.boot import restore
from nan.config import Settings
from nan.events import sink
from nan.execution import Sandbox
from nan.gateway import Gateway
from nan.injection import InjectionHub
from nan.llm import LLMProvider
from nan.persistence import SnapshotStore
from nan.turn import InboxMessage
from nan.utils.autostart import Autostart

_SHUTDOWN_TIMEOUT = 10.0
_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class CoreState(Enum):
    """The process lifetime: INIT -> RUNNING -> STOPPING -> STOPPED."""

    INIT = "init"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class Core:
    """Owns the lifecycle and the wired components."""

    def __init__(self) -> None:
        self._state = CoreState.INIT
        self._boot_id = uuid.uuid4().hex[:12]
        self._stop: asyncio.Event | None = None

        self._settings: Settings | None = None
        self._llm: LLMProvider | None = None
        self._hub: InjectionHub | None = None
        self._sandbox: Sandbox | None = None
        self._snapshots: SnapshotStore | None = None
        self._root: Agent | None = None
        self._gateway: Gateway | None = None
        self._autostart: Autostart | None = None

        self._gateway_task: asyncio.Task[None] | None = None
        self._signal_handlers: list[int] = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        if self._state is not CoreState.INIT:
            raise RuntimeError("Core can only be run once")

        try:
            await self._initialize()
            await self._serve()

        finally:
            await self._shutdown()

    async def _initialize(self) -> None:
        self._stop = asyncio.Event()

        self._settings = Settings()
        self._llm = LLMProvider(self._settings)
        self._hub = InjectionHub(self._settings)

        self._settings.paths.workspace.mkdir(parents=True, exist_ok=True)

        self._sandbox = Sandbox(self._settings)

        # Reset before restore: its events must belong to this boot.
        sink.reset(boot_id=self._boot_id)

        self._snapshots = SnapshotStore(self._settings.paths.snapshots)

        self._root = await restore(
            settings=self._settings,
            llm=self._llm,
            injections=self._hub,
            sandbox=self._sandbox,
            snapshots=self._snapshots,
        )

        if self._root is None:
            self._root = Agent(
                settings=self._settings,
                llm=self._llm,
                injections=self._hub,
                sandbox=self._sandbox,
                snapshots=self._snapshots,
            )

        self._gateway = Gateway(
            settings=self._settings,
            on_input=self._on_input,
            on_pause=self._on_pause,
            state_provider=self._state_provider,
        )

    async def _serve(self) -> None:
        assert self._settings is not None
        assert self._hub is not None and self._root is not None
        assert self._gateway is not None and self._sandbox is not None

        await self._hub.start()

        if self._settings.autostart.enabled:
            self._start_autostart()

        # Restored agents were started by restore; a fresh root starts here.
        if self._root.task is None:
            self._root.start()

        self._gateway_task = asyncio.create_task(
            self._gateway.serve(open_browser=True),
            name="gateway",
        )

        self._install_signals()
        self._state = CoreState.RUNNING

        logger.info(
            "nan running: gateway http://{}:{}, injection registry :{}, workspace {}, sandbox {}",
            self._settings.gateway.host,
            self._settings.gateway.port,
            self._settings.injection.port,
            self._settings.paths.workspace,
            "on" if self._sandbox.available else "degraded",
        )

        await self._wait_for_stop()

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        if self._state is CoreState.STOPPED:
            return

        self._state = CoreState.STOPPING

        # Suspend is synchronous: it survives cancellation of this
        # coroutine, so the snapshot is written before the first await.
        # Everything after it is best-effort cleanup.
        self._suspend_root()

        await self._await_root_task()
        await self._stop_autostart()
        await self._close_gateway()
        await self._stop_hub()
        await self._close_llm()
        self._remove_signals()

        self._state = CoreState.STOPPED

        logger.info("nan stopped")

    def _suspend_root(self) -> None:
        if self._root is None:
            return

        try:
            self._root.suspend()

        except Exception:
            logger.exception("Root suspend failed")

    async def _await_root_task(self) -> None:
        task = self._root.task if self._root is not None else None

        if task is None:
            return

        _, pending = await asyncio.wait({task}, timeout=_SHUTDOWN_TIMEOUT)

        if task in pending:
            logger.warning("Root task did not finish within {}s; cancelling", _SHUTDOWN_TIMEOUT)

            task.cancel()

            _, still_pending = await asyncio.wait({task}, timeout=_SHUTDOWN_TIMEOUT)

            if task in still_pending:
                logger.error("Root task ignored cancellation")

        if task.done() and not task.cancelled():
            error = task.exception()

            if error is not None:
                logger.error("Root task failed during shutdown: {}", error)

    async def _stop_autostart(self) -> None:
        if self._autostart is None:
            return

        # The boot scripts' services are children of this core: stop
        # them while the gateway and the hub are still up, so they can
        # unregister and close their stores, and so the next boot finds
        # the ports free.
        try:
            await self._autostart.stop()

        except Exception:
            logger.exception("Autostart stop failed")

    async def _close_gateway(self) -> None:
        if self._gateway is not None:
            try:
                await self._gateway.close()

            except Exception:
                logger.exception("Gateway close failed")

        task = self._gateway_task

        if task is not None:
            if not task.done():
                task.cancel()

            try:
                await task

            except asyncio.CancelledError:
                pass

            except Exception:
                logger.exception("Gateway task failed during shutdown")

    async def _stop_hub(self) -> None:
        if self._hub is None:
            return

        try:
            await self._hub.stop()

        except Exception:
            logger.exception("Hub stop failed")

    async def _close_llm(self) -> None:
        if self._llm is None:
            return

        try:
            await self._llm.close()

        except Exception:
            logger.exception("LLM close failed")

    # ------------------------------------------------------------------
    # signals
    # ------------------------------------------------------------------

    def _install_signals(self) -> None:
        loop = asyncio.get_running_loop()

        for sig in _SIGNALS:
            try:
                loop.add_signal_handler(sig, self._request_stop)

            except (NotImplementedError, ValueError):
                logger.warning("Signal {} unsupported; relying on KeyboardInterrupt", sig)

            else:
                self._signal_handlers.append(sig)

    def _remove_signals(self) -> None:
        loop = asyncio.get_running_loop()

        for sig in self._signal_handlers:
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(sig)

        self._signal_handlers.clear()

    def _request_stop(self) -> None:
        if self._stop is None or self._stop.is_set():
            return

        if self._state in (CoreState.STOPPING, CoreState.STOPPED):
            return

        logger.info("Stop signal received")

        self._stop.set()

    # ------------------------------------------------------------------
    # serving
    # ------------------------------------------------------------------

    async def _wait_for_stop(self) -> None:
        stop = self._stop
        assert stop is not None and self._gateway_task is not None

        stop_task = asyncio.create_task(stop.wait(), name="stop")

        try:
            done, _ = await asyncio.wait(
                {stop_task, self._gateway_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

        finally:
            if not stop_task.done():
                stop_task.cancel()

                with contextlib.suppress(BaseException):
                    await stop_task

        if self._gateway_task in done:
            self._report_gateway_exit()

            # Gateway is gone; keep serving until a stop signal.
            await stop.wait()

    def _report_gateway_exit(self) -> None:
        task = self._gateway_task
        assert task is not None

        if task.cancelled():
            return

        error = task.exception()

        if error is not None:
            logger.error("Gateway stopped; core continues: {}", error)

        elif self._stop is not None and not self._stop.is_set():
            logger.warning("Gateway exited normally; core continues")

    # ------------------------------------------------------------------
    # gateway callbacks
    # ------------------------------------------------------------------

    def _on_input(self, text: str, mid: str | None = None) -> None:
        if self._state is not CoreState.RUNNING or self._root is None:
            logger.warning("Input dropped: core state is {}", self._state.value)

            return

        # `mid` is the transport-level correlation id; it rides on the
        # event only. The inbox message gets its own id from
        # `InboxMessage.from_user`.
        self._root.deliver(InboxMessage.from_user(text))

        content: dict[str, str] = {"text": text}

        if mid:
            content["mid"] = mid

        sink.emit("user_input", content=content)

    def _on_pause(self, paused: bool) -> None:
        if self._root is None or self._state is not CoreState.RUNNING:
            return

        if paused:
            self._root.pause()

        else:
            self._root.resume()

    def _state_provider(self) -> dict[str, object]:
        return {
            "boot": self._boot_id,
            "model": self._llm.model if self._llm is not None else None,
            "paused": self._root.paused if self._root is not None else False,
        }

    # ------------------------------------------------------------------
    # autostart
    # ------------------------------------------------------------------

    def _start_autostart(self) -> None:
        sandbox = self._sandbox
        assert sandbox is not None and self._settings is not None

        self._autostart = Autostart(
            (
                ("user", self._settings.paths.autostart, False),
                ("agent", self._settings.paths.agent_autostart, True),
            ),
            env={
                **os.environ,
                "NAN_CORE_HOME": str(self._settings.paths.package),
                "NAN_WORKSPACE": str(self._settings.paths.workspace),
                "NAN_INJECTION_PORT": str(self._settings.injection.port),
                "NAN_CORE_PID": str(os.getpid()),
            },
            wrap=sandbox.wrap,
        )

        self._autostart.start()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(Core().run())


if __name__ == "__main__":
    main()
