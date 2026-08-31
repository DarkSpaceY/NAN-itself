from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from .agent.core import CoreAgent
from .agent.loop import (
    AgentLoop,
    DEFAULT_TURN_GRACE,
    Inbox,
)
from .config import get_settings
from .events import EventBus
from .gateway import Gateway
from .modules import (
    Facade as ModuleFacade,
)
from .skills import SkillRuntime
from .tools import ProviderRuntime
from .utils.llm import LLMProvider

settings = get_settings()


def _llm_from_config(settings) -> LLMProvider:
    return LLMProvider(
        provider=settings.llm.provider,
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
    )


async def _read_stdin(
    ingest,
) -> None:
    """
    Read stdin directly from the asyncio event loop.

    Do NOT use run_in_executor()/to_thread() here.

    A blocking sys.stdin.readline() in a worker thread can survive
    cancellation and make asyncio.run() wait for the default executor
    during shutdown. That is what causes Ctrl+C to require a final
    Enter before the process exits.

    On macOS/Unix, add_reader() lets asyncio monitor fd 0 directly.
    """
    loop = asyncio.get_running_loop()

    try:
        fd = sys.stdin.fileno()
    except (OSError, ValueError):
        logger.warning(
            "stdin is not available; stdin reader disabled"
        )
        return

    stopped = loop.create_future()

    def finish() -> None:
        try:
            loop.remove_reader(fd)
        except Exception:
            pass

        if not stopped.done():
            stopped.set_result(None)

    def on_stdin_readable() -> None:
        try:
            chunk = os.read(fd, 4096)

        except BlockingIOError:
            return

        except OSError as exc:
            logger.debug(
                "stdin read failed: {}",
                exc,
            )
            finish()
            return

        if not chunk:
            # EOF.
            finish()
            return

        # Keep partial UTF-8 / line data between reads.
        state = getattr(
            on_stdin_readable,
            "_buffer",
            None,
        )

        if state is None:
            state = bytearray()
            setattr(
                on_stdin_readable,
                "_buffer",
                state,
            )

        state.extend(chunk)

        while True:
            newline = state.find(b"\n")

            if newline < 0:
                break

            raw = bytes(
                state[:newline]
            )

            del state[:newline + 1]

            text = raw.rstrip(
                b"\r"
            ).decode(
                "utf-8",
                errors="replace",
            ).strip()

            if text:
                ingest(
                    text,
                    None,
                )

    try:
        loop.add_reader(
            fd,
            on_stdin_readable,
        )

        await stopped

    finally:
        try:
            loop.remove_reader(fd)
        except Exception:
            pass


async def run_agent_process() -> None:
    # NAN talks to local models on loopback interfaces; ambient
    # shell proxies must never intercept that traffic.
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        os.environ.pop(key, None)

    llm = _llm_from_config(settings)

    providers = ProviderRuntime(
        scan_interval=settings.providers.scan_interval,
        tool_timeout=settings.providers.tool_timeout,
    )

    from .modules.builtin import (
        MemoryModule,
        PlanModule,
    )

    builtin_modules: tuple[type, ...] = (
        MemoryModule,
        PlanModule,
    )

    # Hearing is hardware-dependent; the composition root decides
    # whether it mounts at all.
    if os.getenv(
        "NAN_AUDIO_ENABLED",
        "1",
    ) != "0":
        from .modules.builtin.audio import (
            AudioModule,
        )
        from .modules.builtin.network import (
            NetworkModule,
        )
        from .modules.builtin.system import (
            SystemModule,
        )

        builtin_modules = (
            MemoryModule,
            PlanModule,
            AudioModule,
            SystemModule,
            NetworkModule,
        )

        logger.info(
            "audio module enabled"
        )

    bus = EventBus(
        history_limit=settings.events.history_limit,
        subscriber_queue_size=(
            settings.events.subscriber_queue_size
        ),
    )

    modules = ModuleFacade(
        llm=llm,
        builtin_modules=builtin_modules,
        retry_interval=settings.modules.retry_interval,
        scan_interval=settings.modules.scan_interval,
    )

    skills = SkillRuntime()

    persona_path = (
        Path(__file__).resolve().parents[2]
        / "workspace"
        / "persona.md"
    )

    # --------------------------------------------------------------
    # Runtime tasks / resources.
    # --------------------------------------------------------------

    loop_task: asyncio.Task[None] | None = None
    gateway_task: asyncio.Task[None] | None = None
    stdin_task: asyncio.Task[None] | None = None

    gateway: Gateway | None = None

    running_loop = asyncio.get_running_loop()

    def remove_signal_handlers() -> None:
        for sig in (
            signal.SIGINT,
            signal.SIGTERM,
        ):
            try:
                running_loop.remove_signal_handler(
                    sig
                )
            except Exception:
                pass

    try:
        # ----------------------------------------------------------
        # Start providers.
        # ----------------------------------------------------------

        logger.info(
            "Starting tool providers"
        )

        await providers.start()

        # ----------------------------------------------------------
        # Start module facade.
        # ----------------------------------------------------------

        logger.info(
            "Starting module facade"
        )

        await modules.start()

        skills.discover()

        if not persona_path.is_file():
            logger.error(
                "Persona file not found: {}; "
                "cannot start without it",
                persona_path,
            )

            return

        def read_persona() -> str:
            # Re-read on every access: the agent calls this at each
            # turn start, so persona edits hot-reload.
            return persona_path.read_text(
                encoding="utf-8",
            )

        agent = CoreAgent(
            llm=llm,
            modules=modules,
            providers=providers,
            skills=skills,
            persona_source=read_persona,
            bus=bus,
            max_subagent_depth=(
                settings.agent.max_subagent_depth
            ),
        )

        inbox = Inbox(
            maxsize=settings.runtime.inbox.max_size,
        )

        agent.agent_runtime.set_interrupt_event(
            inbox.wake_event(),
        )

        loop = AgentLoop(
            agent,
            inbox,
            turn_grace=settings.runtime.turn.grace,
            backoff=settings.runtime.retry.backoff,
        )

        stop_received = asyncio.Event()

        loop_task = asyncio.create_task(
            loop.run_forever(),
            name="agent-loop",
        )

        boot_id = uuid.uuid4().hex[:12]

        # 回执去重：客户端断线重连会以同一 mid 补发（可能已投递过），
        # 这里按 mid 保证恰好一次；容量有界，旧 mid 随 LRU 淘汰。
        seen_mids: dict[str, None] = {}

        def ingest(
            text: str,
            mid: str | None = None,
        ) -> None:
            # 唯一的接收点：进入 Inbox 的同时立刻回显，
            # 用户消息不因 sleep/长回合而“消失”。
            if mid:
                if mid in seen_mids:
                    logger.info(
                        "duplicate input dropped (mid={})",
                        mid,
                    )
                    return

                seen_mids[mid] = None

                if len(seen_mids) > 256:
                    for key in list(
                        seen_mids
                    )[:128]:
                        seen_mids.pop(
                            key,
                            None,
                        )

            inbox.put(text)

            if text.strip():
                echo: dict[str, Any] = {
                    "t": "user_input",
                    "id": (
                        f"u{time.time_ns()}"
                    ),
                    "text": text,
                    "boot_id": boot_id,
                }

                if mid:
                    echo["mid"] = mid

                bus.emit(echo)

        # ----------------------------------------------------------
        # Gateway.
        # ----------------------------------------------------------

        gateway = Gateway(
            bus=bus,
            host=settings.gateway.host,
            port=settings.gateway.port,
            on_input=ingest,
            state_provider=None,
            frontend_dir=(
                Path(__file__).resolve().parents[2]
                / "frontend"
                / "app"
                / "dist"
            ),
        )

        gateway_task = asyncio.create_task(
            gateway.serve(),
            name="ws-gateway",
        )

        # ----------------------------------------------------------
        # stdin
        #
        # No executor / background thread.
        # ----------------------------------------------------------

        stdin_task = asyncio.create_task(
            _read_stdin(ingest),
            name="stdin-reader",
        )

        def _request_stop() -> None:
            logger.info(
                "Stop signal received; grace {}s "
                "for the current turn",
                DEFAULT_TURN_GRACE,
            )

            loop.request_stop()
            stop_received.set()

        for sig in (
            signal.SIGINT,
            signal.SIGTERM,
        ):
            running_loop.add_signal_handler(
                sig,
                _request_stop,
            )

        logger.info(
            "NAN is running. "
            "Type a message and press Enter."
        )

        # ----------------------------------------------------------
        # Wait for Ctrl+C / SIGTERM.
        # ----------------------------------------------------------

        await stop_received.wait()

    finally:
        # ==========================================================
        # Shutdown
        # ==========================================================

        remove_signal_handlers()

        # ----------------------------------------------------------
        # Stop stdin reader first.
        #
        # Because stdin is directly attached to the event loop,
        # cancellation is immediate and leaves no executor thread.
        # ----------------------------------------------------------

        if stdin_task is not None:
            stdin_task.cancel()

            try:
                await stdin_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "stdin reader shutdown failed"
                )

        # ----------------------------------------------------------
        # Stop accepting / processing agent work.
        # ----------------------------------------------------------

        if loop_task is not None:
            if not loop_task.done():
                try:
                    await loop_task

                except asyncio.CancelledError:
                    logger.debug(
                        "Agent loop cancelled during shutdown"
                    )

                except Exception:
                    logger.exception(
                        "Agent loop shutdown failed"
                    )

        # ----------------------------------------------------------
        # Gateway.
        # ----------------------------------------------------------

        if gateway is not None:
            try:
                await gateway.close()

            except BaseException:
                logger.exception(
                    "Gateway shutdown failed"
                )

        # The serve() task may still be alive after close().
        if gateway_task is not None:
            if not gateway_task.done():
                gateway_task.cancel()

            try:
                await gateway_task

            except asyncio.CancelledError:
                pass

            except Exception:
                logger.exception(
                    "Gateway task shutdown failed"
                )

        # ----------------------------------------------------------
        # Modules first, then providers.
        #
        # Modules may still reference/use tool providers, so stop
        # their consumers before stopping MCP workers.
        # ----------------------------------------------------------

        logger.info(
            "Stopping modules and tool providers"
        )

        try:
            await modules.stop()

        except BaseException:
            logger.exception(
                "Module shutdown failed; continuing"
            )

        try:
            await providers.stop()

        except BaseException:
            logger.exception(
                "Provider shutdown failed; continuing"
            )

        logger.info(
            "NAN stopped cleanly"
        )


def main() -> None:
    try:
        asyncio.run(
            run_agent_process()
        )

    except KeyboardInterrupt:
        # Signal handlers normally turn Ctrl+C into graceful shutdown,
        # but keep this as a final safety net.
        pass


if __name__ == "__main__":
    main()