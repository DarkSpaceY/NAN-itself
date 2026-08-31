from __future__ import annotations

import asyncio
import os
import signal
import uuid
import sys
import time
from pathlib import Path

from loguru import logger

from .agent.core import CoreAgent
from .agent.loop import (
    AgentLoop,
    DEFAULT_TURN_GRACE,
    Inbox,
)
from .modules import (
    Facade as ModuleFacade,
)
from .skills import SkillRuntime
from .tools import ProviderRuntime
from .events import EventBus
from .gateway import Gateway
from .utils.llm import LLMProvider
from .config import get_settings

settings = get_settings()


def _llm_from_config(settings: Settings) -> LLMProvider:
    return LLMProvider(
        provider=settings.llm.provider,
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
    )


async def _read_stdin(inbox: Inbox, ingest) -> None:
    loop = asyncio.get_running_loop()

    while True:
        line = await loop.run_in_executor(
            None,
            sys.stdin.readline,
        )

        if not line:
            # EOF.
            break

        text = line.strip()

        if text:
            ingest(text, None)


async def run_agent_process() -> None:
    # NAN talks to local models on loopback interfaces; ambient
    # shell proxies must never intercept that traffic.
    for key in (
        "ALL_PROXY", "all_proxy",
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
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
    if os.getenv("NAN_AUDIO_ENABLED", "1") != "0":
        from .modules.builtin.audio import (
            AudioModule,
        )

        from .modules.builtin.system import (
            SystemModule,
        )

        from .modules.builtin.network import (
            NetworkModule,
        )

        builtin_modules = (
            MemoryModule,
            PlanModule,
            AudioModule,
            SystemModule,
            NetworkModule,
        )

        logger.info("audio module enabled")

    bus = EventBus(
        history_limit=settings.events.history_limit,
        subscriber_queue_size=settings.events.subscriber_queue_size,
    )

    modules = ModuleFacade(
        llm=llm,
        builtin_modules=builtin_modules,
        retry_interval=settings.modules.retry_interval,
        scan_interval=settings.modules.scan_interval,
    )

    skills = SkillRuntime()

    logger.info("Starting tool providers")

    await providers.start()

    logger.info("Starting module facade")

    await modules.start()

    skills.discover()

    persona_path = (
        Path(__file__).resolve().parents[2]
        / "workspace"
        / "persona.md"
    )

    if not persona_path.is_file():
        logger.error(
            "Persona file not found: {}; "
            "cannot start without it",
            persona_path,
        )

        await providers.stop()
        await modules.stop()

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
        max_subagent_depth=settings.agent.max_subagent_depth,
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

    # 回执去重:客户端断线重连会以同一 mid 补发(可能已投递过),
    # 这里按 mid 保证恰好一次;容量有界,旧 mid 随 LRU 淘汰。
    seen_mids: dict[str, None] = {}

    def ingest(text: str, mid: str | None = None) -> None:
        # 唯一的接收点:进入 Inbox 的同时立刻回显,
        # 用户消息不因 sleep/长回合而"消失"。
        if mid:
            if mid in seen_mids:
                logger.info("duplicate input dropped (mid={})", mid)
                return
            seen_mids[mid] = None
            if len(seen_mids) > 256:
                for key in list(seen_mids)[:128]:
                    seen_mids.pop(key, None)

        inbox.put(text)

        if text.strip():
            echo: dict[str, Any] = {
                "t": "user_input",
                "id": f"u{time.time_ns()}",
                "text": text,
            }
            if mid:
                echo["mid"] = mid
            bus.emit(echo)

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

    stdin_task = asyncio.create_task(
        _read_stdin(inbox, ingest),
        name="stdin-reader",
    )

    running_loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        logger.info(
            "Stop signal received; grace {}s "
            "for the current turn",
            DEFAULT_TURN_GRACE,
        )

        loop.request_stop()

        stop_received.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        running_loop.add_signal_handler(
            sig,
            _request_stop,
        )

    logger.info(
        "NAN is running. Type a message and press Enter."
    )

    await stop_received.wait()

    stdin_task.cancel()

    try:
        await stdin_task
    except asyncio.CancelledError:
        pass

    # The reader thread is blocked inside sys.stdin.readline();
    # asyncio.run() joins the default executor on close and would
    # wait forever. Redirecting fd 0 hands the thread an EOF.
    try:
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, sys.stdin.fileno())
    except OSError:
        pass

    await loop_task

    await gateway.close()

    logger.info("Stopping modules and tool providers")

    for stop_step in (
        modules.stop,
        providers.stop,
    ):
        try:
            await stop_step()
        except BaseException:
            logger.exception(
                "Shutdown step failed; continuing"
            )

    logger.info("NAN stopped cleanly")


def main() -> None:
    try:
        asyncio.run(run_agent_process())

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
