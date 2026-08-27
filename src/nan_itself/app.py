from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from loguru import logger

from src.nan_itself.agent.core import CoreAgent
from src.nan_itself.agent.loop import (
    AgentLoop,
    DEFAULT_TURN_GRACE,
    Inbox,
)
from src.nan_itself.modules import (
    Facade as ModuleFacade,
)
from src.nan_itself.skills import SkillRuntime
from src.nan_itself.tools import ProviderRuntime
from src.nan_itself.utils.llm import LLMProvider


def _llm_from_env() -> LLMProvider:
    """
    Local-first defaults: any OpenAI-compatible server works.
    """
    return LLMProvider(
        provider=os.getenv("NAN_LLM_PROVIDER", "openai"),
        api_key=os.getenv("NAN_LLM_API_KEY", "local"),
        model=os.getenv("NAN_LLM_MODEL", "local-model"),
        base_url=os.getenv(
            "NAN_LLM_BASE_URL",
            "http://127.0.0.1:11434/v1",
        ),
        timeout=float(os.getenv("NAN_LLM_TIMEOUT", "600")),
        max_retries=int(os.getenv("NAN_LLM_MAX_RETRIES", "2")),
    )


async def _read_stdin(inbox: Inbox) -> None:
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
            inbox.put(text)


async def run_agent_process() -> None:
    # NAN talks to local models on loopback interfaces; ambient
    # shell proxies must never intercept that traffic.
    for key in (
        "ALL_PROXY", "all_proxy",
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
    ):
        os.environ.pop(key, None)

    llm = _llm_from_env()

    providers = ProviderRuntime()

    from src.nan_itself.modules.builtin import (
        MemoryModule,
    )

    builtin_modules: tuple[type, ...] = (MemoryModule,)

    # Hearing is hardware-dependent; the composition root decides
    # whether it mounts at all.
    if os.getenv("NAN_AUDIO_ENABLED", "1") != "0":
        from src.nan_itself.modules.builtin.audio import (
            AudioModule,
        )

        builtin_modules = (
            MemoryModule,
            AudioModule,
        )

        logger.info("audio module enabled")

    modules = ModuleFacade(
        llm=llm,
        builtin_modules=builtin_modules,
    )

    skills = SkillRuntime()

    logger.info("Starting tool providers")

    await providers.start()

    logger.info("Starting module facade")

    await modules.start()

    skills.discover()

    persona_path = Path(
        os.getenv(
            "NAN_PERSONA",
            str(
                Path(__file__).resolve().parents[2]
                / "workspace"
                / "persona.md"
            ),
        )
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
    )

    inbox = Inbox()

    loop = AgentLoop(agent, inbox)

    stop_received = asyncio.Event()

    loop_task = asyncio.create_task(
        loop.run_forever(),
        name="agent-loop",
    )

    stdin_task = asyncio.create_task(
        _read_stdin(inbox),
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

    logger.info("Stopping tool providers and modules")

    # Teardown must never flip the exit code: MCP stdio stacks can
    # raise CancelledError/anyio errors while their transports die.
    for stop_step in (providers.stop, modules.stop):
        try:
            await stop_step()
        except (Exception, asyncio.CancelledError):
            # CancelledError is BaseException on 3.12+ and anyio's
            # stdio teardown raises it while transports die.
            logger.exception("Shutdown step failed; continuing")

    logger.info("NAN stopped cleanly")


def main() -> None:
    try:
        asyncio.run(run_agent_process())

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
