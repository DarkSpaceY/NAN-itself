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
from src.nan_itself.modules.facade import (
    Facade as ModuleFacade,
)
from src.nan_itself.skills.facade import SkillRuntime
from src.nan_itself.tools.facade import ProviderRuntime
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

    providers = ProviderRuntime()

    modules = ModuleFacade()

    skills = SkillRuntime()

    logger.info("Starting tool providers")

    await providers.start()

    logger.info("Starting module facade")

    await modules.start()

    skills.discover()

    if "core" not in skills.names():
        logger.error(
            "No 'core' skill found in {}; "
            "cannot start without it",
            skills.workspace_skills,
        )

        await providers.stop()
        await modules.stop()

        return

    core_skill = skills.activate("core")

    agent = CoreAgent(
        llm=_llm_from_env(),
        modules=modules,
        providers=providers,
        skills=skills,
        core_skill=core_skill,
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

    await loop_task

    logger.info("Stopping tool providers and modules")

    await providers.stop()

    await modules.stop()

    logger.info("NAN stopped cleanly")


def main() -> None:
    try:
        asyncio.run(run_agent_process())

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
