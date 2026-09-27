from __future__ import annotations

import asyncio
import os
import signal
import uuid

from loguru import logger

from .agent import Agent, StepEngine
from .config import get_settings
from .events import EventBus, sink
from .gateway import Gateway
from .modules import (
    Facade as ModuleFacade,
)
from .skills import SkillRuntime
from .tools import ProviderRuntime
from .utils import paths as _paths
from .utils.llm import LLMProvider

settings = get_settings()


async def run_agent_process() -> None:
    # --------------------------------------------------------------
    # Environment: NAN talks to local models on loopback
    # interfaces; ambient shell proxies must never intercept that
    # traffic.
    # --------------------------------------------------------------

    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        os.environ.pop(key, None)

    # --------------------------------------------------------------
    # Model and capabilities.
    # --------------------------------------------------------------

    llm = LLMProvider(
        provider=settings.llm.provider,
        api_key=settings.llm.api_key,
        model=settings.llm.model,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
        max_retries=settings.llm.max_retries,
    )

    providers = ProviderRuntime(
        scan_interval=settings.providers.scan_interval,
        tool_timeout=settings.providers.tool_timeout,
    )

    modules = ModuleFacade(
        llm=llm,
        retry_interval=settings.modules.retry_interval,
        scan_interval=settings.modules.scan_interval,
    )

    skills = SkillRuntime(
        resource_char_limit=settings.skills.resource_char_limit,
        script_timeout=settings.skills.script_timeout,
    )

    # --------------------------------------------------------------
    # Agent: the trunk.
    # --------------------------------------------------------------

    engine = StepEngine(
        llm=llm,
    )

    agent = Agent(
        engine=engine,
        modules=modules,
        tools=providers,
        skills=skills,
        max_subagent_depth=settings.agent.max_subagent_depth,
        history_char_limit=settings.agent.history_char_limit,
        backoff=settings.runtime.retry.backoff,
    )

    # --------------------------------------------------------------
    # UI channel: agent events reach the browser through the global
    # sink -> bus -> gateway.
    # --------------------------------------------------------------

    bus = EventBus(
        history_limit=settings.events.history_limit,
        subscriber_queue_size=settings.events.subscriber_queue_size,
    )

    boot_id = uuid.uuid4().hex[:12]

    sink.attach(bus, boot_id=boot_id)

    # --------------------------------------------------------------
    # Gateway, with its two app-side callbacks.
    # --------------------------------------------------------------

    def ingest(
        text: str,
        mid: str | None = None,
    ) -> None:
        # The one receiving point: the text enters the Inbox
        # module and is echoed to the UI at once, so a user
        # message never seems to vanish during a long turn.
        # (mid dedup is the gateway's business.)
        inbox = modules.get("inbox")

        if inbox is None:
            logger.warning(
                "Inbox module not running; input dropped"
            )

        else:
            inbox.put(text)

        if text.strip():
            sink.emit(
                "user_input",
                content=(
                    {"text": text}
                    | ({"mid": mid} if mid else {})
                ),
            )

    def gateway_state() -> dict:
        # status 快照由 gateway 负责（从 bus 历史提取），
        # 这里只提供 app 才知道的身份信息。
        return {
            "boot": boot_id,
            "model": settings.llm.model,
            "base_url": settings.llm.base_url,
        }

    gateway = Gateway(
        bus=bus,
        host=settings.gateway.host,
        port=settings.gateway.port,
        on_input=ingest,
        state_provider=gateway_state,
        frontend_dir=(
            _paths.repo_root()
            / "frontend"
            / "app"
            / "dist"
        ),
        dedup_cache_size=settings.events.input_dedup_cache_size,
    )

    # --------------------------------------------------------------
    # Shutdown handles: pre-initialized so the finally block can
    # reference them even when startup fails early.
    # --------------------------------------------------------------

    agent_task: asyncio.Task[None] | None = None
    gateway_task: asyncio.Task[None] | None = None

    running_loop = asyncio.get_running_loop()

    try:
        # ----------------------------------------------------------
        # Start: capabilities first, then the two long-lived tasks.
        # ----------------------------------------------------------

        logger.info(
            "Starting tool providers"
        )

        await providers.start()

        logger.info(
            "Starting module facade"
        )

        await modules.start()

        agent_task = asyncio.create_task(
            agent.loop(),
            name="agent-loop",
        )

        gateway_task = asyncio.create_task(
            gateway.serve(),
            name="ws-gateway",
        )

        # ----------------------------------------------------------
        # Stop signal.
        # ----------------------------------------------------------

        stop_received = asyncio.Event()

        def _request_stop() -> None:
            logger.info(
                "Stop signal received; stopping agent"
            )

            agent.request_stop()
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
            "NAN is running. Open the gateway frontend to talk."
        )

        # ----------------------------------------------------------
        # Run until stopped.
        # ----------------------------------------------------------

        await stop_received.wait()

    finally:
        # ----------------------------------------------------------
        # Shutdown.
        #
        # Best-effort cleanup that must run to completion: the
        # steps below catch BaseException on purpose, so a
        # cancellation arriving during shutdown cannot leave
        # modules or providers running.
        # ----------------------------------------------------------

        for sig in (
            signal.SIGINT,
            signal.SIGTERM,
        ):
            try:
                running_loop.remove_signal_handler(
                    sig
                )
            except Exception:
                logger.debug(
                    "Signal handler for {} not removed",
                    sig,
                )

        # ----------------------------------------------------------
        # Stop accepting / processing agent work.
        # ----------------------------------------------------------

        if agent_task is not None:
            # Awaiting unconditionally retrieves an earlier
            # failure; cancellation is swallowed on purpose so
            # the remaining cleanup still runs (see above).
            try:
                await agent_task

            except asyncio.CancelledError:
                if agent_task.cancelled():
                    logger.debug(
                        "Agent loop cancelled during shutdown"
                    )

                else:
                    logger.debug(
                        "Agent loop wait cancelled during "
                        "shutdown"
                    )

            except Exception:
                logger.exception(
                    "Agent loop shutdown failed"
                )

        # ----------------------------------------------------------
        # Gateway.
        # ----------------------------------------------------------

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