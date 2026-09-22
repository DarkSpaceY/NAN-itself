from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from typing import Any

from loguru import logger

from .agent import CoreAgent
from .config import get_settings
from .events import EventBus
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

    bus = EventBus(
        history_limit=settings.events.history_limit,
        subscriber_queue_size=(
            settings.events.subscriber_queue_size
        ),
    )

    modules = ModuleFacade(
        llm=llm,
        retry_interval=settings.modules.retry_interval,
        scan_interval=settings.modules.scan_interval,
    )

    skills = SkillRuntime(
        resource_char_limit=(
            settings.skills.resource_char_limit
        ),
        script_timeout=settings.skills.script_timeout,
    )

    persona_path = (
        _paths.repo_root()
        / "workspace"
        / "persona.md"
    )

    # --------------------------------------------------------------
    # Runtime tasks / resources.
    # --------------------------------------------------------------

    gateway_task: asyncio.Task[None] | None = None

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
            tools=providers,
            skills=skills,
            persona_source=read_persona,
            bus=bus,
            max_subagent_depth=(
                settings.agent.max_subagent_depth
            ),
            history_char_limit=(
                settings.agent.history_char_limit
            ),
            turn_grace=(
                settings.runtime.turn.grace
            ),
            backoff=(
                settings.runtime.retry.backoff
            ),
        )

        agent_task = asyncio.create_task(
            agent.run_forever(),
            name="agent-loop",
        )

        boot_id = uuid.uuid4().hex[:12]

        def ingest(
            text: str,
            mid: str | None = None,
        ) -> None:
            # 唯一的接收点：进入 Inbox 模块的同时立刻回显，
            # 用户消息不因 sleep/长回合而“消失”。
            # （mid 去重由 gateway 负责。）
            inbox = modules.get("inbox")

            if inbox is None:
                logger.warning(
                    "Inbox module not running; input dropped"
                )

            else:
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
        )

        gateway_task = asyncio.create_task(
            gateway.serve(),
            name="ws-gateway",
        )

        # ----------------------------------------------------------
        # Wait for Ctrl+C / SIGTERM.
        # ----------------------------------------------------------

        stop_received = asyncio.Event()

        def _request_stop() -> None:
            logger.info(
                "Stop signal received; grace {}s "
                "for the current turn",
                agent.turn_grace,
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
        # Wait for Ctrl+C / SIGTERM.
        # ----------------------------------------------------------

        await stop_received.wait()

    finally:
        # ==========================================================
        # Shutdown
        # ==========================================================

        remove_signal_handlers()

        # ----------------------------------------------------------
        # Stop accepting / processing agent work.
        # ----------------------------------------------------------

        if agent_task is not None:
            if not agent_task.done():
                try:
                    await agent_task

                except asyncio.CancelledError:
                    logger.debug(
                        "Agent loop cancelled during shutdown"
                    )

                except Exception:
                    logger.exception(
                        "Agent loop shutdown failed"
                    )

        # ----------------------------------------------------------
        # Stop all Subagents.
        #
        # A Subagent may outlive its parent Agent turn during normal
        # operation, so it must have an explicit application-level
        # shutdown boundary.
        #
        # This MUST happen before Modules and Providers are stopped,
        # because a running Subagent may still be using them.
        # ----------------------------------------------------------

        try:
            await agent.agent_runtime.shutdown()

        except BaseException:
            logger.exception(
                "Subagent runtime shutdown failed; continuing"
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