"""
Composition root: core process = agents + injection registry +
gateway + execution sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import uuid
from typing import cast

from loguru import logger

from nan.agent import Agent, AgentConfig
from nan.agent.handle import Book
from nan.agent.sandbox import Sandbox
from nan.config import PACKAGE_DIR, Settings
from nan.config.loader import load_settings, resolve_llm
from nan.events import sink
from nan.gateway import Gateway
from nan.injection import InjectionHub
from nan.llm import LLMProvider, ProviderName

_PROXY_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


async def run_core(config_path: str | None = None) -> None:
    cfg = load_settings(config_path)

    # Strip ambient proxies: model traffic targets loopback
    # endpoints.
    for key in _PROXY_KEYS:
        os.environ.pop(key, None)

    llm_cfg = resolve_llm(cfg)

    # The active entry's protocol must be one of the two supported.
    if llm_cfg.provider not in ("openai", "anthropic"):
        raise ValueError(
            f"llm.provider must resolve to 'openai' or 'anthropic', got {llm_cfg.provider!r}"
        )

    llm = LLMProvider(
        provider=cast(ProviderName, llm_cfg.provider),
        # empty config values = SDK defaults (env fallback / default endpoint)
        api_key=cast(str, llm_cfg.api_key or None),
        model=llm_cfg.model,
        base_url=cast(str, llm_cfg.base_url or None),
        timeout=llm_cfg.timeout,
        max_retries=llm_cfg.max_retries,
    )

    book = Book()

    hub = InjectionHub(
        host=cfg.injection.host,
        port=cfg.injection.port,
        request_timeout=cfg.injection.timeout,
        heartbeat_ttl=cfg.injection.heartbeat_ttl,
    )

    cfg.workspace.mkdir(parents=True, exist_ok=True)

    sandbox = _build_sandbox(cfg)

    sandbox.ensure_profile()

    _ensure_persona(cfg)

    root = Agent(
        book=book,
        llm=llm,
        injections=hub,
        workspace=cfg.workspace,
        persona_path=cfg.agent.persona_path,  # re-read every turn
        sandbox=sandbox,
        config=AgentConfig(
            max_depth=cfg.agent.max_depth,
            max_children=cfg.agent.max_children,
            autonomous_interval=cfg.agent.autonomous_interval,
            shell_timeout=cfg.agent.shell_timeout,
            shell_output_limit=cfg.agent.shell_output_limit,
            backoff=tuple(cfg.agent.backoff),
            history_char_limit=cfg.agent.history_char_limit,
        ),
    )

    # ------------------------------------------------------------------
    # Gateway: subscribes to the `sink` singleton on its own.
    # ------------------------------------------------------------------

    BOOT_ID = uuid.uuid4().hex[:12]

    sink.reset(boot_id=BOOT_ID)

    def ingest(text: str, mid: str | None = None) -> None:
        # User input enters the root handle directly: no inbox
        # module, no indirection.
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

    frontend_dir = cfg.gateway.frontend_dir

    gateway = Gateway(
        host=cfg.gateway.host,
        port=cfg.gateway.port,
        on_input=ingest,
        on_pause=on_pause,
        state_provider=lambda: {"boot": BOOT_ID, "model": llm_cfg.model},
        frontend_dir=frontend_dir if (frontend_dir and frontend_dir.exists()) else None,
    )

    gateway_task: asyncio.Task[None] | None = None

    loop = asyncio.get_running_loop()

    try:
        await hub.start()

        if cfg.autostart.enabled:
            _run_autostart(cfg)

        root.start()

        gateway_task = asyncio.create_task(
            gateway.serve(open_browser=False), name="gateway"
        )

        stop = asyncio.Event()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: _shutdown(root, stop))

        logger.info(
            "nan running: gateway http://{}:{}, injection registry :{}, "
            "workspace {}, sandbox {}",
            cfg.gateway.host,
            cfg.gateway.port,
            cfg.injection.port,
            cfg.workspace,
            sandbox.available and "on" or "degraded",
        )

        await stop.wait()

    finally:
        _shutdown(root)

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


def _build_sandbox(cfg: Settings) -> Sandbox:
    return Sandbox(
        workspace=cfg.workspace,
        package_dir=PACKAGE_DIR,
        builtin_dir=cfg.builtin_dir if cfg.builtin_dir.exists() else None,
        enabled=cfg.sandbox.enabled,
        required=cfg.sandbox.required,
    )


def _ensure_persona(cfg: Settings) -> None:
    """Boot-time guard: the persona file must exist (fail fast)."""

    if not cfg.agent.persona_path.is_file():
        raise FileNotFoundError(f"persona file not found: {cfg.agent.persona_path}")


def _run_autostart(cfg: Settings) -> None:
    """
    Launch the user's autostart script (default components) once
    at boot, outside any agent's authority and outside the
    sandbox. Fire-and-forget: its output goes to the log file,
    its failure must not block the core.
    """
    autostart_path = cfg.autostart.path

    if autostart_path is None or not autostart_path.is_file():
        return

    env = {
        **os.environ,
        "NAN_CORE_HOME": str(autostart_path.parent),
        "NAN_WORKSPACE": str(cfg.workspace),
        "NAN_INJECTION_PORT": str(cfg.injection.port),
    }

    async def launch() -> None:
        proc = await asyncio.create_subprocess_shell(
            f"bash {autostart_path}",
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        logger.info("Autostart launched: {} (pid {})", autostart_path, proc.pid)

    asyncio.get_running_loop().create_task(launch(), name="autostart")


def _shutdown(root: Agent, stop_event: asyncio.Event | None = None) -> None:
    logger.info("Stop signal received")

    root.request_death("shutdown")

    if stop_event is not None:
        stop_event.set()


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_core(config_path))


if __name__ == "__main__":
    main()
