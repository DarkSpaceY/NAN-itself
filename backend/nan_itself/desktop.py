"""Desktop shell: a pywebview window around the gateway.

Runs the same agent process as the terminal entry, but in a
background thread, and owns a native window on the main
thread. Closing the window stops the agent cleanly.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import webview
from loguru import logger

from .app import run_agent_process
from .config import get_settings


def _wait_for_gateway(
    host: str,
    port: int,
    timeout_s: float = 30.0,
) -> bool:
    """Block until the gateway accepts TCP connections."""
    deadline = time.monotonic() + timeout_s
    target = "127.0.0.1" if host in ("", "0.0.0.0") else host

    while time.monotonic() < deadline:
        try:
            with socket.create_connection(
                (target, port),
                timeout=0.4,
            ):
                return True

        except OSError:
            time.sleep(0.2)

    return False


def main() -> None:
    settings = get_settings()

    stop = asyncio.Event()

    core = threading.Thread(
        target=lambda: asyncio.run(
            run_agent_process(
                stop=stop,
                install_signals=False,
            )
        ),
        name="nan-core",
        daemon=True,
    )
    core.start()

    host = settings.gateway.host
    port = settings.gateway.port

    if not _wait_for_gateway(host, port):
        logger.error(
            "Gateway did not come up on {}:{}; "
            "desktop window not started",
            host,
            port,
        )
        stop.set()
        core.join(timeout=10.0)
        return

    url = (
        f"http://{'127.0.0.1' if host in ('', '0.0.0.0') else host}"
        f":{port}/"
    )

    logger.info("Opening desktop window: {}", url)

    window = webview.create_window(
        "NAN",
        url,
        width=1280,
        height=800,
        min_size=(720, 480),
    )

    webview.start()

    # Window closed by the user -> stop the agent core.
    logger.info("Desktop window closed; stopping NAN")

    stop.set()
    core.join(timeout=15.0)


if __name__ == "__main__":
    main()
