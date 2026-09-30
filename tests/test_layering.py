"""
Layering regression locks.

Keeps the internal import graph honest:

    - the agent layer couples to the modules data model only
      (nan_itself.modules.model), never to the modules runtime:
      importing the engine must not load nan_itself.modules.runtime
      (the package surface eagerly exports it, so the agent layer
      imports the pure-contract submodule directly)
    - the gateway owns its protocol helpers (mid dedup, status
      snapshot) without app-level help
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

from nan_itself.events import EventBus
from nan_itself.gateway import Gateway


# ============================================================================
# Agent -> modules import discipline
# ============================================================================


PROBE = """
import sys

import nan_itself.agent.engine  # noqa: F401

loaded = "nan_itself.modules.runtime" in sys.modules

print("modules_runtime_loaded:", loaded)
"""


def test_agent_engine_does_not_load_modules_runtime():
    """
    A fresh interpreter importing the agent engine must not pull
    in the whole modules runtime; the agent layer only needs the
    data model.
    """
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
    )

    assert (
        "modules_runtime_loaded: False"
        in out.stdout
    )


# ============================================================================
# Gateway protocol helpers
# ============================================================================


def _gateway() -> Gateway:
    return Gateway(
        bus=EventBus(),
        host="127.0.0.1",
        port=0,
        on_input=lambda text, mid: None,
    )


def test_gateway_accepts_mid_once():
    gateway = _gateway()

    assert gateway._accept_mid("m1")

    assert not gateway._accept_mid("m1")

    assert gateway._accept_mid("m2")

    assert gateway._accept_mid(None)


def test_gateway_hello_snapshots_latest_status():
    bus = EventBus()

    bus.emit(
        {
            "t": "status",
            "content": {"state": "working"},
        }
    )

    gateway = Gateway(
        bus=bus,
        host="127.0.0.1",
        port=0,
        on_input=lambda text, mid: None,
    )

    payload = gateway._hello_payload()

    assert payload["content"]["status"] == {
        "state": "working",
    }

    assert "seq" in payload


def test_gateway_hello_defaults_to_idle():
    gateway = _gateway()

    payload = gateway._hello_payload()

    assert payload["content"]["status"] == {
        "state": "idle",
    }


def test_gateway_routes_pause_and_resume_to_on_pause():
    calls: list[bool] = []

    gateway = Gateway(
        bus=EventBus(),
        host="127.0.0.1",
        port=0,
        on_input=lambda text, mid: None,
        on_pause=calls.append,
    )

    # The pause/resume branch touches only on_pause: no websocket
    # is needed to drive it.
    gateway._handle_message(None, '{"t":"pause"}')

    gateway._handle_message(None, '{"t":"resume"}')

    assert calls == [True, False]


def test_gateway_contains_on_pause_callback_errors():
    def boom(paused: bool) -> None:
        raise RuntimeError("boom")

    gateway = Gateway(
        bus=EventBus(),
        host="127.0.0.1",
        port=0,
        on_pause=boom,
    )

    # A raising callback must not break the receive loop.
    gateway._handle_message(None, '{"t":"pause"}')

    gateway._handle_message(None, '{"t":"resume"}')


def test_gateway_ignores_pause_without_callback():
    gateway = _gateway()

    # No on_pause wired: the message is simply ignored.
    gateway._handle_message(None, '{"t":"pause"}')

    gateway._handle_message(None, '{"t":"resume"}')
