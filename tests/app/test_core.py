"""The composition root: lifecycle guards, gateway callbacks and
shutdown ordering (no real server or model is started)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nan.app import Core, CoreState

pytestmark = pytest.mark.asyncio


class FakeRoot:
    def __init__(self) -> None:
        self.delivered: list[Any] = []
        self.paused = False
        self.task: asyncio.Task[None] | None = None

    def deliver(self, message: Any) -> None:
        self.delivered.append(message)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def suspend(self) -> None:
        pass


class FakeLLM:
    model = "fake"

    async def close(self) -> None:
        pass


async def test_run_refuses_a_second_call() -> None:
    core = Core()

    core._state = CoreState.RUNNING

    with pytest.raises(RuntimeError, match="only be run once"):
        await core.run()


async def test_ingest_drops_input_until_running(no_sink: Any) -> None:
    core = Core()

    root = FakeRoot()

    core._root = root  # type: ignore[assignment]
    core._state = CoreState.INIT

    core._ingest("early")

    assert root.delivered == []

    core._state = CoreState.RUNNING

    core._ingest("hello", mid="m1")

    assert [message.text for message in root.delivered] == ["hello"]
    assert root.delivered[0].kind == "user"

    echoed = next(event for event in no_sink.history() if event["t"] == "user_input")

    assert echoed["content"] == {"text": "hello", "mid": "m1"}


async def test_ingest_without_mid_emits_text_only(no_sink: Any) -> None:
    core = Core()

    root = FakeRoot()

    core._root = root  # type: ignore[assignment]
    core._state = CoreState.RUNNING

    core._ingest("hello")

    echoed = next(event for event in no_sink.history() if event["t"] == "user_input")

    assert echoed["content"] == {"text": "hello"}


async def test_pause_is_ignored_unless_running() -> None:
    core = Core()

    root = FakeRoot()

    core._root = root  # type: ignore[assignment]
    core._state = CoreState.INIT

    core._on_pause(True)

    assert root.paused is False

    core._state = CoreState.RUNNING

    core._on_pause(True)

    assert root.paused is True

    core._on_pause(False)

    assert root.paused is False


async def test_state_provider_reports_boot_model_and_pause() -> None:
    core = Core()

    root = FakeRoot()

    core._root = root  # type: ignore[assignment]
    core._llm = FakeLLM()  # type: ignore[assignment]

    state = core._state_provider()

    assert state == {"boot": core._boot_id, "model": "fake", "paused": False}

    root.paused = True

    assert core._state_provider()["paused"] is True


async def test_shutdown_suspends_then_closes_in_order() -> None:
    calls: list[str] = []

    class Root(FakeRoot):
        def suspend(self) -> None:
            calls.append("suspend")

    class Autostart:
        async def stop(self) -> None:
            calls.append("autostart")

    class Gateway:
        async def close(self) -> None:
            calls.append("gateway")

    class Hub:
        async def stop(self) -> None:
            calls.append("hub")

    class LLM:
        async def close(self) -> None:
            calls.append("llm")

    core = Core()

    core._state = CoreState.RUNNING
    core._root = Root()  # type: ignore[assignment]
    core._autostart = Autostart()  # type: ignore[assignment]
    core._gateway = Gateway()  # type: ignore[assignment]
    core._gateway_task = asyncio.create_task(asyncio.sleep(0))
    core._hub = Hub()  # type: ignore[assignment]
    core._llm = LLM()  # type: ignore[assignment]

    await core._shutdown()

    # Suspend is synchronous and first: it must survive a cancelled
    # coroutine. Everything else is best-effort cleanup.
    assert calls == ["suspend", "autostart", "gateway", "hub", "llm"]
    assert core._state is CoreState.STOPPED

    await core._shutdown()

    assert calls == ["suspend", "autostart", "gateway", "hub", "llm"]


async def test_shutdown_survives_a_failing_component() -> None:
    calls: list[str] = []

    class Root(FakeRoot):
        def suspend(self) -> None:
            raise RuntimeError("boom")

    class Hub:
        async def stop(self) -> None:
            calls.append("hub")

    core = Core()

    core._state = CoreState.RUNNING
    core._root = Root()  # type: ignore[assignment]
    core._hub = Hub()  # type: ignore[assignment]

    await core._shutdown()

    assert calls == ["hub"]
    assert core._state is CoreState.STOPPED
