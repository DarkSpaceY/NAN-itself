from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from nan_itself.modules.model import (
    ChannelSpec,
    MessageReceipt,
    Module,
    ModuleState,
)
from nan_itself.modules.reload import hot_reload
from nan_itself.modules.runtime import (
    Facade,
)
from nan_itself.tools.local import (
    LocalToolProvider,
    validate_class as validate_local_class,
)
from nan_itself.tools.runtime import (
    ProviderRuntime,
)


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Helpers
# ============================================================================


class DemoModule(Module):
    id = "demo"

    channels = {
        "do": ChannelSpec(
            description="Do something",
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                },
            },
        ),
    }

    def __init__(self):
        self.received = []

    async def on_message(
        self,
        channel: str,
        message: dict,
    ) -> MessageReceipt:
        self.received.append(
            (channel, message)
        )

        return MessageReceipt(
            accepted=True,
            task_id="t-1",
            channel_state="queued",
        )

    async def start(self):
        await asyncio.Event().wait()


class PlainModule(Module):
    """Pre-action-face module: no channels declared."""

    id = "plain"

    async def start(self):
        await asyncio.Event().wait()


class HangingModule(Module):
    id = "hanging"

    channels = {
        "stuck": ChannelSpec(
            description="Never acks",
            input_schema={"type": "object"},
        ),
    }

    async def on_message(
        self,
        channel: str,
        message: dict,
    ) -> MessageReceipt:
        await asyncio.Event().wait()

    async def start(self):
        await asyncio.Event().wait()


def _make_facade(
    tmp_path: Path,
    **kwargs,
) -> Facade:
    (tmp_path / "modules").mkdir(
        parents=True, exist_ok=True
    )

    return Facade(
        workspace_modules=tmp_path
        / "modules",
        data_dir=tmp_path / "data",
        **kwargs,
    )


async def _run_to_running(
    facade: Facade,
    cls: type[Module],
):
    facade._register_module_class(
        cls,
        source=str(
            facade.workspace_modules
            / "source.py"
        ),
    )

    record = facade.modules[cls.id]

    started = asyncio.Event()

    task = asyncio.create_task(
        facade._run_module(record, started)
    )

    await started.wait()

    return record, task


def _receipt_text(result) -> str:
    assert result.isError is not True

    return result.content[0].text


# ============================================================================
# Tool view
# ============================================================================


def test_channels_exposed_as_module_provider(
    tmp_path,
):
    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime()

        facade.tool_runtime = runtime

        record, task = None, None

        async def inner():
            nonlocal record, task
            record, task = (
                await _run_to_running(
                    facade, DemoModule
                )
            )

        async def driver():
            inner_task = asyncio.create_task(
                inner()
            )

            await asyncio.sleep(0.05)

            pairs = (
                runtime.list_all_tools()
            )

            names = {
                f"{provider}/{tool.name}"
                for provider, tool in pairs
            }

            assert (
                "module:demo/do" in names
            )

            provider = (
                runtime.get_provider(
                    "module:demo"
                )
            )

            assert (
                provider is not None
            )

            assert (
                provider.spec.kind
                == "local"
            )

            inner_task.cancel()

            with pytest.raises(
                asyncio.CancelledError
            ):
                await inner_task

        asyncio.run(driver())

        task.cancel()


def test_plain_module_not_exposed(tmp_path):
    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime()

        facade.tool_runtime = runtime

        async def driver():
            inner_task = asyncio.create_task(
                _run_to_running(
                    facade, PlainModule
                )
            )

            await asyncio.sleep(0.05)

            assert (
                runtime.get_provider(
                    "module:plain"
                )
                is None
            )

            inner_task.cancel()

            with pytest.raises(
                asyncio.CancelledError
            ):
                await inner_task

        asyncio.run(driver())


# ============================================================================
# Invoke / ack
# ============================================================================


def test_invoke_returns_ack_receipt(tmp_path):
    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime(
            tool_timeout=5.0
        )

        facade.tool_runtime = runtime

        async def driver():
            inner_task = asyncio.create_task(
                _run_to_running(
                    facade, DemoModule
                )
            )

            await asyncio.sleep(0.05)

            resolved = (
                await runtime.resolve_tool(
                    "module:demo/do"
                )
            )

            assert resolved is not None

            provider, tool = resolved

            assert tool.name == "do"

            assert tool.inputSchema[
                "type"
            ] == "object"

            result = (
                await runtime.call_tool(
                    provider.spec.name,
                    tool.name,
                    {"x": 7},
                )
            )

            payload = json.loads(
                _receipt_text(result)
            )

            assert payload["accepted"] is True

            assert (
                payload["task_id"] == "t-1"
            )

            instance = (
                facade.modules["demo"].instance
            )

            assert instance.received == [
                ("do", {"x": 7})
            ]

            inner_task.cancel()

            with pytest.raises(
                asyncio.CancelledError
            ):
                await inner_task

        asyncio.run(driver())


def test_cancel_kind_message_reaches_channel(
    tmp_path,
):
    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime(
            tool_timeout=5.0
        )

        facade.tool_runtime = runtime

        async def driver():
            inner_task = asyncio.create_task(
                _run_to_running(
                    facade, DemoModule
                )
            )

            await asyncio.sleep(0.05)

            # Cancel is just a message the Module owns;
            # the runtime transports it as-is.
            result = await runtime.call_tool(
                "module:demo",
                "do",
                {"action": "cancel"},
            )

            payload = json.loads(
                _receipt_text(result)
            )

            assert (
                payload["accepted"] is True
            )

            instance = (
                facade.modules["demo"].instance
            )

            assert instance.received == [
                ("do", {"action": "cancel"})
            ]

            inner_task.cancel()

            with pytest.raises(
                asyncio.CancelledError
            ):
                await inner_task

        asyncio.run(driver())


# ============================================================================
# Errors
# ============================================================================


def test_unknown_module_provider_errors(
    tmp_path,
):
    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime(
            tool_timeout=5.0
        )

        facade.tool_runtime = runtime

        async def driver():
            result = await runtime.call_tool(
                "module:ghost",
                "do",
                {},
            )

            assert result.isError is True

            inner_task = asyncio.create_task(
                _run_to_running(
                    facade, DemoModule
                )
            )

            await asyncio.sleep(0.05)

            result = await runtime.call_tool(
                "module:demo",
                "missing",
                {},
            )

            assert result.isError is True

            inner_task.cancel()

            with pytest.raises(
                asyncio.CancelledError
            ):
                await inner_task

        asyncio.run(driver())


def test_hanging_on_message_becomes_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "nan_itself.modules.runtime."
        "_MESSAGE_ACK_TIMEOUT",
        0.05,
    )

    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime(
            tool_timeout=5.0
        )

        facade.tool_runtime = runtime

        async def driver():
            inner_task = asyncio.create_task(
                _run_to_running(
                    facade, HangingModule
                )
            )

            await asyncio.sleep(0.05)

            result = await runtime.call_tool(
                "module:hanging",
                "stuck",
                {},
            )

            assert result.isError is True

            inner_task.cancel()

            with pytest.raises(
                asyncio.CancelledError
            ):
                await inner_task

        asyncio.run(driver())


# ============================================================================
# Lifecycle
# ============================================================================


def test_face_detached_on_terminal_state(
    tmp_path,
):
    class ExitingModule(Module):
        id = "exiting"

        channels = {
            "do": ChannelSpec(
                description="x",
                input_schema={
                    "type": "object"
                },
            ),
        }

        async def start(self):
            return  # exit immediately -> DOWN

    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime()

        facade.tool_runtime = runtime

        async def driver():
            record, task = (
                await _run_to_running(
                    facade, ExitingModule
                )
            )

            await task

            assert record.state is (
                ModuleState.DOWN
            )

            assert (
                runtime.get_provider(
                    "module:exiting"
                )
                is None
            )

        asyncio.run(driver())


def test_hot_reload_swaps_action_face(
    tmp_path,
):
    class ReloadedModule(Module):
        id = "demo"

        channels = {
            "fresh": ChannelSpec(
                description="New channel",
                input_schema={
                    "type": "object"
                },
            ),
        }

        async def start(self):
            await asyncio.Event().wait()

    def scenario():
        facade = _make_facade(tmp_path)

        runtime = ProviderRuntime()

        facade.tool_runtime = runtime

        async def driver():
            inner_task = asyncio.create_task(
                _run_to_running(
                    facade, DemoModule
                )
            )

            await asyncio.sleep(0.05)

            old = facade.modules["demo"]

            await hot_reload(
                facade,
                old=old,
                cls=ReloadedModule,
                imported_name="reloaded",
                fingerprint=(2, 2),
            )

            await asyncio.sleep(0.05)

            tools = {
                tool.name
                for provider, tool in (
                    runtime.list_all_tools()
                )
                if provider == "module:demo"
            }

            assert tools == {"fresh"}

            inner_task.cancel()

            with pytest.raises(
                asyncio.CancelledError
            ):
                await inner_task

        asyncio.run(driver())


# ============================================================================
# Name reservation
# ============================================================================


def test_local_provider_cannot_shadow_module_prefix():
    class Impostor(LocalToolProvider):
        id = "module:evil"

    with pytest.raises(
        ValueError, match="module:"
    ):
        validate_local_class(Impostor)
