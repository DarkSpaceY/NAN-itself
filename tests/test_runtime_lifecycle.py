"""
ProviderRuntime lifecycle tests.

start/stop idempotency, book clearing on stop, refresh
semantics per backend kind, and the call_tool error surface.
No real MCP processes are spawned: builtin config points at a
missing temp file and workspace dirs are empty.
"""

from __future__ import annotations

import pytest

from src.nan_itself.tools import (
    LocalToolProvider,
    Provider,
    ProviderSpec,
    ProviderRuntime,
    tool,
)
from src.nan_itself.tools.results import (
    error_result,
)


class NoteTools(LocalToolProvider):
    id = "notes"

    @tool(description="Return hi.")
    def hi(self) -> str:
        return "hi"


def make_runtime(tmp_path) -> ProviderRuntime:
    return ProviderRuntime(
        builtin_config_path=tmp_path / "no-such.yaml",
        workspace_mcp_dir=tmp_path / "mcps",
        workspace_local_dir=tmp_path / "local",
        builtin_tools=(),
    )


def install_local(runtime: ProviderRuntime, name: str = "notes"):
    instance = NoteTools()

    runtime.providers[name] = Provider(
        spec=ProviderSpec(
            name=name,
            kind="local",
            source="<test>",
            origin="workspace",
        ),
        tools=instance.build_tools(),
        instance=instance,
    )


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path):
    runtime = make_runtime(tmp_path)

    await runtime.start()

    supervisor = runtime._supervisor_task

    assert supervisor is not None

    await runtime.start()

    assert runtime._supervisor_task is supervisor

    await runtime.stop()

    assert runtime._supervisor_task is None


@pytest.mark.asyncio
async def test_stop_closes_providers_and_clears_books(tmp_path):
    runtime = make_runtime(tmp_path)

    install_local(runtime)

    runtime._mcp_sources[(tmp_path / "a.yaml").resolve()] = {"x"}
    runtime._local_imported_names[
        (tmp_path / "n.py").resolve()
    ] = "_workspace_tool_n_abc"

    await runtime.start()  # scans empty dirs; keeps installed provider
    await runtime.stop()

    assert runtime.providers == {}
    assert runtime._mcp_sources == {}
    assert runtime._local_sources == {}
    assert runtime._local_imported_names == {}


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path):
    runtime = make_runtime(tmp_path)

    await runtime.start()
    await runtime.stop()
    await runtime.stop()  # must not raise


@pytest.mark.asyncio
async def test_refresh_local_provider_is_a_noop(tmp_path):
    """
    Local tools are fixed at instantiation time: refresh must
    neither raise nor touch anything.
    """
    runtime = make_runtime(tmp_path)

    install_local(runtime)

    before = dict(runtime.providers["notes"].tools)

    await runtime.refresh_provider_tools("notes")

    assert runtime.providers["notes"].tools == before


@pytest.mark.asyncio
async def test_refresh_unknown_provider_raises_key_error(tmp_path):
    runtime = make_runtime(tmp_path)

    with pytest.raises(KeyError):
        await runtime.refresh_provider_tools("ghost")


@pytest.mark.asyncio
async def test_call_tool_unknown_provider_returns_error(tmp_path):
    runtime = make_runtime(tmp_path)

    result = await runtime.call_tool("ghost", "anything", {})

    assert result.isError is True
    assert "Unknown tool provider: ghost" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_passes_arguments_through(tmp_path):
    runtime = make_runtime(tmp_path)

    install_local(runtime)

    result = await runtime.call_tool("notes", "hi", None)

    assert result.isError is False
    assert result.content[0].text == "hi"
