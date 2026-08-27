"""
AgentToolView flow tests.

Route is the progressive-disclosure gate: lock every branch of
its contract plus the view's resilience when providers vanish.
"""

from __future__ import annotations

import pytest

import mcp.types as types

from src.nan_itself.tools import (
    AgentToolView,
    LocalToolProvider,
    Provider,
    ProviderRuntime,
    ProviderSpec,
    tool,
)


class EchoTools(LocalToolProvider):
    id = "echo"

    @tool(description="Echo a word.")
    def echo(self, word: str) -> str:
        return word


def install_local(runtime: ProviderRuntime, name: str):
    instance = EchoTools()

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


def tool_names(tools) -> set[str]:
    return {tool.name for tool in tools}


@pytest.mark.asyncio
async def test_fresh_view_exposes_only_route():
    view = AgentToolView(ProviderRuntime())

    assert await view.list_tools() != []
    assert tool_names(await view.list_tools()) == {"route"}


@pytest.mark.asyncio
async def test_call_without_route_is_rejected():
    view = AgentToolView(ProviderRuntime())

    result = await view.call_tool("anything", {})

    assert result.isError is True
    assert "No tool provider is active" in result.content[0].text


@pytest.mark.asyncio
async def test_route_requires_argument():
    view = AgentToolView(ProviderRuntime())

    result = await view._route_provider(None)

    assert result.isError is True
    assert "provider_name is required" in result.content[0].text


@pytest.mark.asyncio
async def test_route_unknown_provider_lists_available():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")

    view = AgentToolView(runtime)

    result = await view._route_provider("nope")

    assert result.isError is True
    assert "Unknown tool provider: nope" in result.content[0].text
    assert "echo" in result.content[0].text


@pytest.mark.asyncio
async def test_route_same_provider_twice_is_idempotent():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")
    view = AgentToolView(runtime)

    first = await view._route_provider("echo")
    again = await view._route_provider("echo")

    assert "activated" in first.content[0].text
    assert "already active" in again.content[0].text


@pytest.mark.asyncio
async def test_activation_reveals_provider_tools():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")
    view = AgentToolView(runtime)

    await view.call_tool("route", {"provider_name": "echo"})

    names = tool_names(await view.list_tools())

    assert names == {"route", "echo"}


@pytest.mark.asyncio
async def test_call_after_activation_reaches_runtime():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")
    view = AgentToolView(runtime)

    await view.call_tool("route", {"provider_name": "echo"})

    result = await view.call_tool("echo", {"word": "ping"})

    assert result.isError is False
    assert result.content[0].text == "ping"


@pytest.mark.asyncio
async def test_unknown_tool_on_active_provider_reports_missing():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")
    view = AgentToolView(runtime)

    await view.call_tool("route", {"provider_name": "echo"})

    result = await view.call_tool("ghost", {})

    assert result.isError is True
    assert (
        "not available from provider 'echo'"
        in result.content[0].text
    )


@pytest.mark.asyncio
async def test_view_resets_when_active_provider_disappears():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")
    view = AgentToolView(runtime)

    await view.call_tool("route", {"provider_name": "echo"})

    # Provider vanishes behind the view's back (hot removal).
    del runtime.providers["echo"]

    names = tool_names(await view.list_tools())

    assert names == {"route"}
    assert view.active_provider is None


@pytest.mark.asyncio
async def test_call_after_provider_disappears_reports_gone():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")
    view = AgentToolView(runtime)

    await view.call_tool("route", {"provider_name": "echo"})

    del runtime.providers["echo"]

    result = await view.call_tool("echo", {"word": "x"})

    assert result.isError is True
    assert "no longer available" in result.content[0].text


def test_available_providers_mirrors_runtime():
    runtime = ProviderRuntime()

    install_local(runtime, "b")
    install_local(runtime, "a")

    view = AgentToolView(runtime)

    assert view.available_providers() == ("a", "b")


def test_route_tool_schema_locks_enum_to_live_providers():
    runtime = ProviderRuntime()

    install_local(runtime, "echo")
    view = AgentToolView(runtime)

    schema = view._route_tool().inputSchema

    assert schema["required"] == ["provider_name"]
    assert schema["properties"]["provider_name"]["enum"] == ["echo"]
