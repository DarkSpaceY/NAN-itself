"""
Provider unit tests.

Covers the kind-dispatch and every local-call branch:
success, missing method, validation error, handler crash.
"""

from __future__ import annotations

import pytest

import mcp.types as types
from pydantic import ValidationError

from src.nan_itself.tools import (
    LocalToolProvider,
    Provider,
    ProviderSpec,
    tool,
)


class MathTools(LocalToolProvider):
    id = "math"

    @tool(description="Add two integers.")
    def add(self, a: int, b: int) -> int:
        return a + b

    @tool(description="Always crash.")
    def boom(self) -> str:
        raise RuntimeError("kaboom")


def make_local_provider() -> Provider:
    instance = MathTools()

    return Provider(
        spec=ProviderSpec(
            name="math",
            kind="local",
            source="<test>",
            origin="builtin",
        ),
        tools=instance.build_tools(),
        instance=instance,
    )


class RecordingSession:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))

        return self.reply


def make_mcp_provider(reply=None):
    session = RecordingSession(reply)

    provider = Provider(
        spec=ProviderSpec(name="srv"),
        tools={"t": types.Tool(
            name="t",
            description="",
            inputSchema={},
        )},
        stack=None,
        session=session,
    )

    return provider, session


@pytest.mark.asyncio
async def test_local_success_serializes_return_value():
    provider = make_local_provider()

    result = await provider.call_tool(
        "add",
        {"a": 2, "b": 40},
    )

    assert result.isError is False
    assert result.content[0].text == "42"


@pytest.mark.asyncio
async def test_local_missing_method_reports_not_exposed():
    provider = make_local_provider()

    result = await provider.call_tool("nope", {})

    assert result.isError is True
    assert "not exposed" in result.content[0].text
    assert "'math'" in result.content[0].text


@pytest.mark.asyncio
async def test_local_invalid_arguments_report_validation_error():
    provider = make_local_provider()

    result = await provider.call_tool(
        "add",
        {"a": "not-an-int", "b": 1},
    )

    assert result.isError is True
    assert "Invalid arguments" in result.content[0].text


@pytest.mark.asyncio
async def test_local_handler_crash_reports_exception_type():
    provider = make_local_provider()

    result = await provider.call_tool("boom", {})

    assert result.isError is True
    assert result.content[0].text == "RuntimeError: kaboom"


@pytest.mark.asyncio
async def test_missing_instance_reports_not_exposed():
    provider = make_local_provider()

    provider.instance = None

    result = await provider.call_tool("add", {"a": 1, "b": 2})

    assert result.isError is True
    assert "not exposed" in result.content[0].text


@pytest.mark.asyncio
async def test_mcp_kind_dispatches_to_session():
    reply = types.CallToolResult(content=[])

    provider, session = make_mcp_provider(reply)

    returned = await provider.call_tool(
        "t",
        None,  # None arguments must become {}
    )

    assert returned is reply
    assert session.calls == [("t", {})]


def test_kind_mirrors_spec():
    assert make_local_provider().kind == "local"
    assert make_mcp_provider()[0].kind == "mcp"


def test_validation_error_type_is_pydantic():
    # Guard the import used inside Provider._call_local: if pydantic
    # ever changes its exception surface, this fails loudly here.
    assert issubclass(ValidationError, Exception)
