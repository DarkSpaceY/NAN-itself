from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.nan_itself.tools import (
    LocalToolProvider,
    ProviderRuntime,
    ProviderSpec,
    Provider,
    tool,
)


def make_runtime(
    tmp_path: Path,
    **kwargs,
) -> ProviderRuntime:
    return ProviderRuntime(
        builtin_config_path=(
            tmp_path / "does-not-exist.yaml"
        ),
        workspace_mcp_dir=(
            tmp_path / "workspace" / "tools" / "mcps"
        ),
        workspace_local_dir=(
            tmp_path / "workspace" / "tools" / "local"
        ),
        **kwargs,
    )


class Stall(LocalToolProvider):
    id = "stall"

    @tool(description="Sleep forever.")
    async def stall(self) -> str:
        await asyncio.sleep(60)

        return "never"


@pytest.mark.asyncio
async def test_tool_call_timeout_becomes_error_result(tmp_path):
    runtime = make_runtime(
        tmp_path,
        builtin_tools=(Stall,),
        tool_timeout=0.05,
    )

    runtime._load_builtin_tools()

    result = await runtime.call_tool(
        "stall",
        "stall",
        {},
    )

    assert result.isError is True

    assert "timed out" in (
        result.content[0].text
    )


@pytest.mark.asyncio
async def test_fast_local_tools_are_unaffected(tmp_path):
    class Quick(LocalToolProvider):
        id = "quick"

        @tool(description="Return fast.")
        def ping(self) -> str:
            return "pong"

    runtime = make_runtime(
        tmp_path,
        builtin_tools=(Quick,),
        tool_timeout=0.05,
    )

    runtime._load_builtin_tools()

    result = await runtime.call_tool(
        "quick",
        "ping",
        {},
    )

    assert result.isError is False

    assert result.content[0].text == "pong"
