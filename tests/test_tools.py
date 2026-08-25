from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.nan_itself.tools.facade import (
    AgentToolView,
    Provider,
    ProviderSpec,
    ProviderRuntime,
)


class FakeSession:
    def __init__(self, tools=None):
        self.tools = {
            tool.name: tool
            for tool in (tools or [])
        }
        self.calls = []

    async def initialize(self):
        pass

    async def list_tools(self):
        return SimpleNamespace(
            tools=list(self.tools.values())
        )

    async def call_tool(self, name, arguments):
        self.calls.append(
            (name, arguments)
        )

        return {
            "tool": name,
            "arguments": arguments,
        }


class FakeStack:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def fake_tool(name: str):
    from mcp.types import Tool

    return Tool(
        name=name,
        description=f"fake tool {name}",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    )


def install_fake_provider(
    runtime: ProviderRuntime,
    name: str,
    tools: list,
    *,
    origin: str = "workspace",
    source: str = "fake.yaml",
):
    session = FakeSession(tools)
    stack = FakeStack()

    spec = ProviderSpec(
        name=name,
        command="fake",
        source=source,
        origin=origin,
    )

    runtime.providers[name] = Provider(
        spec=spec,
        stack=stack,
        session=session,
        tools={
            tool.name: tool
            for tool in tools
        },
    )

    return session, stack


@pytest.mark.asyncio
async def test_agent_view_has_only_route_without_active_provider():
    runtime = ProviderRuntime()

    install_fake_provider(
        runtime,
        "files",
        [fake_tool("read_file")],
    )

    view = AgentToolView(runtime)

    tools = await view.list_tools()

    assert [tool.name for tool in tools] == [
        "route",
    ]


@pytest.mark.asyncio
async def test_route_exposes_only_active_provider_tools():
    runtime = ProviderRuntime()

    install_fake_provider(
        runtime,
        "files",
        [
            fake_tool("read_file"),
            fake_tool("write_file"),
        ],
    )

    install_fake_provider(
        runtime,
        "playwright",
        [
            fake_tool("browser_open"),
        ],
    )

    view = AgentToolView(runtime)

    result = await view.call_tool(
        "route",
        {
            "provider_name": "files",
        },
    )

    assert not result.isError
    assert view.active_provider == "files"

    tools = await view.list_tools()

    assert {
        tool.name
        for tool in tools
    } == {
        "route",
        "read_file",
        "write_file",
    }

    await view.call_tool(
        "route",
        {
            "provider_name": "playwright",
        },
    )

    tools = await view.list_tools()

    assert {
        tool.name
        for tool in tools
    } == {
        "route",
        "browser_open",
    }


@pytest.mark.asyncio
async def test_agent_views_have_independent_active_provider():
    runtime = ProviderRuntime()

    install_fake_provider(
        runtime,
        "files",
        [fake_tool("read_file")],
    )

    install_fake_provider(
        runtime,
        "playwright",
        [fake_tool("browser_open")],
    )

    main = runtime.create_agent_view()
    subagent = runtime.create_agent_view()

    await main.call_tool(
        "route",
        {"provider_name": "files"},
    )

    await subagent.call_tool(
        "route",
        {"provider_name": "playwright"},
    )

    assert main.active_provider == "files"
    assert subagent.active_provider == "playwright"

    main_tools = await main.list_tools()
    subagent_tools = await subagent.list_tools()

    assert {
        tool.name
        for tool in main_tools
    } == {
        "route",
        "read_file",
    }

    assert {
        tool.name
        for tool in subagent_tools
    } == {
        "route",
        "browser_open",
    }


@pytest.mark.asyncio
async def test_agent_views_share_same_provider_connection():
    runtime = ProviderRuntime()

    session, _ = install_fake_provider(
        runtime,
        "files",
        [fake_tool("read_file")],
    )

    main = runtime.create_agent_view()
    subagent = runtime.create_agent_view()

    await main.call_tool(
        "route",
        {"provider_name": "files"},
    )

    await subagent.call_tool(
        "route",
        {"provider_name": "files"},
    )

    provider = runtime.get_provider("files")

    assert provider is not None
    assert main.active_provider == "files"
    assert subagent.active_provider == "files"

    # Both views ultimately use the same MCP session.
    await main.call_tool(
        "read_file",
        {"path": "a.txt"},
    )

    await subagent.call_tool(
        "read_file",
        {"path": "b.txt"},
    )

    assert session.calls == [
        (
            "read_file",
            {"path": "a.txt"},
        ),
        (
            "read_file",
            {"path": "b.txt"},
        ),
    ]


@pytest.mark.asyncio
async def test_unknown_mcp_cannot_be_routed():
    runtime = ProviderRuntime()

    view = runtime.create_agent_view()

    result = await view.call_tool(
        "route",
        {"provider_name": "missing"},
    )

    assert result.isError
    assert view.active_provider is None


@pytest.mark.asyncio
async def test_tool_call_without_active_provider_is_rejected():
    runtime = ProviderRuntime()

    install_fake_provider(
        runtime,
        "files",
        [fake_tool("read_file")],
    )

    view = runtime.create_agent_view()

    result = await view.call_tool(
        "read_file",
        {},
    )

    assert result.isError
    assert view.active_provider is None


@pytest.mark.asyncio
async def test_tool_call_is_forwarded_to_active_provider():
    runtime = ProviderRuntime()

    session, _ = install_fake_provider(
        runtime,
        "files",
        [fake_tool("read_file")],
    )

    view = runtime.create_agent_view()

    await view.call_tool(
        "route",
        {"provider_name": "files"},
    )

    result = await view.call_tool(
        "read_file",
        {
            "path": "hello.txt",
        },
    )

    assert result == {
        "tool": "read_file",
        "arguments": {
            "path": "hello.txt",
        },
    }

    assert session.calls == [
        (
            "read_file",
            {
                "path": "hello.txt",
            },
        )
    ]


@pytest.mark.asyncio
async def test_builtin_yaml_parser(tmp_path):
    config = tmp_path / "tools.yaml"

    config.write_text(
        """
mcp_servers:
  files:
    command: npx
    args:
      - "--yes"
      - "filesystem"

  calculator:
    command: uvx
    args:
      - "calculator"
    env:
      MODE: test
    cwd: /tmp
""",
        encoding="utf-8",
    )

    raw = yaml.safe_load(
        config.read_text(encoding="utf-8")
    )

    specs = ProviderRuntime._parse_builtin_config(
        raw,
        config,
    )

    assert {
        spec.name
        for spec in specs
    } == {
        "files",
        "calculator",
    }

    calculator = next(
        spec
        for spec in specs
        if spec.name == "calculator"
    )

    assert calculator.command == "uvx"
    assert calculator.args == (
        "calculator",
    )
    assert dict(calculator.env) == {
        "MODE": "test",
    }
    assert calculator.cwd == "/tmp"
    assert calculator.origin == "builtin"


@pytest.mark.asyncio
async def test_workspace_single_mcp_yaml_parser(tmp_path):
    config = tmp_path / "browser.yaml"

    config.write_text(
        """
name: browser

command: npx

args:
  - "--yes"
  - "@playwright/mcp@0.0.79"

cwd: /tmp
""",
        encoding="utf-8",
    )

    raw = yaml.safe_load(
        config.read_text(encoding="utf-8")
    )

    specs = ProviderRuntime._parse_workspace_config(
        raw,
        config,
    )

    assert len(specs) == 1

    spec = specs[0]

    assert spec.name == "browser"
    assert spec.command == "npx"
    assert spec.args == (
        "--yes",
        "@playwright/mcp@0.0.79",
    )
    assert spec.cwd == "/tmp"
    assert spec.origin == "workspace"


@pytest.mark.asyncio
async def test_workspace_multi_mcp_yaml_parser(tmp_path):
    config = tmp_path / "extensions.yaml"

    config.write_text(
        """
mcp_servers:
  browser:
    command: npx
    args:
      - browser

  calculator:
    command: uvx
    args:
      - calculator
""",
        encoding="utf-8",
    )

    raw = yaml.safe_load(
        config.read_text(encoding="utf-8")
    )

    specs = ProviderRuntime._parse_workspace_config(
        raw,
        config,
    )

    assert {
        spec.name
        for spec in specs
    } == {
        "browser",
        "calculator",
    }


@pytest.mark.asyncio
async def test_workspace_source_reload_is_atomic(tmp_path, monkeypatch):
    workspace = (
        tmp_path
        / "workspace"
        / "tools"
        / "mcps"
    )
    workspace.mkdir(parents=True)

    config = workspace / "extensions.yaml"

    config.write_text(
        """
mcp_servers:
  files:
    command: fake
""",
        encoding="utf-8",
    )

    runtime = ProviderRuntime(
        workspace_mcp_dir=workspace,
        workspace_local_dir=(
            tmp_path / "workspace" / "tools" / "local"
        ),
    )

    async def fake_connect(spec):
        if spec.name == "broken":
            raise RuntimeError("connection failed")

        return Provider(
            spec=spec,
            stack=FakeStack(),
            session=FakeSession(
                [fake_tool("ok")]
            ),
            tools={
                "ok": fake_tool("ok"),
            },
        )

    monkeypatch.setattr(
        runtime,
        "_connect_provider",
        fake_connect,
    )

    await runtime._scan_workspace()

    assert set(runtime.providers) == {
        "files",
    }

    old = runtime.providers["files"]

    config.write_text(
        """
mcp_servers:
  files:
    command: fake

  broken:
    command: fake
""",
        encoding="utf-8",
    )

    # Ensure mtime/size definitely changes.
    await asyncio.sleep(0.01)

    await runtime._scan_workspace()

    # Failed candidate transaction leaves the old source intact.
    assert set(runtime.providers) == {
        "files",
    }

    assert runtime.providers["files"] is old


@pytest.mark.asyncio
async def test_workspace_provider_can_be_added_and_removed(
    tmp_path,
    monkeypatch,
):
    workspace = (
        tmp_path
        / "workspace"
        / "tools"
        / "mcps"
    )
    workspace.mkdir(parents=True)

    runtime = ProviderRuntime(
        workspace_mcp_dir=workspace,
    )

    async def fake_connect(spec):
        return Provider(
            spec=spec,
            stack=FakeStack(),
            session=FakeSession(
                [fake_tool("hello")]
            ),
            tools={
                "hello": fake_tool("hello"),
            },
        )

    monkeypatch.setattr(
        runtime,
        "_connect_provider",
        fake_connect,
    )

    config = workspace / "hello.yaml"

    config.write_text(
        """
name: hello

command: fake
""",
        encoding="utf-8",
    )

    await runtime._scan_workspace()

    assert "hello" in runtime.providers

    config.unlink()

    await runtime._scan_workspace()

    assert "hello" not in runtime.providers


@pytest.mark.asyncio
async def test_workspace_cannot_override_builtin(tmp_path):
    workspace = (
        tmp_path
        / "workspace"
        / "tools"
        / "mcps"
    )
    workspace.mkdir(parents=True)

    runtime = ProviderRuntime(
        workspace_mcp_dir=workspace,
    )

    builtin_session, _ = install_fake_provider(
        runtime,
        "files",
        [fake_tool("read_file")],
        origin="builtin",
        source="<builtin>",
    )

    config = workspace / "files.yaml"

    config.write_text(
        """
name: files

command: fake
""",
        encoding="utf-8",
    )

    await runtime._scan_workspace()

    assert runtime.providers["files"].session is (
        builtin_session
    )

    assert config.resolve() not in runtime._workspace_sources


@pytest.mark.asyncio
async def test_workspace_scan_is_not_repeated_for_unchanged_file(
    tmp_path,
    monkeypatch,
):
    workspace = (
        tmp_path
        / "workspace"
        / "tools"
        / "mcps"
    )
    workspace.mkdir(parents=True)

    runtime = ProviderRuntime(
        workspace_mcp_dir=workspace,
    )

    count = 0

    async def fake_connect(spec):
        nonlocal count
        count += 1

        return Provider(
            spec=spec,
            stack=FakeStack(),
            session=FakeSession(
                [fake_tool("hello")]
            ),
            tools={
                "hello": fake_tool("hello"),
            },
        )

    monkeypatch.setattr(
        runtime,
        "_connect_provider",
        fake_connect,
    )

    config = workspace / "hello.yaml"

    config.write_text(
        """
name: hello
command: fake
""",
        encoding="utf-8",
    )

    await runtime._scan_workspace()
    await runtime._scan_workspace()

    assert count == 1


@pytest.mark.asyncio
async def test_runtime_stop_closes_all_providers():
    runtime = ProviderRuntime()

    _, files_stack = install_fake_provider(
        runtime,
        "files",
        [fake_tool("read_file")],
    )

    _, browser_stack = install_fake_provider(
        runtime,
        "browser",
        [fake_tool("open")],
    )

    await runtime.stop()

    assert files_stack.closed is True
    assert browser_stack.closed is True
    assert runtime.providers == {}
    