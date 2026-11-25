"""
Config and API tests. No MCP server is started: the contribution
carries the configured categories and the HTTP surface answers
without spawning npx or uvx.
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from main import MCPServer, MCPService, ServerSpec, build_app, load_config


def test_config_parses_the_shipped_file():
    config = load_config()

    assert config.service_id == "mcp"
    assert config.port == 9005

    specs = MCPService(config).specs

    assert any(spec.name == "search" for spec in specs)

    # Placeholders are expanded before a server is launched.
    for spec in specs:
        assert "{" not in spec.command

        for arg in spec.args:
            assert "{" not in arg


def test_contribution_returns_at_context_starts():
    service = MCPService(load_config())

    pointer = service.contribution({"origin": "first"})

    assert "mcp.sh" in pointer
    assert service.contribution({"origin": "context_cleared"}) == pointer
    assert service.contribution({"origin": "normal"}) == ""


def test_contribution_lists_the_configured_categories():
    service = MCPService(load_config())

    pointer = service.contribution({"origin": "first"})

    names = sorted({spec.name for spec in service.specs})

    assert names
    assert f"Categories: {', '.join(names)}." in pointer


def test_contribution_omits_categories_when_nothing_is_configured():
    service = MCPService(load_config())

    service.specs = []

    pointer = service.contribution({"origin": "first"})

    assert "mcp.sh" in pointer
    assert "Categories" not in pointer


def test_http_surface_without_servers():
    service = MCPService(load_config())

    client = TestClient(build_app(service))

    assert client.get("/tools").json() == {"servers": [], "unavailable": {}}
    assert client.post("/turn", json={}).json() == {"text": ""}

    missing = client.post("/call", json={"server": "search", "tool": "x"})

    assert missing.json() == {
        "ok": False,
        "error": {"code": "unknown_server", "message": "unknown server: search"},
    }
    assert client.get("/schema/search/x").json()["error"]["code"] == "unknown_server"


def test_tools_reports_failed_servers():
    service = MCPService(load_config())

    service._failed["playwright"] = "npx not found"

    client = TestClient(build_app(service))

    assert client.get("/tools").json()["unavailable"] == {"playwright": "npx not found"}


class _StubEntry:
    tools = [{"name": "echo", "description": "Echo", "input_schema": {}}]

    async def call(self, tool, arguments):
        return {"ok": True, "result": arguments}


def test_call_validates_the_tool_and_arguments():
    service = MCPService(load_config())

    service._running["stub"] = _StubEntry()  # type: ignore[assignment]

    client = TestClient(build_app(service))

    unknown = client.post("/call", json={"server": "stub", "tool": "nope"}).json()

    assert unknown["error"]["code"] == "unknown_tool"

    invalid = client.post(
        "/call", json={"server": "stub", "tool": "echo", "arguments": []}
    ).json()

    assert invalid["error"]["code"] == "invalid_arguments"

    ok = client.post("/call", json={"server": "stub", "tool": "echo", "arguments": {"x": 1}})

    assert ok.json() == {"ok": True, "result": {"x": 1}}


class TextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResult:
    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.content = [TextContent(text)]

        self.is_error = is_error

        self.data = None


class _FakeClient:
    def __init__(self, result: _FakeResult) -> None:
        self.result = result

    async def call_tool(self, tool, arguments):
        return self.result


def _server_with(result: _FakeResult) -> MCPServer:
    server = MCPServer(ServerSpec({"name": "calc", "command": "true"}, {}), load_config())

    server._client = _FakeClient(result)  # type: ignore[assignment]

    return server


def test_server_call_parses_json_text():
    server = _server_with(_FakeResult('{"value": 4}'))

    assert asyncio.run(server.call("add", {})) == {"ok": True, "result": {"value": 4}}


def test_server_call_keeps_plain_text():
    server = _server_with(_FakeResult("hello"))

    assert asyncio.run(server.call("add", {})) == {"ok": True, "result": "hello"}


def test_server_call_reports_a_tool_error():
    server = _server_with(_FakeResult("bad input", is_error=True))

    result = asyncio.run(server.call("add", {}))

    assert result["ok"] is False
    assert result["error"]["code"] == "tool_error"
