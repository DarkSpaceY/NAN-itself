"""
Config and API tests. No MCP server is started: the catalog is empty,
so the contribution is silent and the HTTP surface answers without
spawning npx or uvx.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import MCPService, build_app, load_config


def test_config_parses_the_shipped_file():
    config = load_config()

    assert config.service_id == "mcp"
    assert config.port == 9005
    assert any(spec.name == "search" for spec in config.servers)

    # Placeholders are expanded before a server is launched.
    for spec in config.servers:
        assert "{" not in spec.command

        for arg in spec.args:
            assert "{" not in arg


def test_contribution_returns_at_context_starts():
    service = MCPService(load_config())

    pointer = service.contribution({"origin": "first"})

    assert "mcp.sh" in pointer
    assert service.contribution({"origin": "context_cleared"}) == pointer
    assert service.contribution({"origin": "normal"}) == ""


def test_http_surface_without_servers():
    service = MCPService(load_config())

    client = TestClient(build_app(service))

    assert client.get("/tools").json() == {"servers": []}
    assert client.post("/turn", json={}).json() == {"text": ""}

    missing = client.post("/call", json={"server": "search", "tool": "x"})

    assert missing.json()["ok"] is False
    assert client.get("/schema/search/x").json()["ok"] is False
