"""
CLI contract tests: compact output, parsed results, structured failures.
The HTTP layer is monkeypatched; the real path is exercised by the
service tests and the end-to-end check.
"""

from __future__ import annotations

import json

import mcp_cli


def test_type_name_handles_nullable_anyof():
    assert mcp_cli._type_name({"anyOf": [{"type": "string"}, {"type": "null"}]}) == "string|null"
    assert mcp_cli._type_name({"type": "integer"}) == "integer"
    assert mcp_cli._type_name({}) == "any"


def test_list_is_compact(monkeypatch, capsys):
    monkeypatch.setattr(
        mcp_cli,
        "_get",
        lambda path: {
            "servers": [
                {
                    "name": "search",
                    "tools": [{"name": "web_search", "description": "Search the web"}],
                }
            ],
            "unavailable": {"playwright": "npx not found"},
        },
    )

    assert mcp_cli.main(["list"]) == 0

    out = capsys.readouterr().out

    assert "search: web_search" in out
    assert "unavailable: playwright (npx not found)" in out


def test_list_full_includes_descriptions(monkeypatch, capsys):
    monkeypatch.setattr(
        mcp_cli,
        "_get",
        lambda path: {
            "servers": [
                {"name": "search", "tools": [{"name": "web_search", "description": "Search"}]}
            ],
            "unavailable": {},
        },
    )

    assert mcp_cli.main(["list", "--full"]) == 0

    assert "web_search — Search" in capsys.readouterr().out


def test_list_reports_servers_still_starting(monkeypatch, capsys):
    # A cold boot: the gateway is up, servers are still being spawned.
    monkeypatch.setattr(
        mcp_cli,
        "_get",
        lambda path: {"servers": [], "starting": ["playwright", "search"], "unavailable": {}},
    )

    assert mcp_cli.main(["list"]) == 0

    out = capsys.readouterr().out

    assert "starting: playwright" in out
    assert "starting: search" in out
    assert "no MCP servers configured" not in out


def test_list_without_configuration_says_so(monkeypatch, capsys):
    monkeypatch.setattr(
        mcp_cli, "_get", lambda path: {"servers": [], "starting": [], "unavailable": {}}
    )

    assert mcp_cli.main(["list"]) == 0

    assert "no MCP servers configured" in capsys.readouterr().out


def test_schema_prints_a_table(monkeypatch, capsys):
    monkeypatch.setattr(
        mcp_cli,
        "_get",
        lambda path: {
            "ok": True,
            "server": "search",
            "tool": "web_search",
            "description": "Search the web",
            "schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    )

    assert mcp_cli.main(["schema", "search", "web_search"]) == 0

    out = capsys.readouterr().out

    assert "search.web_search — Search the web" in out
    assert "query" in out
    assert "string" in out
    assert "yes" in out


def test_call_prints_parsed_json(monkeypatch, capsys):
    monkeypatch.setattr(
        mcp_cli, "_post", lambda path, body: {"ok": True, "result": {"value": 4}}
    )

    assert mcp_cli.main(["call", "calc", "add", '{"a": 2}']) == 0

    assert json.loads(capsys.readouterr().out) == {"value": 4}


def test_call_prints_plain_text(monkeypatch, capsys):
    monkeypatch.setattr(mcp_cli, "_post", lambda path, body: {"ok": True, "result": "hello"})

    assert mcp_cli.main(["call", "calc", "echo", "{}"]) == 0

    assert capsys.readouterr().out.strip() == "hello"


def test_call_maps_unknown_tool_to_exit_3(monkeypatch, capsys):
    monkeypatch.setattr(
        mcp_cli,
        "_post",
        lambda path, body: {
            "ok": False,
            "error": {"code": "unknown_tool", "message": "unknown tool: search.x"},
        },
    )

    assert mcp_cli.main(["call", "search", "x"]) == 3

    error = json.loads(capsys.readouterr().err)

    assert error["error"]["code"] == "unknown_tool"


def test_call_rejects_non_object_arguments(capsys):
    assert mcp_cli.main(["call", "search", "web_search", "[]"]) == 2

    error = json.loads(capsys.readouterr().err)

    assert error["error"]["code"] == "invalid_arguments"


def test_gateway_failure_is_reported(monkeypatch, capsys):
    def fail(path):
        raise mcp_cli.CliError("service_unavailable", "cannot reach the mcp gateway", 3)

    monkeypatch.setattr(mcp_cli, "_get", fail)

    assert mcp_cli.main(["list"]) == 3

    error = json.loads(capsys.readouterr().err)

    assert error["error"]["code"] == "service_unavailable"
