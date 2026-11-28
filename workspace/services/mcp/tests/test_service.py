"""
Config and API tests. No MCP server is started: the contribution
carries the configured categories and the HTTP surface answers
without spawning npx or uvx.
"""

from __future__ import annotations

import asyncio
import base64

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

    data = client.get("/tools").json()

    # Nothing is up yet: every configured server is still queued, and
    # none of them is inside a start attempt.
    assert data["servers"] == []
    assert data["unavailable"] == {}
    assert data["starting"] == []
    assert data["pending"] == sorted(spec.name for spec in service.specs)

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

    data = client.get("/tools").json()

    assert data["unavailable"] == {"playwright": "npx not found"}

    # A failed server is unavailable, not still starting or queued.
    assert "playwright" not in data["starting"]
    assert "playwright" not in data["pending"]


def test_tools_separates_starting_from_queued_servers():
    service = MCPService(load_config())

    service._starting.add("search")

    client = TestClient(build_app(service))

    data = client.get("/tools").json()

    # Only the server inside a start attempt counts as starting; the
    # rest are queued until the background starter reaches them.
    assert data["starting"] == ["search"]
    assert "search" not in data["pending"]
    assert data["pending"] == sorted(spec.name for spec in service.specs if spec.name != "search")


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

    invalid = client.post("/call", json={"server": "stub", "tool": "echo", "arguments": []}).json()

    assert invalid["error"]["code"] == "invalid_arguments"

    ok = client.post("/call", json={"server": "stub", "tool": "echo", "arguments": {"x": 1}})

    assert ok.json() == {"ok": True, "result": {"x": 1}}


class TextContent:
    def __init__(self, text: str) -> None:
        self.type = "text"

        self.text = text


class ImageContent:
    def __init__(self, data: str, mime_type: str = "image/png") -> None:
        self.type = "image"

        self.data = data

        self.mimeType = mime_type


class AudioContent:
    def __init__(self, data: str, mime_type: str = "audio/mpeg") -> None:
        self.type = "audio"

        self.data = data

        self.mimeType = mime_type


class _Resource:
    def __init__(self, *, text: str | None, blob: str | None, mime_type: str | None) -> None:
        self.text = text

        self.blob = blob

        self.mimeType = mime_type


class EmbeddedResource:
    def __init__(
        self, *, text: str | None = None, blob: str | None = None, mime_type: str | None = None
    ) -> None:
        self.type = "resource"

        self.resource = _Resource(text=text, blob=blob, mime_type=mime_type)


class ResourceLink:
    """A reference the client can fetch; it carries no payload."""

    def __init__(
        self, uri: str, name: str, *, mime_type: str | None = None, size: int | None = None
    ) -> None:
        self.type = "resource_link"

        self.uri = uri

        self.name = name

        self.mimeType = mime_type

        self.size = size


class UnknownContent:
    """A kind the spec's closed union does not define."""

    def __init__(self) -> None:
        self.type = "hologram"


class _FakeResult:
    def __init__(self, text: str, *, is_error: bool = False) -> None:
        self.content = [TextContent(text)]

        self.is_error = is_error

        self.data = None


class _BlocksResult:
    def __init__(self, blocks: list, *, is_error: bool = False) -> None:
        self.content = blocks

        self.is_error = is_error

        self.data = None


class _FakeClient:
    def __init__(self, result: _FakeResult | _BlocksResult) -> None:
        self.result = result

    async def call_tool(self, tool, arguments):
        return self.result


def _server_with(
    result: _FakeResult | _BlocksResult,
    *,
    artifact_root=None,
    workspace=None,
) -> MCPServer:
    server = MCPServer(
        ServerSpec({"name": "calc", "command": "true"}, {}),
        load_config(),
        artifact_root=artifact_root,
        workspace=workspace,
    )

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


def test_non_text_block_is_saved_and_reported_as_a_path(tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"payload"

    server = _server_with(
        _BlocksResult([ImageContent(base64.b64encode(png).decode())]),
        artifact_root=tmp_path / "data",
        workspace=tmp_path,
    )

    result = asyncio.run(server.call("screenshot", {}))

    assert result["ok"] is True
    assert "result" not in result

    (record,) = result["files"]

    assert record["mime"] == "image/png"
    assert record["bytes"] == len(png)
    assert record["path"].startswith("data/calc/")
    assert record["path"].endswith(".png")
    assert (tmp_path / record["path"]).read_bytes() == png


def test_text_and_artifact_arrive_together(tmp_path):
    jpeg = b"\xff\xd8\xff\xe0" + b"payload"

    server = _server_with(
        _BlocksResult(
            [
                TextContent("took one screenshot"),
                ImageContent(base64.b64encode(jpeg).decode(), "image/jpeg"),
            ]
        ),
        artifact_root=tmp_path / "data",
        workspace=tmp_path,
    )

    result = asyncio.run(server.call("screenshot", {}))

    assert result["result"] == "took one screenshot"
    assert result["files"][0]["mime"] == "image/jpeg"
    assert result["files"][0]["path"].endswith(".jpg")


def test_embedded_resource_blob_is_saved_with_its_extension(tmp_path):
    pdf = b"%PDF-1.4 binary"

    server = _server_with(
        _BlocksResult(
            [EmbeddedResource(blob=base64.b64encode(pdf).decode(), mime_type="application/pdf")]
        ),
        artifact_root=tmp_path / "data",
        workspace=tmp_path,
    )

    result = asyncio.run(server.call("render", {}))

    record = result["files"][0]

    assert record["path"].endswith(".pdf")
    assert (tmp_path / record["path"]).read_bytes() == pdf


def test_embedded_text_resource_is_text(tmp_path):
    server = _server_with(
        _BlocksResult([EmbeddedResource(text="hello", mime_type="text/plain")]),
        artifact_root=tmp_path / "data",
        workspace=tmp_path,
    )

    assert asyncio.run(server.call("read", {})) == {"ok": True, "result": "hello"}


def test_audio_is_saved_like_an_image(tmp_path):
    audio = b"ID3payload"

    server = _server_with(
        _BlocksResult([AudioContent(base64.b64encode(audio).decode())]),
        artifact_root=tmp_path / "data",
        workspace=tmp_path,
    )

    result = asyncio.run(server.call("speak", {}))

    record = result["files"][0]

    assert record["mime"] == "audio/mpeg"
    assert record["path"].endswith(".mp3")


def test_resource_link_is_reported_as_a_link_not_saved(tmp_path):
    server = _server_with(
        _BlocksResult(
            [ResourceLink("file:///tmp/a.pdf", "a.pdf", mime_type="application/pdf", size=1234)]
        ),
        artifact_root=tmp_path / "data",
        workspace=tmp_path,
    )

    result = asyncio.run(server.call("find", {}))

    assert "files" not in result
    assert result["links"] == [
        {
            "uri": "file:///tmp/a.pdf",
            "name": "a.pdf",
            "mimeType": "application/pdf",
            "size": 1234,
        }
    ]

    assert not (tmp_path / "data").exists()


def test_an_unknown_content_type_becomes_a_warning(tmp_path):
    server = _server_with(
        _BlocksResult([UnknownContent()]),
        artifact_root=tmp_path / "data",
        workspace=tmp_path,
    )

    result = asyncio.run(server.call("project", {}))

    assert result == {"ok": True, "warnings": ["unsupported content type: hologram"]}

    assert not (tmp_path / "data").exists()
