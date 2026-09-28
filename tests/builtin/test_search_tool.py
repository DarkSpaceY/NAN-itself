"""
Tests for the builtin 'search' local tool provider.

The provider shells out to `uvx searxng-cli`; every test replaces the
subprocess with a fake so no network access happens here.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nan_itself.tools import local
from nan_itself.tools.provider import Provider

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEARCH_FILE = (
    PROJECT_ROOT / "builtin" / "tools" / "local" / "search.py"
)


def _build_provider(cls: type) -> Provider:
    return local.build_provider(
        local.local_provider_spec(
            name="search",
            file=SEARCH_FILE,
            source=str(SEARCH_FILE),
        ),
        cls,
    )

FAKE_PAYLOAD = {
    "query": "test query",
    "results": [
        {
            "url": "https://example.com/",
            "title": "Example Result",
            "content": "Example content",
            "engine": "bing",
            "score": 1.0,
        }
    ],
    "engine_stats": {
        "bing": {"elapsed_sec": 0.5, "error": None}
    },
    "elapsed_sec": 0.5,
    "result_count": 1,
}


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes,
        stderr: bytes,
        returncode: int,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False

    async def communicate(
        self,
    ) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True


def _load_module() -> tuple[
    ModuleType,
    type,
]:
    cls, module_name = local.load_class_from_file(
        SEARCH_FILE
    )

    return sys.modules[module_name], cls


def _install_fake_exec(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: bytes,
    stderr: bytes = b"",
    returncode: int = 0,
) -> tuple[_FakeProcess, dict]:
    process = _FakeProcess(
        stdout, stderr, returncode
    )

    capture: dict = {}

    async def fake_exec(
        *argv: str, **kwargs: object
    ) -> _FakeProcess:
        capture["argv"] = argv
        capture["env"] = kwargs.get("env")

        return process

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_exec,
    )

    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "/fake/uvx",
    )

    return process, capture


def test_loads_single_provider() -> None:
    module, cls = _load_module()

    assert cls.__name__ == "SearchProvider"

    assert cls.id == "search"

    instance = cls()

    assert instance.tool_names() == (
        "web_search",
    )


def test_web_search_input_schema() -> None:
    _, cls = _load_module()

    instance = cls()

    tools = instance.build_tools()

    schema = tools[
        "web_search"
    ].inputSchema

    assert schema["required"] == ["query"]

    properties = schema["properties"]

    assert (
        properties["max_results"]["default"]
        == 10
    )

    assert (
        properties["language"]["default"]
        == "all"
    )


@pytest.mark.asyncio
async def test_web_search_parses_cli_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # One load only: load_class_from_file re-execs the source
    # each call, so the class's globals belong to the module
    # object returned alongside it -- patch that one.
    module, cls = _load_module()

    settings = tmp_path / "settings.yml"

    settings.write_text(
        "use_default_settings: true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "_settings_file",
        lambda: str(settings),
    )

    _, capture = _install_fake_exec(
        monkeypatch,
        stdout=json.dumps(
            FAKE_PAYLOAD
        ).encode("utf-8"),
    )

    provider = _build_provider(cls)

    result = await provider.call_tool(
        "web_search",
        {
            "query": "test query",
            "engines": "bing",
            "category": "it",
        },
    )

    assert not result.isError

    payload = json.loads(
        result.content[0].text
    )

    assert payload["result_count"] == 1

    assert (
        payload["results"][0]["title"]
        == "Example Result"
    )

    argv = capture["argv"]

    assert "/fake/uvx" in argv

    assert "--from" in argv

    assert "searxng-cli==0.1.0" in argv

    assert "test query" in argv

    assert "--engines" in argv

    assert "bing" in argv

    assert "--category" in argv

    assert "it" in argv

    env = capture["env"]

    assert env["SEARXNG_CLI_SETTINGS"] == str(
        settings
    )


@pytest.mark.asyncio
async def test_web_search_error_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_module()

    _install_fake_exec(
        monkeypatch,
        stdout=b"",
        stderr=b"engine boom",
        returncode=1,
    )

    cls = _load_module()[1]

    provider = _build_provider(cls)

    result = await provider.call_tool(
        "web_search",
        {"query": "q"},
    )

    assert result.isError

    assert "engine boom" in (
        result.content[0].text
    )


@pytest.mark.asyncio
async def test_web_search_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_module()

    _install_fake_exec(
        monkeypatch,
        stdout=b"not json",
    )

    cls = _load_module()[1]

    provider = _build_provider(cls)

    result = await provider.call_tool(
        "web_search",
        {"query": "q"},
    )

    assert result.isError

    assert "invalid JSON" in (
        result.content[0].text
    )


def test_settings_file_resolution() -> None:
    module, _ = _load_module()

    resolved = module._settings_file()

    assert resolved is not None

    assert resolved.endswith(
        "config/searxng.yml"
    )
