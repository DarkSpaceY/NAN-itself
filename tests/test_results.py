"""
results + spec contract locks.

Small modules, but their exact shapes are public API: lock the
defaults and serialization behaviors so future refactors cannot
silently drift.
"""

from __future__ import annotations

from types import MappingProxyType

import mcp.types as types

from src.nan_itself.tools import (
    DEFAULT_TOOL_TIMEOUT,
    LOCAL_TOOL_HEADER,
    LOCAL_TOOL_HEADER_SCAN_LINES,
    PROVIDER_KIND_LOCAL,
    PROVIDER_KIND_MCP,
    ProviderSpec,
)
from src.nan_itself.tools.results import (
    error_result,
    serialize_value,
    text_result,
)


def test_constants_have_locked_values():
    assert PROVIDER_KIND_MCP == "mcp"
    assert PROVIDER_KIND_LOCAL == "local"
    assert LOCAL_TOOL_HEADER == "# @tool"
    assert LOCAL_TOOL_HEADER_SCAN_LINES == 20
    assert DEFAULT_TOOL_TIMEOUT == 300.0


def test_provider_spec_defaults_are_mcp_builtin():
    spec = ProviderSpec(name="x")

    assert spec.kind == "mcp"
    assert spec.origin == "builtin"
    assert spec.source == "<unknown>"
    assert spec.args == ()
    assert spec.command is None
    assert spec.env is None
    assert spec.cwd is None
    assert spec.file is None


def test_text_result_wraps_single_text_content():
    result = text_result("hello")

    assert isinstance(result, types.CallToolResult)
    assert result.isError is False
    assert len(result.content) == 1

    content = result.content[0]

    assert isinstance(content, types.TextContent)
    assert content.text == "hello"


def test_error_result_flags_is_error():
    result = error_result("boom")

    assert isinstance(result, types.CallToolResult)
    assert result.isError is True
    assert result.content[0].text == "boom"


def test_serialize_value_none_is_null_literal():
    assert serialize_value(None) == "null"


def test_serialize_value_passes_strings_through_verbatim():
    assert serialize_value("") == ""
    assert serialize_value("中文保持原样") == "中文保持原样"


def test_serialize_value_dumps_json_ensuring_ascii_false():
    value = {"名称": ["a", 1]}

    encoded = serialize_value(value)

    assert "名称" in encoded
    assert '"a"' in encoded


def test_serialize_value_falls_back_to_str():
    sentinel = object()

    encoded = serialize_value(sentinel)

    # Unserializable objects go through json's default=str hook,
    # so the result is the str() rendered *as a JSON string*.
    assert encoded.startswith('"')
    assert encoded.endswith('"')
    assert "<object object" in encoded


def test_spec_env_field_accepts_mapping_proxy():
    env = MappingProxyType({"A": "b"})

    spec = ProviderSpec(name="x", env=env)

    assert dict(spec.env) == {"A": "b"}
