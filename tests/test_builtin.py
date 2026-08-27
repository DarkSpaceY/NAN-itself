"""
Builtin registration package tests.

The packaged YAML ships with the wheel and must stay parseable
by the same code path workspace configs use. The local registry
is empty today but its type is part of the contract.
"""

from __future__ import annotations

from src.nan_itself.tools.builtin import (
    BUILTIN_MCP_CONFIG,
    BUILTIN_TOOLS,
)
from src.nan_itself.tools import mcp as mcp_backend


EXPECTED_BUILTIN_SERVERS = {
    "files",
    "playwright",
    "command",
    "calculation",
    "prolog",
}


def test_builtin_config_exists_inside_package():
    assert BUILTIN_MCP_CONFIG.is_file()
    assert BUILTIN_MCP_CONFIG.parent.name == "mcp"


def test_builtin_config_parses_via_standard_path():
    config = mcp_backend.load_yaml(BUILTIN_MCP_CONFIG)

    specs = mcp_backend.parse_builtin_config(
        config,
        BUILTIN_MCP_CONFIG,
    )

    assert {spec.name for spec in specs} == (
        EXPECTED_BUILTIN_SERVERS
    )

    for spec in specs:
        assert spec.origin == "builtin"
        assert spec.kind == "mcp"
        assert spec.command


def test_builtin_local_registry_is_a_tuple():
    assert isinstance(BUILTIN_TOOLS, tuple)

    # Every registered class must satisfy the validator.
    from src.nan_itself.tools.local import validate_class

    for cls in BUILTIN_TOOLS:
        validate_class(cls)
