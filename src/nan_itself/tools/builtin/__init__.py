"""
Built-in tool provider registration.

Two kinds of built-ins live here:

    mcp/    one YAML file describing builtin stdio MCP servers
    local/  Python modules defining builtin LocalToolProvider classes

Builtin means: shipped with the package, loaded before any
workspace discovery, and never shadowable by workspace files.
"""

from __future__ import annotations

from pathlib import Path

from ..local import (
    LocalToolProvider,
)


# Directory of this package; also the anchor for packaged config.
BUILTIN_DIR = Path(__file__).resolve().parent

# Builtin stdio MCP server configuration.
BUILTIN_MCP_CONFIG = BUILTIN_DIR / "mcp" / "tools.yaml"

# Register builtin local tool classes here.
#
# Example:
#
# from .builtin.local.example import ExampleTools
#
# BUILTIN_TOOLS = (ExampleTools,)
BUILTIN_TOOLS: tuple[
    type[LocalToolProvider],
    ...,
] = ()
