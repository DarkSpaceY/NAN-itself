"""
Provider contracts.

Pure value types and constants shared by every layer of the
tools package. Nothing in here imports from sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal


PROVIDER_KIND_MCP = "mcp"
PROVIDER_KIND_LOCAL = "local"

ProviderKind = Literal[
    "mcp",
    "local",
]

ProviderOrigin = Literal[
    "builtin",
    "workspace",
]

LOCAL_TOOL_HEADER = "# @tool"

LOCAL_TOOL_HEADER_SCAN_LINES = 20

# A hung tool call must never freeze the agent loop. Every call is
# bounded; timeouts surface to the model as error results.
DEFAULT_TOOL_TIMEOUT = 300.0


@dataclass(frozen=True)
class ProviderSpec:
    """
    Declarative configuration for one tool provider.

    kind="mcp":
        A stdio MCP server described by command/args/env/cwd.

    kind="local":
        An in-process Python class tool provider. `file` points at
        the workspace source it was loaded from, or stays None for
        builtin classes registered directly.
    """

    name: str

    kind: ProviderKind = PROVIDER_KIND_MCP

    command: str | None = None
    args: tuple[str, ...] = ()
    env: MappingProxyType[str, str] | None = None
    cwd: str | None = None

    file: Path | None = None

    source: str = "<unknown>"
    origin: ProviderOrigin = "builtin"
