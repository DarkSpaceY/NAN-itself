"""
Tools package public surface.

Layering (dependencies point downward only):

    server ──> runtime ──> provider <── { mcp, local }
                  │             │
                  └──> results <── spec

        watcher: hot-reload bookkeeping used by runtime

Import from this package, never from sibling modules.
"""

from .local import (
    LocalToolMethod,
    LocalToolProvider,
    tool,
)
from .provider import (
    Provider,
)
from .runtime import (
    ProviderRuntime,
)
from .server import (
    MCPFacade,
)
from .spec import (
    DEFAULT_TOOL_TIMEOUT,
    LOCAL_TOOL_HEADER,
    LOCAL_TOOL_HEADER_SCAN_LINES,
    PROVIDER_KIND_LOCAL,
    PROVIDER_KIND_MCP,
    ProviderKind,
    ProviderSpec,
)

__all__ = [
    # local backend framework
    "LocalToolMethod",
    "LocalToolProvider",
    "tool",
    # provider layer
    "Provider",
    "ProviderSpec",
    "ProviderKind",
    "PROVIDER_KIND_MCP",
    "PROVIDER_KIND_LOCAL",
    "LOCAL_TOOL_HEADER",
    "LOCAL_TOOL_HEADER_SCAN_LINES",
    "DEFAULT_TOOL_TIMEOUT",
    # runtime
    "ProviderRuntime",
    # standalone server shell
    "MCPFacade",
]
