"""
Tools package public surface.

Layering (dependencies point downward only):

    server ──> runtime ──> { mcp, local } ──> provider

    shared leaves (imported by the layers above, import nothing
    from siblings): spec, results, watcher
        spec     <- provider, mcp, local, runtime
        results  <- provider, local, runtime, server
        watcher  <- runtime (hot-reload bookkeeping)

Import from this package, never from sibling modules.
"""

from .local import (
    LocalToolMethod,
    ToolSet,
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
    "ToolSet",
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
