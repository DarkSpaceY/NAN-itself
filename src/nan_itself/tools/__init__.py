"""
Tools package public surface.

Layering (dependencies point downward only):

    view ─┐
    server ├─ runtime ──> provider <── { mcp, local }
          │                   │             │
          └────────────────> results <── spec

        watcher: hot-reload bookkeeping used by runtime
        builtin: built-in registration consumed by runtime

Import from this package, never from sibling modules.
"""

from src.nan_itself.tools.local import (
    LocalToolMethod,
    LocalToolProvider,
    tool,
)
from src.nan_itself.tools.provider import (
    Provider,
)
from src.nan_itself.tools.runtime import (
    BUILTIN_TOOLS,
    ProviderRuntime,
)
from src.nan_itself.tools.server import (
    MCPFacade,
)
from src.nan_itself.tools.spec import (
    DEFAULT_TOOL_TIMEOUT,
    LOCAL_TOOL_HEADER,
    LOCAL_TOOL_HEADER_SCAN_LINES,
    PROVIDER_KIND_LOCAL,
    PROVIDER_KIND_MCP,
    ProviderKind,
    ProviderOrigin,
    ProviderSpec,
)
from src.nan_itself.tools.view import (
    ROUTE_TOOL_ARGUMENT,
    ROUTE_TOOL_NAME,
    AgentToolView,
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
    "ProviderOrigin",
    "PROVIDER_KIND_MCP",
    "PROVIDER_KIND_LOCAL",
    "LOCAL_TOOL_HEADER",
    "LOCAL_TOOL_HEADER_SCAN_LINES",
    "DEFAULT_TOOL_TIMEOUT",
    # runtime
    "ProviderRuntime",
    "BUILTIN_TOOLS",
    # exposure
    "AgentToolView",
    "ROUTE_TOOL_NAME",
    "ROUTE_TOOL_ARGUMENT",
    # standalone server shell
    "MCPFacade",
]
