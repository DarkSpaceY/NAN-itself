"""
Tools package public surface.

Layering (dependencies point downward only):

    server ──> runtime ──> { mcp, local } ──> provider

    shared leaves (imported by the layers above, import nothing
    from siblings): spec, results
        spec     <- provider, mcp, local, runtime
        results  <- provider, local, runtime, server

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
    "DEFAULT_TOOL_TIMEOUT",
    # runtime
    "ProviderRuntime",
    # standalone server shell
    "MCPFacade",
]
