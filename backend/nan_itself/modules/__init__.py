"""
Modules package public surface.

Layering (dependencies point downward only):

    runtime ──> { deps, loading, persistence, reload } ──> model

Import from this package, never from sibling modules.

The runtime (and everything it pulls in) loads lazily on first
attribute access. This keeps light consumers of this package --
such as the agent layer, which only needs the data model --
able to import nan_itself.modules.model directly and never pay
for the runtime at import time.
"""

from .model import (
    ChannelSpec,
    DataSpace,
    DataSpaceReader,
    DuplicateModuleError,
    Module,
    ModuleRecord,
    ModuleState,
    Turn,
)

__all__ = [
    "ChannelSpec",
    "DataSpace",
    "DataSpaceReader",
    "Module",
    "ModuleRecord",
    "ModuleState",
    "Turn",
    "DuplicateModuleError",
    "Facade",
]


def __getattr__(name: str):
    # Lazy re-export of the runtime machinery: keeps the agent
    # layer's `import nan_itself.modules.model` free of runtime
    # side effects. Do NOT hoist this import.
    if name == "Facade":
        from .runtime import Facade

        return Facade

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
