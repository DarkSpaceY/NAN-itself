"""
Modules package public surface.

Layering (dependencies point downward only):

    runtime ──> { deps, loading, persistence, reload } ──> model

Import from this package, never from sibling modules.

The runtime (and everything it pulls in) loads lazily on first
attribute access, so light consumers of this package -- such as
the agent layer, which only needs the data model -- do not pay
for it at import time.
"""

from .model import (
    DataSpace,
    DataSpaceReader,
    MODULE_HEADER,
    Module,
    ModuleRecord,
    ModuleState,
    Turn,
    DuplicateModuleError,
)

__all__ = [
    "DataSpace",
    "DataSpaceReader",
    "MODULE_HEADER",
    "Module",
    "ModuleRecord",
    "ModuleState",
    "Turn",
    "DuplicateModuleError",
    "Facade",
]


def __getattr__(name: str):
    # Lazy export: keeps the heavy runtime machinery out of the
    # import of this package.
    if name == "Facade":
        from .runtime import Facade

        return Facade

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
