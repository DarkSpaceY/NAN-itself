"""
Modules package public surface.

Layering (dependencies point downward only):

    runtime ──> { deps, loading, persistence, reload } ──> model

Import from this package, never from sibling modules.
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

from .runtime import (
    Facade,
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
