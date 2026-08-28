"""
Modules package public surface.

Layering (dependencies point downward only):

    runtime ──> { deps, loading, persistence, reload } ──> model

        builtin: built-in registration consumed by runtime

Import from this package, never from sibling modules.
"""

from .model import (
    DataSpace,
    DataSpaceReader,
    MODULE_HEADER,
    Module,
    ModuleRecord,
    ModuleState,
    ModuleTurn,
    TurnRecord,
    DuplicateModuleError,
)

from .runtime import (
    BUILTIN_MODULES,
    Facade,
)

__all__ = [
    "DataSpace",
    "DataSpaceReader",
    "MODULE_HEADER",
    "Module",
    "ModuleRecord",
    "ModuleState",
    "ModuleTurn",
    "TurnRecord",
    "DuplicateModuleError",
    "BUILTIN_MODULES",
    "Facade",
]
