"""
Built-in module registration.

Register builtin Module classes here; the Facade loads them
before any workspace discovery and they participate in the same
dependency graph, persistence and supervision as workspace
modules — minus file hot reload.

Example:

    class Memory(Module):
        id = "memory"

        async def start(self): ...

    BUILTIN_MODULES = (Memory,)
"""

from __future__ import annotations

from src.nan_itself.modules.model import (
    Module,
)


# Register builtin module classes here.
from src.nan_itself.modules.builtin.memory import (
    MemoryModule,
)

from src.nan_itself.modules.builtin.plan import (
    PlanModule,
)

BUILTIN_MODULES: tuple[
    type[Module],
    ...,
] = ()

__all__ = ["BUILTIN_MODULES", "MemoryModule", "PlanModule"]
