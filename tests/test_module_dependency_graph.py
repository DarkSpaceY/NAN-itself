"""
Module dependency graph fail fast.

A dependency cycle must surface at registration time as a
RuntimeError carrying the cycle path; it must never degrade
into a silently tolerated graph. Acyclic chains register
normally and stop in dependents-before-dependencies order.

Every Facade is pointed at tmp dirs only (never the real
builtin/modules -- real modules would start).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nan_itself.modules.model import (
    Module,
)
from nan_itself.modules.runtime import (
    Facade,
)


def run(coro):
    return asyncio.run(coro)


def _facade(tmp_path: Path) -> Facade:
    return Facade(
        workspace_modules=tmp_path / "modules",
        builtin_modules_dir=tmp_path / "builtin",
        data_dir=tmp_path / "data",
    )


# ============================================================================
# Registration rejects cycles
# ============================================================================


_A_SOURCE = """# @module
import asyncio

from nan_itself.modules import Module


class A(Module):
    id = "a"
    requires = ("b",)

    async def start(self):
        await asyncio.Event().wait()
"""


_B_SOURCE = """# @module
import asyncio

from nan_itself.modules import Module


class B(Module):
    id = "b"
    requires = ("a",)

    async def start(self):
        await asyncio.Event().wait()
"""


async def _register_cycle(tmp_path: Path) -> None:
    facade = _facade(tmp_path)

    a_source = tmp_path / "a.py"
    b_source = tmp_path / "b.py"

    a_source.write_text(_A_SOURCE, encoding="utf-8")

    b_source.write_text(_B_SOURCE, encoding="utf-8")

    # Loading a alone is fine: the missing dependency is only
    # a warning, not a cycle.
    await facade._load_or_reload_file(
        a_source,
        facade._fingerprint(a_source),
    )

    # Loading b completes the cycle: registration must fail fast.
    await facade._load_or_reload_file(
        b_source,
        facade._fingerprint(b_source),
    )


def test_registering_cyclic_modules_fails_fast(tmp_path):
    with pytest.raises(RuntimeError) as exc_info:
        run(_register_cycle(tmp_path))

    message = str(exc_info.value)

    assert "Module dependency cycle detected" in message

    assert "a" in message

    assert "b" in message


# ============================================================================
# Acyclic chain registers and stops dependents first
# ============================================================================


def test_acyclic_chain_stops_dependents_first(tmp_path):
    stop_order: list[str] = []

    class C(Module):
        id = "c"

        async def start(self):
            await asyncio.Event().wait()

        async def stop(self):
            stop_order.append("c")

    class B(Module):
        id = "b"
        requires = ("c",)

        async def start(self):
            await asyncio.Event().wait()

        async def stop(self):
            stop_order.append("b")

    class A(Module):
        id = "a"
        requires = ("b",)

        async def start(self):
            await asyncio.Event().wait()

        async def stop(self):
            stop_order.append("a")

    facade = _facade(tmp_path)

    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text(
            "# placeholder",
            encoding="utf-8",
        )

    facade._register_module_class(
        C,
        source_path=str(tmp_path / "c.py"),
        source_fingerprint=(1, 1),
    )

    facade._register_module_class(
        B,
        source_path=str(tmp_path / "b.py"),
        source_fingerprint=(1, 1),
    )

    facade._register_module_class(
        A,
        source_path=str(tmp_path / "a.py"),
        source_fingerprint=(1, 1),
    )

    facade._rebuild_dependency_graph()

    # The acyclic chain registered successfully.
    assert set(facade.modules) == {"a", "b", "c"}

    async def scenario() -> None:
        for module_id in ("a", "b", "c"):
            await facade._try_start(
                facade.modules[module_id]
            )

        await facade.stop()

    run(scenario())

    # Dependents stop before their dependencies.
    assert stop_order == ["a", "b", "c"]
