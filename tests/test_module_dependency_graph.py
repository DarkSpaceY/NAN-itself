"""
Module dependency graph fail fast.

A dependency cycle must surface at registration time as a
RuntimeError carrying the cycle path; it must never degrade
into a silently tolerated graph. The check runs before anything
is installed, so the Module whose arrival closed the cycle
leaves no trace and the remaining Modules keep working.
Acyclic chains register normally and stop in
dependents-before-dependencies order.

Every Facade is pointed at tmp dirs only (never the real
builtin/modules -- real modules would start).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from nan_itself.modules.model import (
    Module,
)
from nan_itself.modules.runtime import (
    Facade,
)
from nan_itself.utils import paths as _paths
from nan_itself.utils.hotload import directory_fingerprint


def run(coro):
    return asyncio.run(coro)


def _write_module(modules_dir: Path, name: str, source: str) -> Path:
    folder = modules_dir / name

    folder.mkdir(parents=True, exist_ok=True)

    (folder / "main.py").write_text(source, encoding="utf-8")

    return folder


def _facade(tmp_path: Path, monkeypatch) -> Facade:
    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    return Facade()


# ============================================================================
# Registration rejects cycles
# ============================================================================


_A_SOURCE = """
import asyncio

from nan_itself.modules import Module


class A(Module):
    requires = ("b",)

    async def start(self):
        await asyncio.Event().wait()
"""


_B_SOURCE = """
import asyncio

from nan_itself.modules import Module


class B(Module):
    requires = ("a",)

    async def start(self):
        await asyncio.Event().wait()
"""


async def _register_cycle(tmp_path: Path, monkeypatch) -> None:
    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    a_folder = _write_module(modules_dir, "a", _A_SOURCE)
    b_folder = _write_module(modules_dir, "b", _B_SOURCE)

    # Loading a alone is fine: the missing dependency is only
    # a warning, not a cycle.
    await facade._load_or_reload_folder(
        a_folder,
        directory_fingerprint(a_folder),
    )

    # Loading b completes the cycle: registration must fail fast.
    await facade._load_or_reload_folder(
        b_folder,
        directory_fingerprint(b_folder),
    )


def test_registering_cyclic_modules_fails_fast(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError) as exc_info:
        run(_register_cycle(tmp_path, monkeypatch))

    message = str(exc_info.value)

    assert "Module dependency cycle detected" in message

    assert "a" in message

    assert "b" in message


def test_rejected_cycle_registration_leaves_live_state_untouched(
    tmp_path,
    monkeypatch,
):
    """
    The cycle error is unchanged -- it still propagates -- but it must
    be raised before anything is installed.

    The graph is checked on a copy, so a Module whose arrival would
    close a cycle never reaches the registry. If it did, the graph
    would stay cyclic and every later _topological_order() call would
    poison start(), stop() and _reconcile() for every Module, not just
    the offending one.
    """

    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True)

    a_folder = _write_module(modules_dir, "a", _A_SOURCE)
    b_folder = _write_module(modules_dir, "b", _B_SOURCE)

    async def scenario() -> None:
        await facade._load_or_reload_folder(
            a_folder,
            directory_fingerprint(a_folder),
        )

        record_before = facade.modules["a"]

        dataspace_before = facade.dataspaces["a"]

        readers_before = dict(record_before.instance.dependencies)

        # Loading b completes the cycle.
        with pytest.raises(RuntimeError) as exc_info:
            await facade._load_or_reload_folder(
                b_folder,
                directory_fingerprint(b_folder),
            )

        assert "Module dependency cycle detected" in str(exc_info.value)

        # Nothing was installed: the rejected id is absent and the
        # surviving record is the very same object.
        assert set(facade.modules) == {"a"}

        assert facade.modules["a"] is record_before

        assert set(facade.dataspaces) == {"a"}

        assert facade.dataspaces["a"] is dataspace_before

        # No reader was handed to a dependent either.
        assert record_before.instance.dependencies == readers_before

        # The graph is still acyclic, so ordering -- and therefore
        # start(), stop() and _reconcile() -- keep working.
        assert facade._topological_order() == ["a"]

        # Nothing durable was written for a Module that never
        # finished registering.
        assert not (modules_dir / "b" / "data" / "dataspace.json").exists()

        # The rejected folder left no synthetic package behind.
        assert not any(name.startswith("_nan_pkg_module_b") for name in sys.modules)

        # Both ends of the lifecycle still work afterwards.
        await facade.start()

        await facade.stop()

    run(scenario())


# ============================================================================
# Acyclic chain registers and stops dependents first
# ============================================================================


def test_acyclic_chain_stops_dependents_first(tmp_path, monkeypatch):
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

    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    for name in ("a", "b", "c"):
        (modules_dir / name).write_text(
            "# placeholder",
            encoding="utf-8",
        )

    facade._register_module_class(
        C,
        source=str(modules_dir / "c"),
        source_fingerprint=(1, 1),
    )

    facade._register_module_class(
        B,
        source=str(modules_dir / "b"),
        source_fingerprint=(1, 1),
    )

    facade._register_module_class(
        A,
        source=str(modules_dir / "a"),
        source_fingerprint=(1, 1),
    )

    facade._rebuild_dependency_graph()

    # The acyclic chain registered successfully.
    assert set(facade.modules) == {"a", "b", "c"}

    async def scenario() -> None:
        for module_id in ("a", "b", "c"):
            await facade._try_start(facade.modules[module_id])

        await facade.stop()

    run(scenario())

    # Dependents stop before their dependencies.
    assert stop_order == ["a", "b", "c"]
