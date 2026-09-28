"""
Module removal releases its DataSpace and detaches dependents.

Removing a Module must:

    - persist the record's state so a later re-add restores the
      state as of removal
    - drop the module id from Facade.dataspaces (no tombstone)
    - detach that dependency edge from every remaining dependent

Registering a Module must re-attach the missing reader to its
existing dependents, and only to those that declare it.

Every Facade is pointed at tmp dirs only (never the real
builtin/modules -- real modules would start).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nan_itself.modules.model import (
    Module,
)
from nan_itself.modules.runtime import (
    Facade,
)
from nan_itself.utils import paths as _paths


def run(coro):
    return asyncio.run(coro)


def _facade(
    tmp_path: Path,
    monkeypatch,
) -> Facade:
    monkeypatch.setattr(
        _paths,
        "repo_root",
        lambda: tmp_path,
    )

    return Facade()


def _register(
    facade: Facade,
    cls,
    modules_dir: Path,
    name: str,
):
    return facade._register_module_class(
        cls,
        source=str(modules_dir / name),
        source_fingerprint=(1, 1),
    )


# ============================================================================
# Removal detaches the dependent's reader
# ============================================================================


def _declare_modules():
    class A(Module):
        id = "a"

    class B(Module):
        id = "b"
        requires = ("a",)

    return A, B


def test_removal_detaches_dependent_reader(
    tmp_path,
    monkeypatch,
):
    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    A, B = _declare_modules()

    a_record = _register(
        facade,
        A,
        modules_dir,
        "a.py",
    )

    b_record = _register(
        facade,
        B,
        modules_dir,
        "b.py",
    )

    facade._rebuild_dependency_graph()

    assert "a" in b_record.instance.dependencies

    run(
        facade._remove_record(a_record)
    )

    # The live dependent must not keep reading a removed Module.
    assert "a" not in b_record.instance.dependencies


# ============================================================================
# Removal releases the DataSpace
# ============================================================================


def test_removal_releases_dataspace(
    tmp_path,
    monkeypatch,
):
    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    A, _ = _declare_modules()

    a_record = _register(
        facade,
        A,
        modules_dir,
        "a.py",
    )

    assert "a" in facade.dataspaces

    run(
        facade._remove_record(a_record)
    )

    # No tombstone: the removed id no longer owns a DataSpace.
    assert "a" not in facade.dataspaces


# ============================================================================
# Removal persists the state as of removal
# ============================================================================


def test_removal_persists_state_for_readd(
    tmp_path,
    monkeypatch,
):
    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    class A(Module):
        id = "a"

        def serialize_state(self):
            return {
                "counter": self.counter
            }

    a_record = _register(
        facade,
        A,
        modules_dir,
        "a.py",
    )

    # Seed an OLDER disk snapshot.
    a_record.instance.counter = 1

    facade.dataspaces["a"].publish(
        {
            "value": "old"
        }
    )

    facade._save_record_state(
        a_record
    )

    # Move the live state forward: this is the state as of
    # removal.
    a_record.instance.counter = 7

    facade.dataspaces["a"].publish(
        {
            "value": "live"
        }
    )

    run(
        facade._remove_record(a_record)
    )

    class AReadded(Module):
        id = "a"

        def restore_state(self, state):
            self.restored = state

    readded = _register(
        facade,
        AReadded,
        modules_dir,
        "a.py",
    )

    # The state as of removal is restored, not the older
    # snapshot and not an empty state.
    assert readded.instance.restored == {
        "counter": 7
    }

    assert facade.dataspaces["a"].snapshot() == {
        "value": "live"
    }


# ============================================================================
# Re-registration re-attaches existing dependents, targeted
# ============================================================================


def test_readd_reattaches_dependent_targeted(
    tmp_path,
    monkeypatch,
):
    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    class A(Module):
        id = "a"

    class B(Module):
        id = "b"
        requires = ("a",)

    class C(Module):
        id = "c"
        requires = ()

    a_record = _register(
        facade,
        A,
        modules_dir,
        "a.py",
    )

    b_record = _register(
        facade,
        B,
        modules_dir,
        "b.py",
    )

    c_record = _register(
        facade,
        C,
        modules_dir,
        "c.py",
    )

    facade._rebuild_dependency_graph()

    run(
        facade._remove_record(a_record)
    )

    # B lost the reader; C never had one.
    assert "a" not in b_record.instance.dependencies

    c_dependencies = c_record.instance.dependencies

    # Re-register A.
    _register(
        facade,
        A,
        modules_dir,
        "a.py",
    )

    # B regained a live reader onto A's DataSpace.
    assert "a" in b_record.instance.dependencies

    facade.dataspaces["a"].publish(
        {
            "value": "after"
        }
    )

    assert (
        b_record.instance.dependencies["a"].snapshot()
        == {
            "value": "after"
        }
    )

    # C declared no such dependency: it must be untouched.
    assert (
        c_record.instance.dependencies
        is c_dependencies
    )

    assert dict(
        c_record.instance.dependencies
    ) == {}


# ============================================================================
# Re-registration never grants a reader to a non-dependent
# ============================================================================


def test_readd_does_not_grant_reader_to_non_dependent(
    tmp_path,
    monkeypatch,
):
    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    class A(Module):
        id = "a"

    class C(Module):
        id = "c"
        requires = ()

    # C is registered before A and declares no dependency.
    c_record = _register(
        facade,
        C,
        modules_dir,
        "c.py",
    )

    _register(
        facade,
        A,
        modules_dir,
        "a.py",
    )

    assert "a" not in c_record.instance.dependencies


# ============================================================================
# Dependent registered before its dependency gains the reader
# ============================================================================


def test_dependent_registered_before_dependency_gets_reader(
    tmp_path,
    monkeypatch,
):
    facade = _facade(tmp_path, monkeypatch)

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    A, B = _declare_modules()

    # Boot order hazard: B is registered while A is absent.
    b_record = _register(
        facade,
        B,
        modules_dir,
        "b.py",
    )

    assert "a" not in b_record.instance.dependencies

    # A later appears: the missing edge must be attached.
    _register(
        facade,
        A,
        modules_dir,
        "a.py",
    )

    assert "a" in b_record.instance.dependencies
