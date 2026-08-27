from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.nan_itself.modules import (
    DataSpace,
    DataSpaceReader,
    Facade,
    Module,
    ModuleState,
    ModuleTurn,
    TurnRecord,
)


# ============================================================================
# DataSpace
# ============================================================================


def test_dataspace_isolated_on_write_and_read():
    space = DataSpace("foo")

    original = {
        "nested": {
            "items": [1, 2],
        },
    }

    space.publish(original)

    # Caller must not retain a reference into DataSpace.
    original["nested"]["items"].append(3)

    assert space.snapshot() == {
        "nested": {
            "items": [1, 2],
        },
    }

    # Reader must receive a detached copy.
    snapshot = space.snapshot()
    snapshot["nested"]["items"].append(999)

    assert space.snapshot() == {
        "nested": {
            "items": [1, 2],
        },
    }


def test_dataspace_publish_replaces_state_atomically():
    space = DataSpace("foo")

    space.publish({
        "version": 1,
        "items": [1, 2],
    })

    first = space.snapshot()

    space.publish({
        "version": 2,
        "items": [3, 4],
    })

    second = space.snapshot()

    assert first == {
        "version": 1,
        "items": [1, 2],
    }

    assert second == {
        "version": 2,
        "items": [3, 4],
    }

    assert space.revision == 2


def test_dataspace_rejects_non_mapping():
    space = DataSpace("foo")

    with pytest.raises(TypeError):
        space.publish(["not", "a", "mapping"])


# ============================================================================
# Builtin discovery
# ============================================================================


@pytest.mark.asyncio
async def test_builtin_module_is_loaded_from_constructor(tmp_path):
    class BuiltinModule(Module):
        id = "builtin"

        async def start(self):
            self.data.publish({
                "source": "builtin",
            })

            while True:
                await asyncio.sleep(10)

    facade = Facade(
        workspace_modules=tmp_path / "workspace" / "modules",
        builtin_modules=(BuiltinModule,),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.05,
        retry_interval=0.05,
    )

    await facade.start()

    try:
        assert set(facade.modules) == {"builtin"}

        record = facade.modules["builtin"]

        assert record.origin == "builtin"
        assert record.source == "<builtin>"
        assert record.state is ModuleState.RUNNING

        assert facade.dataspaces["builtin"].snapshot() == {
            "source": "builtin",
        }
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_builtin_module_is_not_loaded_twice(tmp_path):
    class BuiltinModule(Module):
        id = "builtin"

        async def start(self):
            while True:
                await asyncio.sleep(10)

    facade = Facade(
        builtin_modules=(BuiltinModule,),
        workspace_modules=tmp_path / "workspace" / "modules",
        data_dir=tmp_path / "data" / "modules",
    )

    await facade.start()

    try:
        await facade.start()

        assert list(facade.modules) == ["builtin"]
    finally:
        await facade.stop()


# ============================================================================
# Workspace discovery
# ============================================================================


@pytest.mark.asyncio
async def test_workspace_module_is_discovered_from_file(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    module_file = workspace / "foo.py"

    module_file.write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    async def start(self):
        self.data.publish({
            "source": "workspace",
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.05,
        retry_interval=0.05,
    )

    await facade.start()

    try:
        assert set(facade.modules) == {"foo"}

        record = facade.modules["foo"]

        assert record.origin == "workspace"
        assert Path(record.source).resolve() == module_file.resolve()
        assert record.state is ModuleState.RUNNING

        assert facade.dataspaces["foo"].snapshot() == {
            "source": "workspace",
        }
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_workspace_file_without_module_header_is_ignored(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "ignored.py").write_text(
        """
from src.nan_itself.modules import Module


class Ignored(Module):
    id = "ignored"

    async def start(self):
        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
    )

    await facade.start()

    try:
        assert facade.modules == {}
        assert facade.dataspaces == {}
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_workspace_file_with_multiple_modules_is_rejected(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "bad.py").write_text(
        """
# @module

from src.nan_itself.modules import Module


class A(Module):
    id = "a"

    async def start(self):
        pass


class B(Module):
    id = "b"

    async def start(self):
        pass
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        assert "a" not in facade.modules
        assert "b" not in facade.modules
    finally:
        await facade.stop()

# ============================================================================
# Builtin + workspace together
# ============================================================================


@pytest.mark.asyncio
async def test_builtin_and_workspace_modules_share_one_registry(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    class Builtin(Module):
        id = "builtin"

        async def start(self):
            while True:
                await asyncio.sleep(10)

    (workspace / "workspace.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Workspace(Module):
    id = "workspace"

    async def start(self):
        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(Builtin,),
        data_dir=tmp_path / "data" / "modules",
    )

    await facade.start()

    try:
        assert set(facade.modules) == {
            "builtin",
            "workspace",
        }

        assert facade.modules["builtin"].origin == "builtin"
        assert facade.modules["workspace"].origin == "workspace"
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_workspace_cannot_override_builtin(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    class Builtin(Module):
        id = "same"

        async def start(self):
            while True:
                await asyncio.sleep(10)

    (workspace / "same.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Workspace(Module):
    id = "same"

    async def start(self):
        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(Builtin,),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        assert list(facade.modules) == ["same"]
        assert facade.modules["same"].origin == "builtin"
    finally:
        await facade.stop()


# ============================================================================
# Dependency graph / DataSpace dependency
# ============================================================================


@pytest.mark.asyncio
async def test_module_dependency_uses_dataspace_not_instance(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "foo.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    async def start(self):
        self.data.publish({
            "value": 42,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    (workspace / "bar.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Bar(Module):
    id = "bar"
    requires = ("foo",)

    async def start(self):
        foo = self.dependencies["foo"].snapshot()

        self.data.publish({
            "foo_value": foo["value"],
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.05,
        retry_interval=0.05,
    )

    await facade.start()

    try:
        assert facade.modules["foo"].state is ModuleState.RUNNING
        assert facade.modules["bar"].state is ModuleState.RUNNING

        assert facade.dependencies["bar"] == {"foo"}
        assert facade.dependents["foo"] == {"bar"}

        assert facade.dataspaces["bar"].snapshot() == {
            "foo_value": 42,
        }

        dependency = facade.modules[
            "bar"
        ].instance.dependencies["foo"]

        assert dependency.owner == "foo"
        assert dependency.snapshot() == {
            "value": 42,
        }

        # Read-only handle must not expose publish().
        assert not hasattr(dependency, "publish")

        # And it must not be the Foo Module instance.
        assert dependency is not facade.modules["foo"].instance

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_dependency_cycle_is_rejected(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "a.py").write_text(
        """
# @module

from src.nan_itself.modules import Module


class A(Module):
    id = "a"
    requires = ("b",)

    async def start(self):
        pass
""",
        encoding="utf-8",
    )

    (workspace / "b.py").write_text(
        """
# @module

from src.nan_itself.modules import Module


class B(Module):
    id = "b"
    requires = ("a",)

    async def start(self):
        pass
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
    )

    with pytest.raises(RuntimeError, match="cycle"):
        await facade.start()


@pytest.mark.asyncio
async def test_missing_dependency_does_not_crash_facade(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "foo.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"
    requires = ("missing",)

    async def start(self):
        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        assert "foo" in facade.modules
        assert facade.modules["foo"].state is ModuleState.RUNNING
        assert facade.modules["foo"].instance.dependencies == {}
    finally:
        await facade.stop()


# ============================================================================
# Supervision
# ============================================================================


@pytest.mark.asyncio
async def test_module_crash_is_supervised_and_retried(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "foo.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"
    attempts = 0

    async def start(self):
        type(self).attempts += 1

        if type(self).attempts == 1:
            raise RuntimeError("first attempt fails")

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        for _ in range(150):
            if (
                facade.modules["foo"].state
                is ModuleState.RUNNING
            ):
                break

            await asyncio.sleep(0.02)

        assert (
            facade.modules["foo"].state
            is ModuleState.RUNNING
        )

        assert (
            facade.modules["foo"].instance.attempts >= 2
        )
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_state_only_module_is_restarted(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "foo.py").write_text(
        """
# @module

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"
    attempts = 0

    async def start(self):
        type(self).attempts += 1

        self.data.publish({
            "attempts": type(self).attempts,
        })
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        for _ in range(150):
            state = facade.dataspaces["foo"].snapshot()

            if state.get("attempts", 0) >= 2:
                break

            await asyncio.sleep(0.02)

        state = facade.dataspaces["foo"].snapshot()

        assert state["attempts"] >= 2
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_stop_calls_module_stop(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "foo.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"
    stopped = False

    async def start(self):
        while True:
            await asyncio.sleep(10)

    async def stop(self):
        type(self).stopped = True
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    instance = facade.modules["foo"].instance

    await facade.stop()

    assert instance.stopped is True


# ============================================================================
# Turn snapshot / query
# ============================================================================


@pytest.mark.asyncio
async def test_query_receives_one_consistent_dataspace_snapshot(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "a.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class A(Module):
    id = "a"

    async def start(self):
        self.data.publish({
            "value": 1,
        })

        while True:
            await asyncio.sleep(10)

    async def query(self, turn):
        self.data.publish({
            "value": 2,
        })

        await asyncio.sleep(0)

        return str(turn.data["a"]["value"])
""",
        encoding="utf-8",
    )

    (workspace / "b.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class B(Module):
    id = "b"

    async def start(self):
        self.data.publish({
            "value": 10,
        })

        while True:
            await asyncio.sleep(10)

    async def query(self, turn):
        await asyncio.sleep(0)
        return str(turn.data["b"]["value"])
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        results = await facade.query(object())

        assert sorted(results) == [
            "1",
            "10",
        ]

        # Live DataSpace changed during query, but current turn
        # retained the original snapshot.
        assert facade.dataspaces["a"].snapshot() == {
            "value": 2,
        }
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_query_failure_does_not_take_module_down(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    (workspace / "foo.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    async def start(self):
        while True:
            await asyncio.sleep(10)

    async def query(self, turn):
        raise RuntimeError("query failure")
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        assert await facade.query(object()) == []
        assert facade.modules["foo"].state is ModuleState.RUNNING
    finally:
        await facade.stop()


# ============================================================================
# Persistence
# ============================================================================


@pytest.mark.asyncio
async def test_dataspace_persistence(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    data_dir = tmp_path / "data" / "modules"

    module_file = workspace / "foo.py"

    module_file.write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    async def start(self):
        self.data.publish({
            "value": 42,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        await asyncio.sleep(0.05)

        assert facade.dataspaces["foo"].snapshot() == {
            "value": 42,
        }

        facade.save_state()
    finally:
        await facade.stop()

    path = data_dir / "dataspace" / "foo.json"

    assert path.exists()

    stored = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert stored == {
        "value": 42,
    }

    facade2 = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    try:
        await facade2.start()

        assert facade2.dataspaces["foo"].snapshot() == {
            "value": 42,
        }
    finally:
        await facade2.stop()


@pytest.mark.asyncio
async def test_module_private_state_persistence(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    data_dir = tmp_path / "data" / "modules"

    module_file = workspace / "foo.py"

    module_file.write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    def __init__(self):
        self.counter = 0

    def serialize_state(self):
        return {
            "counter": self.counter,
        }

    def restore_state(self, state):
        self.counter = state.get("counter", 0)

    async def start(self):
        self.counter += 1

        self.data.publish({
            "counter": self.counter,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        await asyncio.sleep(0.05)

        assert facade.modules["foo"].instance.counter == 1
    finally:
        await facade.stop()

    private_path = data_dir / "private" / "foo.json"

    assert private_path.exists()

    stored = json.loads(
        private_path.read_text(encoding="utf-8")
    )

    assert stored == {
        "counter": 1,
    }

    facade2 = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    try:
        await facade2.start()

        # Restored value 1, then start() increments to 2.
        assert facade2.modules["foo"].instance.counter == 2
    finally:
        await facade2.stop()


@pytest.mark.asyncio
async def test_corrupt_or_invalid_private_state_does_not_block_other_modules(
    tmp_path,
):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    data_dir = tmp_path / "data" / "modules"
    private_dir = data_dir / "private"
    private_dir.mkdir(parents=True)

    (private_dir / "broken.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )

    (workspace / "broken.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Broken(Module):
    id = "broken"

    async def start(self):
        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    (workspace / "healthy.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Healthy(Module):
    id = "healthy"

    async def start(self):
        self.data.publish({
            "ok": True,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        assert (
            facade.modules["healthy"].state
            is ModuleState.RUNNING
        )

        assert (
            facade.modules["broken"].state
            is ModuleState.RUNNING
        )
    finally:
        await facade.stop()


# ============================================================================
# Hot reload
# ============================================================================


@pytest.mark.asyncio
async def test_hot_reload_preserves_dataspace_and_live_private_state(
    tmp_path,
):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    data_dir = tmp_path / "data" / "modules"

    module_file = workspace / "foo.py"

    module_file.write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    def __init__(self):
        self.counter = 0

    def restore_state(self, state):
        self.counter = state.get("counter", 0)

    def serialize_state(self):
        return {
            "counter": self.counter,
        }

    async def start(self):
        self.counter += 1

        self.data.publish({
            "version": 1,
            "value": 42,
            "counter": self.counter,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        await asyncio.sleep(0.05)

        old = facade.modules["foo"]

        assert old.generation == 0
        assert old.state is ModuleState.RUNNING
        assert old.instance.counter == 1

        module_file.write_text(
            """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    def __init__(self):
        self.counter = 0

    def restore_state(self, state):
        self.counter = state.get("counter", 0)

    def serialize_state(self):
        return {
            "counter": self.counter,
        }

    async def start(self):
        self.counter += 1

        old = self.data.snapshot()

        self.data.publish({
            "version": 2,
            "value": old["value"] + 1,
            "counter": self.counter,
        })

        while True:
            await asyncio.sleep(10)
""",
            encoding="utf-8",
        )

        for _ in range(150):
            await asyncio.sleep(0.02)

            current = facade.modules["foo"]

            if current.generation >= 1:
                break

        current = facade.modules["foo"]

        assert current.generation == 1
        assert current.state is ModuleState.RUNNING

        # Live private state:
        # old counter = 1
        # restore -> 1
        # new start -> 2
        assert current.instance.counter == 2

        # Same DataSpace, updated by the new generation.
        assert facade.dataspaces["foo"].snapshot() == {
            "version": 2,
            "value": 43,
            "counter": 2,
        }
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_hot_reload_failure_keeps_old_generation(
    tmp_path,
):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    data_dir = tmp_path / "data" / "modules"

    module_file = workspace / "foo.py"

    module_file.write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    async def start(self):
        self.data.publish({
            "version": 1,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        await asyncio.sleep(0.05)

        old = facade.modules["foo"]
        old_instance = old.instance

        assert old.generation == 0
        assert old.state is ModuleState.RUNNING

        module_file.write_text(
            """
# @module

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    async def start(self):
        raise RuntimeError("broken hot reload")
""",
            encoding="utf-8",
        )

        await asyncio.sleep(0.15)

        current = facade.modules["foo"]

        assert current.generation == 0
        assert current.instance is old_instance
        assert current.state is ModuleState.RUNNING

        assert facade.dataspaces["foo"].snapshot() == {
            "version": 1,
        }
    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_hot_reload_private_state_restore_failure_keeps_old_generation(
    tmp_path,
):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    data_dir = tmp_path / "data" / "modules"

    module_file = workspace / "foo.py"

    module_file.write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    def __init__(self):
        self.value = 123

    def restore_state(self, state):
        self.value = state["value"]

    def serialize_state(self):
        return {
            "value": self.value,
        }

    async def start(self):
        self.data.publish({
            "value": self.value,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        await asyncio.sleep(0.05)

        old = facade.modules["foo"]
        assert old.generation == 0

        module_file.write_text(
            """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    def restore_state(self, state):
        raise RuntimeError("cannot migrate state")

    async def start(self):
        self.data.publish({
            "value": 999,
        })

        while True:
            await asyncio.sleep(10)
""",
            encoding="utf-8",
        )

        await asyncio.sleep(0.15)

        current = facade.modules["foo"]

        assert current.generation == 0
        assert current.instance is old.instance
        assert current.state is ModuleState.RUNNING

    finally:
        await facade.stop()


# ============================================================================
# File removal / persistence of DataSpace
# ============================================================================


@pytest.mark.asyncio
async def test_removed_workspace_module_is_stopped_but_dataspace_is_retained(
    tmp_path,
):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    module_file = workspace / "foo.py"

    module_file.write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    async def start(self):
        self.data.publish({
            "value": 42,
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data" / "modules",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        await asyncio.sleep(0.05)

        assert "foo" in facade.modules
        assert facade.dataspaces["foo"].snapshot() == {
            "value": 42,
        }

        module_file.unlink()

        for _ in range(150):
            await asyncio.sleep(0.02)

            if "foo" not in facade.modules:
                break

        assert "foo" not in facade.modules

        # DataSpace intentionally survives Module removal.
        assert "foo" in facade.dataspaces
        assert facade.dataspaces["foo"].snapshot() == {
            "value": 42,
        }

    finally:
        await facade.stop()


# ============================================================================
# Persistence path / JSON format
# ============================================================================


@pytest.mark.asyncio
async def test_persistence_layout_is_separated(tmp_path):
    workspace = tmp_path / "workspace" / "modules"
    workspace.mkdir(parents=True)

    data_dir = tmp_path / "data" / "modules"

    (workspace / "foo.py").write_text(
        """
# @module

import asyncio

from src.nan_itself.modules import Module


class Foo(Module):
    id = "foo"

    def __init__(self):
        self.private_value = "private"

    def serialize_state(self):
        return {
            "private_value": self.private_value,
        }

    async def start(self):
        self.data.publish({
            "public_value": "dataspace",
        })

        while True:
            await asyncio.sleep(10)
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        await asyncio.sleep(0.05)
    finally:
        await facade.stop()

    private_path = data_dir / "private" / "foo.json"
    dataspace_path = data_dir / "dataspace" / "foo.json"

    assert private_path.exists()
    assert dataspace_path.exists()

    assert json.loads(
        private_path.read_text(encoding="utf-8")
    ) == {
        "private_value": "private",
    }

    assert json.loads(
        dataspace_path.read_text(encoding="utf-8")
    ) == {
        "public_value": "dataspace",
    }

# ============================================================================
# Enrichment: error caching, revival, reload window, atomicity, builtins
# ============================================================================


async def _wait_for(predicate, timeout=3.0):
    import time as _t

    deadline = _t.monotonic() + timeout

    while _t.monotonic() < deadline:
        if predicate():
            return True

        await asyncio.sleep(0.02)

    return False


MODULE_IMPORT = "from src.nan_itself.modules import Module"


@pytest.mark.asyncio
async def test_unchanged_broken_module_is_not_retried_until_fixed(tmp_path):
    workspace = tmp_path / "ws"

    target = workspace / "flaky.py"

    workspace.mkdir(parents=True)

    target.write_text(
        "# @module\n\n"
        + MODULE_IMPORT
        + "\n\nclass Flaky(Module):\n    pass\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                target.resolve()
                in facade._workspace_load_errors
            )
        )

        assert ok

        await asyncio.sleep(0.08)

        # Unchanged: exactly one cached error, no retry churn.
        assert len(facade._workspace_load_errors) == 1
        assert "flaky" not in facade.modules

        import time as _t

        _t.sleep(0.01)

        target.write_text(
            "# @module\n\n"
            + MODULE_IMPORT
            + "\n\nclass Flaky(Module):\n"
            + "    id = \"flaky\"\n\n"
            + "    async def start(self):\n"
            + "        while True:\n"
            + "            await asyncio.sleep(10)\n",
            encoding="utf-8",
        )

        ok = await _wait_for(
            lambda: "flaky" in facade.modules
        )

        assert ok
        assert target.resolve() not in facade._workspace_load_errors

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_removed_module_leaves_live_dataspace_for_revival(tmp_path):
    workspace = tmp_path / "ws"

    target = workspace / "keep.py"

    workspace.mkdir(parents=True)

    target.write_text(
        "# @module\n\n"
        "import asyncio\n\n"
        + MODULE_IMPORT
        + "\n\nclass Keep(Module):\n"
        + "    id = \"keep\"\n\n"
        + "    async def start(self):\n"
        + "        self.data.publish({\"v\": 1})\n\n"
        + "        while True:\n"
        + "            await asyncio.sleep(10)\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.dataspaces["keep"].snapshot()
                == {"v": 1}
            )
        )

        assert ok

        retained = facade.dataspaces["keep"]

        target.unlink()

        ok = await _wait_for(
            lambda: "keep" not in facade.modules
        )

        assert ok
        assert facade.dataspaces["keep"] is retained
        assert retained.snapshot() == {"v": 1}

        import time as _t

        _t.sleep(0.01)

        target.write_text(
            "# @module\n\n"
            "import asyncio\n\n"
            + MODULE_IMPORT
            + "\n\nclass Keep(Module):\n"
            + "    id = \"keep\"\n\n"
            + "    async def start(self):\n"
            + "        self.data.publish({\"v\": 2})\n\n"
            + "        while True:\n"
            + "            await asyncio.sleep(10)\n",
            encoding="utf-8",
        )

        ok = await _wait_for(
            lambda: (
                facade.dataspaces["keep"].snapshot()
                == {"v": 2}
            )
        )

        assert ok
        assert facade.dataspaces["keep"] is retained

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_hot_reload_candidate_starts_before_old_stops(tmp_path):
    workspace = tmp_path / "ws"

    target = workspace / "swap.py"

    events_file = tmp_path / "events.txt"

    workspace.mkdir(parents=True)

    OLD_BODY = (
        "# @module\n\n"
        "import asyncio\n\n"
        + MODULE_IMPORT
        + "\n\nEVENTS = r\"" + str(events_file) + "\"\n\n"
        "class Swap(Module):\n"
        "    id = \"swap\"\n\n"
        "    async def start(self):\n"
        "        with open(EVENTS, \"a\") as fh:\n"
        "            fh.write(\"old-start\\n\")\n"
        "        while True:\n"
        "            await asyncio.sleep(0.05)\n\n"
        "    async def stop(self):\n"
        "        with open(EVENTS, \"a\") as fh:\n"
        "            fh.write(\"old-stop\\n\")\n"
    )

    NEW_BODY = OLD_BODY.replace("Swap(", "SwapTwo(")
    NEW_BODY = NEW_BODY.replace("old-start", "cand-start")
    NEW_BODY = NEW_BODY.replace("old-stop", "cand-stop")

    target.write_text(OLD_BODY, encoding="utf-8")

    facade = Facade(
        workspace_modules=workspace,
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                events_file.exists()
                and "old-start"
                in events_file.read_text().splitlines()
            )
        )

        assert ok

        import time as _t

        _t.sleep(0.01)

        target.write_text(NEW_BODY, encoding="utf-8")

        ok = await _wait_for(
            lambda: (
                facade.modules["swap"].generation >= 1
                and events_file.exists()
                and "cand-stop"
                not in events_file.read_text().splitlines()
                and False
            )
            or (
                facade.modules["swap"].generation >= 1
                and "old-stop"
                in events_file.read_text().splitlines()
            )
        )

        assert ok

        lines = events_file.read_text().splitlines()

        assert "cand-start" in lines

        assert lines.index("cand-start") < lines.index(
            "old-stop"
        )

    finally:
        await facade.stop()


def test_atomic_write_cleans_temp_file_on_failure(
    tmp_path,
    monkeypatch,
):
    from src.nan_itself.modules import (
        persistence as mp,
    )

    target = tmp_path / "x.json"

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(mp.os, "replace", boom)

    try:
        mp.atomic_write_json(target, {"a": 1})
    except OSError:
        pass

    leftovers = [
        item.name
        for item in tmp_path.iterdir()
        if item.name.endswith(".tmp")
    ]

    assert leftovers == []


@pytest.mark.asyncio
async def test_builtin_crash_is_retried(tmp_path):
    attempts = []

    class Crashy(Module):
        id = "crashy"

        async def start(self):
            attempts.append(len(attempts))

            if len(attempts) == 1:
                raise RuntimeError("first boot fails")

            while True:
                await asyncio.sleep(10)

    facade = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(Crashy,),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.modules["crashy"].state
                == ModuleState.RUNNING
                and len(attempts) >= 2
            )
        )

        assert ok

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_workspace_can_require_builtin(tmp_path):
    base_started: list[str] = []

    class Base(Module):
        id = "base"

        async def start(self):
            base_started.append("base")

            self.data.publish({"v": 7})

            while True:
                await asyncio.sleep(10)

    workspace = tmp_path / "ws"

    workspace.mkdir(parents=True)

    (workspace / "consumer.py").write_text(
        "# @module\n\n"
        "import asyncio\n\n"
        + MODULE_IMPORT
        + "\n\nclass Consumer(Module):\n"
        + "    id = \"consumer\"\n"
        + "    requires = (\"base\",)\n\n"
        + "    async def start(self):\n"
        + "        v = self.dependencies[\"base\"].snapshot()[\"v\"]\n"
        + "        self.data.publish({\"echo\": v})\n\n"
        + "        while True:\n"
        + "            await asyncio.sleep(10)\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(Base,),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.dataspaces["consumer"].snapshot()
                == {"echo": 7}
            )
        )

        # echo==7 is only possible if the builtin generation
        # published before the workspace consumer started.
        assert ok
        assert base_started == ["base"]

    finally:
        await facade.stop()


class CountingBuiltin(Module):
    id = "counter"

    def __init__(self):
        self.count = 0

    async def start(self):
        self.count += 1

        self.data.publish({"count": self.count})

        while True:
            await asyncio.sleep(10)

    def serialize_state(self):
        return {"count": self.count}

    def restore_state(self, state):
        self.count = state.get("count", 0)


@pytest.mark.asyncio
async def test_builtin_persistence_round_trip(tmp_path):
    data_dir = tmp_path / "data"

    first = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(CountingBuiltin,),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await first.start()

    try:
        ok = await _wait_for(
            lambda: (
                first.dataspaces["counter"].snapshot()
                == {"count": 1}
            )
        )

        assert ok

    finally:
        await first.stop()

    second = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(CountingBuiltin,),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await second.start()

    try:
        instance = second.modules["counter"].instance

        ok = await _wait_for(
            lambda: instance.count >= 2
        )

        assert ok
        assert second.dataspaces[
            "counter"
        ].snapshot() == {"count": instance.count}

    finally:
        await second.stop()


@pytest.mark.asyncio
async def test_builtin_query_reaches_ambient_context(tmp_path):
    class Noter(Module):
        id = "noter"

        async def start(self):
            self.data.publish({"note": "hello-from-noter"})

            while True:
                await asyncio.sleep(10)

        async def query(self, turn):
            note = self.data.snapshot()["note"]

            return f"NOTER:{note}"

    facade = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(Noter,),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: bool(
                facade.dataspaces["noter"].snapshot()
            )
        )

        assert ok

        results = await facade.query_snapshot(
            "turn-x",
            facade.snapshot(),
        )

        assert any(
            "NOTER:hello-from-noter" in item
            for item in results
        )

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_conflict_with_builtin_raises_typed_error(tmp_path):
    from src.nan_itself.modules import (
        DuplicateModuleError,
    )

    class Reserved(Module):
        id = "reserved"

        async def start(self):
            while True:
                await asyncio.sleep(10)

    workspace = tmp_path / "ws"

    workspace.mkdir(parents=True)

    (workspace / "reserved.py").write_text(
        "# @module\n\n"
        + MODULE_IMPORT
        + "\n\nclass Reserved(Module):\n"
        + "    id = \"reserved\"\n\n"
        + "    async def start(self):\n"
        + "        while True:\n"
        + "            await asyncio.sleep(10)\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        builtin_modules=(Reserved,),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                (workspace / "reserved.py").resolve()
                in facade._workspace_load_errors
            )
        )

        assert ok

        err = facade._workspace_load_errors[
            (workspace / "reserved.py").resolve()
        ]

        assert isinstance(err, DuplicateModuleError)
        assert "<builtin>" in str(err)
        assert facade.modules["reserved"].origin == "builtin"

    finally:
        await facade.stop()


# ============================================================================
# Round 2 enrichment: contracts, wiring, rejection paths, builtin chains
# ============================================================================


def test_dataspace_revision_counts_publications():
    space = DataSpace("rev-check")

    assert space.revision == 0

    space.publish({"a": 1})
    space.publish({"a": 2})

    assert space.revision == 2


def test_dataspace_reader_is_readonly_projection():
    space = DataSpace("owner-id")

    space.publish({"v": 5})

    reader = DataSpaceReader(space)

    assert reader.owner == "owner-id"
    assert reader.revision == 1
    assert reader.snapshot() == {"v": 5}

    # The whole point: readers cannot publish.
    assert not hasattr(reader, "publish")


def test_module_turn_delegates_to_inner_turn():
    class Probe:
        agent_hash = "abc123"
        depth = 2

    turn = ModuleTurn(
        turn=Probe(),
        data={"x": 1},
    )

    # Attribute access falls through to the inner turn...
    assert turn.agent_hash == "abc123"
    assert turn.depth == 2

    # ...while data stays the ModuleTurn's own field.
    assert turn.data == {"x": 1}


@pytest.mark.asyncio
async def test_query_skips_non_running_modules(tmp_path):
    class Sick(Module):
        id = "sick"

        async def start(self):
            raise RuntimeError("boot fail")

    class HQ(Module):
        id = "hq"

        async def start(self):
            self.data.publish({"on": True})

            while True:
                await asyncio.sleep(10)

        async def query(self, turn):
            return "HQ-OK"

    facade = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(Sick, HQ),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: "sick" in facade.modules
        )

        assert ok

        # Crash retries keep the record mostly DOWN (retry
        # interval 30ms); poll at 5ms to catch the stable window.
        saw_down = False

        for _ in range(600):
            if (
                facade.modules["sick"].state
                == ModuleState.DOWN
            ):
                saw_down = True
                break

            await asyncio.sleep(0.005)

        assert saw_down

        results = await facade.query_snapshot(
            "t",
            facade.snapshot(),
        )

        # The DOWN module is not queried at all.
        assert results == ["HQ-OK"]

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_late_arriving_dependency_gets_wired(tmp_path):
    workspace = tmp_path / "ws"

    workspace.mkdir(parents=True)

    (workspace / "consumer.py").write_text(
        "# @module\n\n"
        "import asyncio\n\n"
        "from src.nan_itself.modules import Module\n\n"
        "class Consumer(Module):\n"
        "    id = \"consumer\"\n"
        "    requires = (\"later\",)\n\n"
        "    async def start(self):\n"
        "        while True:\n"
        "            d = self.dependencies.get(\"later\")\n"
        "            v = d.snapshot()[\"v\"] if d else None\n"
        "            self.data.publish({\"seen\": v})\n"
        "            await asyncio.sleep(0.02)\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        # Phase 1: dependency missing, consumer survives with None.
        ok = await _wait_for(
            lambda: (
                facade.dataspaces["consumer"].snapshot()
                == {"seen": None}
            )
        )

        assert ok

        import time as _t

        _t.sleep(0.01)

        (workspace / "later.py").write_text(
            "# @module\n\n"
            "import asyncio\n\n"
            "from src.nan_itself.modules import Module\n\n"
            "class Later(Module):\n"
            "    id = \"later\"\n\n"
            "    async def start(self):\n"
            "        self.data.publish({\"v\": 9})\n\n"
            "        while True:\n"
            "            await asyncio.sleep(10)\n",
            encoding="utf-8",
        )

        # Phase 2: the next rebuild rebinds readers automatically.
        ok = await _wait_for(
            lambda: (
                facade.dataspaces["consumer"].snapshot()
                == {"seen": 9}
            )
        )

        assert ok

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_crash_captures_error_on_record(tmp_path):
    workspace = tmp_path / "ws"

    workspace.mkdir(parents=True)

    (workspace / "boom.py").write_text(
        "# @module\n\n"
        "from src.nan_itself.modules import Module\n\n"
        "class Boom(Module):\n"
        "    id = \"boom\"\n\n"
        "    async def start(self):\n"
        "        raise RuntimeError(\"explosion reason\")\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.modules["boom"].state
                == ModuleState.DOWN
            )
        )

        assert ok

        err = facade.modules["boom"].error

        assert isinstance(err, RuntimeError)
        assert "explosion reason" in str(err)

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_corrupt_dataspace_file_does_not_prevent_boot(tmp_path):
    data_dir = tmp_path / "data"

    dataspace_dir = data_dir / "dataspace"

    dataspace_dir.mkdir(parents=True)

    # Wrong JSON shape entirely.
    (dataspace_dir / "stub.json").write_text(
        "[\"not\", \"an object\"]",
        encoding="utf-8",
    )

    workspace = tmp_path / "ws"

    workspace.mkdir(parents=True)

    (workspace / "stub.py").write_text(
        "# @module\n\n"
        "import asyncio\n\n"
        "from src.nan_itself.modules import Module\n\n"
        "class Stub(Module):\n"
        "    id = \"stub\"\n\n"
        "    async def start(self):\n"
        "        self.data.publish({\"ok\": True})\n\n"
        "        while True:\n"
        "            await asyncio.sleep(10)\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.dataspaces["stub"].snapshot()
                == {"ok": True}
            )
        )

        assert ok

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_hot_reload_rejects_id_change(tmp_path):
    workspace = tmp_path / "ws"

    target = workspace / "stable.py"

    workspace.mkdir(parents=True)

    target.write_text(
        "# @module\n\n"
        "import asyncio\n\n"
        "from src.nan_itself.modules import Module\n\n"
        "class Stable(Module):\n"
        "    id = \"stable\"\n\n"
        "    async def start(self):\n"
        "        self.data.publish({\"gen\": \"v1\"})\n\n"
        "        while True:\n"
        "            await asyncio.sleep(10)\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.dataspaces["stable"].snapshot()
                == {"gen": "v1"}
            )
        )

        assert ok

        import time as _t

        _t.sleep(0.01)

        target.write_text(
            "# @module\n\n"
            "import asyncio\n\n"
            "from src.nan_itself.modules import Module\n\n"
            "class Renamed(Module):\n"
            "    id = \"renamed\"\n\n"
            "    async def start(self):\n"
            "        while True:\n"
            "            await asyncio.sleep(10)\n",
            encoding="utf-8",
        )

        ok = await _wait_for(
            lambda: (
                (target).resolve()
                in facade._workspace_load_errors
            )
        )

        assert ok

        err = facade._workspace_load_errors[
            target.resolve()
        ]

        assert "changed Module id" in str(err)

        # Old generation untouched, still serving v1.
        assert "stable" in facade.modules
        assert facade.modules["stable"].generation == 0
        assert facade.dataspaces["stable"].snapshot() == {
            "gen": "v1"
        }

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_hot_reload_serialize_failure_keeps_old_generation(tmp_path):
    workspace = tmp_path / "ws"

    target = workspace / "ser.py"

    workspace.mkdir(parents=True)

    target.write_text(
        "# @module\n\n"
        "import asyncio\n\n"
        "from src.nan_itself.modules import Module\n\n"
        "class Ser(Module):\n"
        "    id = \"ser\"\n\n"
        "    async def start(self):\n"
        "        self.data.publish({\"gen\": 1})\n\n"
        "        while True:\n"
        "            await asyncio.sleep(10)\n\n"
        "    def serialize_state(self):\n"
        "        raise RuntimeError(\"serialize-nope\")\n",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.dataspaces["ser"].snapshot()
                == {"gen": 1}
            )
        )

        assert ok

        import time as _t

        _t.sleep(0.01)

        target.write_text(
            "# @module\n\n"
            "import asyncio\n\n"
            "from src.nan_itself.modules import Module\n\n"
            "class Ser(Module):\n"
            "    id = \"ser\"\n\n"
            "    async def start(self):\n"
            "        self.data.publish({\"gen\": 2})\n\n"
            "        while True:\n"
            "            await asyncio.sleep(10)\n",
            encoding="utf-8",
        )

        ok = await _wait_for(
            lambda: (
                target.resolve()
                in facade._workspace_load_errors
            )
        )

        assert ok

        err = facade._workspace_load_errors[
            target.resolve()
        ]

        assert "serialize-nope" in str(err)

        # Rejection happened before any candidate work.
        assert facade.modules["ser"].generation == 0
        assert facade.dataspaces["ser"].snapshot() == {"gen": 1}

    finally:
        await facade.stop()


def test_package_exports_are_complete():
    import src.nan_itself.modules as m

    required = {
        "DataSpace",
        "DataSpaceReader",
        "MODULE_HEADER",
        "Module",
        "ModuleRecord",
        "ModuleState",
        "ModuleTurn",
        "DuplicateModuleError",
        "BUILTIN_MODULES",
        "Facade",
    }

    assert required <= set(m.__all__)


@pytest.mark.asyncio
async def test_builtin_pair_dependency_chain(tmp_path):
    chain: list[str] = []

    class Base(Module):
        id = "chain-base"

        async def start(self):
            chain.append("base")

            self.data.publish({"v": 3})

            while True:
                await asyncio.sleep(10)

    class Derived(Module):
        id = "chain-derived"
        requires = ("chain-base",)

        async def start(self):
            chain.append("derived")

            v = self.dependencies[
                "chain-base"
            ].snapshot()["v"]

            self.data.publish({"echo": v})

            while True:
                await asyncio.sleep(10)

    facade = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(Base, Derived),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.dataspaces[
                    "chain-derived"
                ].snapshot()
                == {"echo": 3}
            )
        )

        assert ok
        assert chain == ["base", "derived"]

    finally:
        await facade.stop()


@pytest.mark.asyncio
async def test_stop_persists_builtin_artifacts(tmp_path):
    class Keeper(Module):
        id = "keeper"

        def __init__(self):
            self.token = "token-1"

        async def start(self):
            self.data.publish({"token": self.token})

            while True:
                await asyncio.sleep(10)

        def serialize_state(self):
            return {"token": self.token}

    data_dir = tmp_path / "data"

    facade = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(Keeper,),
        data_dir=data_dir,
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    ok = await _wait_for(
        lambda: bool(
            facade.dataspaces["keeper"].snapshot()
        )
    )

    assert ok

    await facade.stop()

    assert (data_dir / "dataspace" / "keeper.json").is_file()
    assert (data_dir / "private" / "keeper.json").is_file()


# ============================================================================
# Module infrastructure: TurnRecord delivery + LLM provisioning
# ============================================================================


class _LlmProbe(Module):
    id = "llm-probe"

    def __init__(self):
        self.seen_llm = "unset"

    async def start(self):
        self.seen_llm = (
            "has-llm"
            if self.llm is not None
            else "no-llm"
        )

        while True:
            await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_facade_provisions_llm_to_modules(tmp_path):
    facade = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(_LlmProbe,),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
        llm="SENTINEL-LLM",
    )

    await facade.start()

    try:
        ok = await _wait_for(
            lambda: (
                facade.modules["llm-probe"].instance.seen_llm
                == "has-llm"
            )
        )

        assert ok

    finally:
        await facade.stop()


def test_facade_without_llm_leaves_none(tmp_path):
    facade = Facade(
        workspace_modules=tmp_path / "ws",
        data_dir=tmp_path / "data",
    )

    assert facade.llm is None


class _OrderA(Module):
    id = "order-a"

    def __init__(self):
        self.got: list = []

    async def start(self):
        while True:
            await asyncio.sleep(10)

    async def on_turn(self, record):
        self.got.append(("a", record))


class _OrderB(Module):
    id = "order-b"

    def __init__(self):
        self.got: list = []

    async def start(self):
        while True:
            await asyncio.sleep(10)

    async def on_turn(self, record):
        self.got.append(("b", record))


@pytest.mark.asyncio
async def test_deliver_broadcasts_to_all_modules(tmp_path):
    facade = Facade(
        workspace_modules=tmp_path / "ws",
        builtin_modules=(_OrderA, _OrderB),
        data_dir=tmp_path / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        record = TurnRecord(
            agent_hash="h",
            parent_hash=None,
            depth=0,
            task=None,
            user_input="u",
            world={},
            reply="r",
            error=None,
            started_at=1.0,
            ended_at=2.0,
        )

        facade.deliver_turn(record)

        ok = await _wait_for(
            lambda: (
                len(facade.modules["order-a"].instance.got)
                == 1
                and len(facade.modules["order-b"].instance.got)
                == 1
            )
        )

        assert ok

        # Same record object reached everyone.
        assert (
            facade.modules["order-a"].instance.got[0][1]
            is record
        )
        assert (
            facade.modules["order-b"].instance.got[0][1]
            is record
        )

    finally:
        await facade.stop()
