from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.nan_itself.modules.facade import (
    DataSpace,
    Facade,
    Module,
    ModuleState,
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

from src.nan_itself.modules.facade import Module


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
from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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

from src.nan_itself.modules.facade import Module


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