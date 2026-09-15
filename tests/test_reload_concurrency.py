from __future__ import annotations

import asyncio
from pathlib import Path

from nan_itself.modules import deps as deps_module
from nan_itself.modules.runtime import Facade
from nan_itself.modules.model import (
    Module,
    ModuleState,
)
from nan_itself.modules.reload import (
    hot_reload,
)
from nan_itself.tools import local as local_backend
from nan_itself.tools.runtime import (
    ProviderRuntime,
)


def run(coro):
    return asyncio.run(coro)


def result_text(result):
    return result.content[0].text


# ============================================================================
# Module helpers
# ============================================================================


def _build_facade(tmp_path):
    return Facade(
        workspace_modules=tmp_path / "modules",
        data_dir=tmp_path / "data",
    )


def _install_module(
    facade,
    module_cls,
    source: Path,
):
    source.write_text(
        "# placeholder",
        encoding="utf-8",
    )

    record = facade._register_module_class(
        module_cls,
        source=str(source),
        source_fingerprint=(
            1,
            1,
        ),
        imported_module_name=(
            f"module-{module_cls.id}"
        ),
    )

    record.state = (
        ModuleState.RUNNING
    )

    return record


# ============================================================================
# Independent Module reloads
# ============================================================================


def test_concurrent_reload_of_independent_modules_isolated(
    tmp_path,
    monkeypatch,
):
    """
    A and B may reload concurrently.

    Each candidate may be bound exactly once.

    The important isolation property is not that bind_calls contains
    no "b" while B itself is reloading; rather, it is that each
    candidate is bound exactly once.

    If A's transaction accidentally re-binds B while B's own
    transaction is also running, B appears twice in bind_calls.
    """

    facade = _build_facade(
        tmp_path
    )

    a_source = (
        tmp_path / "a.py"
    )

    b_source = (
        tmp_path / "b.py"
    )

    class AOld(Module):
        id = "a"
        requires = ("b",)

        def serialize_state(self):
            return {
                "value": "a-old",
            }

    class ANew(Module):
        id = "a"
        requires = ("b",)

        def restore_state(
            self,
            state,
        ):
            self.restored = state

    class BOld(Module):
        id = "b"

        def serialize_state(self):
            return {
                "value": "b-old",
            }

    class BNew(Module):
        id = "b"

        def restore_state(
            self,
            state,
        ):
            self.restored = state

    old_a = _install_module(
        facade,
        AOld,
        a_source,
    )

    old_b = _install_module(
        facade,
        BOld,
        b_source,
    )

    facade._rebuild_dependency_graph(
        bind=False
    )

    bind_calls = []

    original_bind = (
        deps_module.bind_instance
    )

    def spy_bind(
        record,
        dataspaces,
    ):
        bind_calls.append(
            record.id
        )

        return original_bind(
            record,
            dataspaces,
        )

    monkeypatch.setattr(
        deps_module,
        "bind_instance",
        spy_bind,
    )

    async def fake_run_module(
        record,
        started=None,
    ):
        record.state = (
            ModuleState.RUNNING
        )

        if started is not None:
            started.set()

        await asyncio.Event().wait()

    facade._run_module = (
        fake_run_module
    )

    async def scenario():
        await asyncio.gather(
            hot_reload(
                facade,
                old=old_a,
                cls=ANew,
                imported_name="candidate-a",
                fingerprint=(
                    2,
                    2,
                ),
            ),
            hot_reload(
                facade,
                old=old_b,
                cls=BNew,
                imported_name="candidate-b",
                fingerprint=(
                    3,
                    3,
                ),
            ),
        )

        new_a = facade.modules["a"]
        new_b = facade.modules["b"]

        assert (
            new_a.instance
            is not old_a.instance
        )

        assert (
            new_b.instance
            is not old_b.instance
        )

        assert (
            new_a.generation
            == 1
        )

        assert (
            new_b.generation
            == 1
        )

        assert (
            new_a.instance.restored
            == {
                "value": "a-old",
            }
        )

        assert (
            new_b.instance.restored
            == {
                "value": "b-old",
            }
        )

        # Only A's transaction binds A and only B's transaction binds B.
        assert bind_calls.count("a") == 1
        assert bind_calls.count("b") == 1

    run(
        scenario()
    )


def test_concurrent_reload_same_module_does_not_stop_old_twice(
    tmp_path,
):
    """
    Two callers reload the same old generation concurrently.

    Exactly one transaction may commit.

    The second transaction becomes stale after the first transaction
    replaces facade.modules["example"] and must not stop the old
    generation a second time.
    """

    facade = _build_facade(
        tmp_path
    )

    source = (
        tmp_path / "module.py"
    )

    class OldModule(Module):
        id = "example"

        def __init__(self):
            self.stop_calls = 0

        def serialize_state(self):
            return {
                "value": "old",
            }

        async def stop(self):
            self.stop_calls += 1

    class NewModule(Module):
        id = "example"

        def restore_state(
            self,
            state,
        ):
            self.restored = state

    old = _install_module(
        facade,
        OldModule,
        source,
    )

    async def fake_run_module(
        record,
        started=None,
    ):
        record.state = (
            ModuleState.RUNNING
        )

        if started is not None:
            started.set()

        await asyncio.Event().wait()

    facade._run_module = (
        fake_run_module
    )

    async def scenario():
        await asyncio.gather(
            hot_reload(
                facade,
                old=old,
                cls=NewModule,
                imported_name="candidate-1",
                fingerprint=(
                    2,
                    2,
                ),
            ),
            hot_reload(
                facade,
                old=old,
                cls=NewModule,
                imported_name="candidate-2",
                fingerprint=(
                    3,
                    3,
                ),
            ),
        )

        active = facade.modules[
            "example"
        ]

        assert (
            active.generation
            == 1
        )

        # Same old generation may only be stopped once.
        assert (
            old.instance.stop_calls
            == 1
        )

        # One successful replacement only.
        assert (
            active.instance
            is not old.instance
        )

    run(
        scenario()
    )


def test_reload_candidate_becomes_active_before_old_stop_finishes(
    tmp_path,
):
    """
    Candidate-first / old-second transaction.

    A slow old.stop() must not prevent the new generation from
    becoming the active Module generation.
    """

    facade = _build_facade(
        tmp_path
    )

    source = (
        tmp_path / "module.py"
    )

    entered_stop = (
        asyncio.Event()
    )

    release_stop = (
        asyncio.Event()
    )

    class OldModule(Module):
        id = "example"

        def __init__(self):
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1

            entered_stop.set()

            await release_stop.wait()

    class NewModule(Module):
        id = "example"

    old = _install_module(
        facade,
        OldModule,
        source,
    )

    async def fake_run_module(
        record,
        started=None,
    ):
        record.state = (
            ModuleState.RUNNING
        )

        if started is not None:
            started.set()

        await asyncio.Event().wait()

    facade._run_module = (
        fake_run_module
    )

    async def scenario():
        reload_task = asyncio.create_task(
            hot_reload(
                facade,
                old=old,
                cls=NewModule,
                imported_name="candidate",
                fingerprint=(
                    2,
                    2,
                ),
            )
        )

        await entered_stop.wait()

        assert (
            old.instance.stop_calls
            == 1
        )

        # Candidate already owns the active slot.
        assert (
            facade.modules["example"]
            is not old
        )

        candidate = (
            facade.modules["example"]
        )

        assert (
            candidate.generation
            == old.generation + 1
        )

        release_stop.set()

        await reload_task

    run(
        scenario()
    )


# ============================================================================
# Local Tool generation boundary
# ============================================================================


def test_local_tool_inflight_call_finishes_on_old_provider_after_swap(
    tmp_path,
):
    """
    Existing Local Tool calls belong to the Provider generation
    looked up when the call began.

    Replacing runtime.providers["example"] must not redirect an
    in-flight call to the new Provider.

        old in-flight call -> old Provider
        new call           -> new Provider
    """

    entered = (
        asyncio.Event()
    )

    release = (
        asyncio.Event()
    )

    class OldProvider(
        local_backend.LocalToolProvider
    ):
        id = "example"

        @local_backend.tool
        async def echo(
            self,
            text: str,
        ) -> str:
            entered.set()

            await release.wait()

            return (
                "old:" + text
            )

    class NewProvider(
        local_backend.LocalToolProvider
    ):
        id = "example"

        @local_backend.tool
        async def echo(
            self,
            text: str,
        ) -> str:
            return (
                "new:" + text
            )

    runtime = ProviderRuntime(
        workspace_local_dir=(
            tmp_path / "tools"
        ),
    )

    old_provider = (
        local_backend.build_provider(
            local_backend.local_provider_spec(
                name="example",
                source="<old>",
            ),
            OldProvider,
        )
    )

    new_provider = (
        local_backend.build_provider(
            local_backend.local_provider_spec(
                name="example",
                source="<new>",
            ),
            NewProvider,
        )
    )

    runtime.providers[
        "example"
    ] = old_provider


    async def scenario():
        inflight = asyncio.create_task(
            runtime.call_tool(
                "example",
                "echo",
                {
                    "text": "hello",
                },
            )
        )

        await entered.wait()

        # Simulate the exact provider replacement performed by reload.
        runtime.providers[
            "example"
        ] = new_provider

        # Old call must remain on old provider.
        release.set()

        old_result = (
            await inflight
        )

        assert (
            result_text(old_result)
            == "old:hello"
        )

        # New call must use new provider.
        new_result = (
            await runtime.call_tool(
                "example",
                "echo",
                {
                    "text": "hello",
                },
            )
        )

        assert (
            result_text(new_result)
            == "new:hello"
        )

    run(
        scenario()
    )


def test_reloading_one_tool_provider_does_not_mutate_unrelated_provider(
    tmp_path,
):
    """
    Provider replacement for alpha must not replace or mutate beta.
    """

    class AlphaV1(
        local_backend.LocalToolProvider
    ):
        id = "alpha"

        @local_backend.tool
        def echo(
            self,
            text: str,
        ) -> str:
            return (
                "alpha-v1:" + text
            )

    class AlphaV2(
        local_backend.LocalToolProvider
    ):
        id = "alpha"

        @local_backend.tool
        def echo(
            self,
            text: str,
        ) -> str:
            return (
                "alpha-v2:" + text
            )

    class Beta(
        local_backend.LocalToolProvider
    ):
        id = "beta"

        @local_backend.tool
        def echo(
            self,
            text: str,
        ) -> str:
            return (
                "beta:" + text
            )

    runtime = ProviderRuntime(
        workspace_local_dir=(
            tmp_path / "tools"
        ),
    )

    alpha_v1 = (
        local_backend.build_provider(
            local_backend.local_provider_spec(
                name="alpha",
                source="<alpha-v1>",
            ),
            AlphaV1,
        )
    )

    alpha_v2 = (
        local_backend.build_provider(
            local_backend.local_provider_spec(
                name="alpha",
                source="<alpha-v2>",
            ),
            AlphaV2,
        )
    )

    beta = (
        local_backend.build_provider(
            local_backend.local_provider_spec(
                name="beta",
                source="<beta>",
            ),
            Beta,
        )
    )

    runtime.providers[
        "alpha"
    ] = alpha_v1

    runtime.providers[
        "beta"
    ] = beta

    beta_before = (
        runtime.get_provider(
            "beta"
        )
    )

    # Replace alpha only.
    runtime.providers[
        "alpha"
    ] = alpha_v2

    assert (
        runtime.get_provider(
            "alpha"
        )
        is alpha_v2
    )

    assert (
        runtime.get_provider(
            "alpha"
        )
        is not alpha_v1
    )

    assert (
        runtime.get_provider(
            "beta"
        )
        is beta_before
    )