from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from nan_itself.modules.runtime import Facade
from nan_itself.modules.model import (
    Module,
    ModuleState,
)
from nan_itself.modules.reload import hot_reload
from nan_itself.skills.model import SkillMetadata
from nan_itself.skills.runtime import SkillRuntime
from nan_itself.tools.runtime import ProviderRuntime
from nan_itself.utils import paths as _paths


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Helpers
# ============================================================================


def _write_tool(
    path: Path,
    *,
    provider_id: str,
    body: str,
) -> None:
    path.write_text(
        f"""# @tool

class Provider(ToolSet):
    id = {provider_id!r}

{body}
""",
        encoding="utf-8",
    )


def _write_valid_skill(
    path: Path,
    name: str,
) -> None:
    path.write_text(
        name,
        encoding="utf-8",
    )


# ============================================================================
# Module reload transaction
# ============================================================================


def test_module_long_running_candidate_can_commit_without_returning(
    tmp_path,
    monkeypatch,
):
    """
    Module.start() is a lifetime coroutine and normally never returns.

    Therefore hot reload must not wait for start() to finish.
    A live candidate task is sufficient to establish the RUNNING state.

    The important contract is that the candidate is alive before the
    old generation is stopped.
    """
    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    facade = Facade()

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    source = modules_dir / "module.py"
    source.write_text(
        "# placeholder",
        encoding="utf-8",
    )

    class OldModule(Module):
        id = "example"

        def __init__(self):
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1

    entered = asyncio.Event()
    release = asyncio.Event()

    class NewModule(Module):
        id = "example"

        async def start(self):
            entered.set()
            await release.wait()

    old = facade._register_module_class(
        OldModule,
        source=str(source),
        source_fingerprint=(1, 1),
        imported_module_name="old-module",
    )

    old.state = ModuleState.RUNNING

    async def fake_run_module(
        record,
        started=None,
    ):
        record.state = ModuleState.RUNNING

        if started is not None:
            started.set()

        try:
            await record.instance.start()

        except asyncio.CancelledError:
            raise

        finally:
            current_task = asyncio.current_task()

            if record.task is current_task:
                record.task = None

    facade._run_module = fake_run_module

    async def scenario():
        reload_task = asyncio.create_task(
            hot_reload(
                facade,
                old=old,
                cls=NewModule,
                imported_name="candidate-module",
                fingerprint=(2, 2),
            )
        )

        await entered.wait()

        # Candidate is alive.
        assert not reload_task.done()

        release.set()

        await reload_task

        assert facade.modules["example"] is not old
        assert facade.modules["example"].generation == 1

        # Old generation was the one replaced.
        assert old.instance.stop_calls == 1

    run(scenario())
def test_module_candidate_immediate_start_failure_keeps_old_generation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    facade = Facade()

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    source = modules_dir / "module.py"
    source.write_text(
        "# placeholder",
        encoding="utf-8",
    )

    class OldModule(Module):
        id = "example"

        def __init__(self):
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1

    class NewModule(Module):
        id = "example"

        async def start(self):
            raise RuntimeError(
                "candidate failed during startup"
            )

    old = facade._register_module_class(
        OldModule,
        source=str(source),
        source_fingerprint=(1, 1),
        imported_module_name="old-module",
    )

    old.state = ModuleState.RUNNING

    run(
        hot_reload(
            facade,
            old=old,
            cls=NewModule,
            imported_name="candidate-module",
            fingerprint=(2, 2),
        )
    )

    assert facade.modules["example"] is old

    assert old.state is (
        ModuleState.RUNNING
    )

    assert old.instance.stop_calls == 0

    assert (
        facade._source_load_errors.get(
            source.resolve()
        )
        is not None
    )

def test_module_candidate_start_failure_keeps_old_generation(
    tmp_path,
    monkeypatch,
):
    """
    If candidate.start() fails, hot reload must reject the candidate
    and leave the old generation installed and running.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    facade = Facade()

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    source = modules_dir / "module.py"
    source.write_text(
        "# placeholder",
        encoding="utf-8",
    )

    class OldModule(Module):
        id = "example"

        def __init__(self):
            self.stop_calls = 0

        async def stop(self):
            self.stop_calls += 1

    class NewModule(Module):
        id = "example"

        async def start(self):
            # Force one scheduler switch so the candidate first appears
            # to have entered start() and then fails.
            await asyncio.sleep(0)
            raise RuntimeError(
                "candidate failed during startup"
            )

    old = facade._register_module_class(
        OldModule,
        source=str(source),
        source_fingerprint=(1, 1),
        imported_module_name="old-module",
    )

    old.state = ModuleState.RUNNING

    async def scenario():
        await hot_reload(
            facade,
            old=old,
            cls=NewModule,
            imported_name="candidate-module",
            fingerprint=(2, 2),
        )

        # Give the candidate task a chance to finish.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert facade.modules["example"] is old

        assert old.state is (
            ModuleState.RUNNING
        )

        assert old.instance.stop_calls == 0

        assert (
            facade._source_load_errors.get(
                source.resolve()
            )
            is not None
        )

    run(scenario())


def test_module_reload_restore_failure_keeps_old_generation(
    tmp_path,
    monkeypatch,
):
    """
    restore_state() failure is rejected before the candidate enters
    the running lifecycle.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    facade = Facade()

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    source = modules_dir / "module.py"
    source.write_text(
        "# placeholder",
        encoding="utf-8",
    )

    class OldModule(Module):
        id = "example"

        def __init__(self):
            self.stop_calls = 0

        def serialize_state(self):
            return {
                "counter": 42,
            }

        async def stop(self):
            self.stop_calls += 1

    class NewModule(Module):
        id = "example"

        def restore_state(self, state):
            raise RuntimeError(
                "restore failed"
            )

    old = facade._register_module_class(
        OldModule,
        source=str(source),
        source_fingerprint=(1, 1),
        imported_module_name="old-module",
    )

    old.state = ModuleState.RUNNING

    run(
        hot_reload(
            facade,
            old=old,
            cls=NewModule,
            imported_name="candidate-module",
            fingerprint=(2, 2),
        )
    )

    assert facade.modules["example"] is old

    assert old.state is (
        ModuleState.RUNNING
    )

    assert old.instance.stop_calls == 0

    assert (
        facade._source_load_errors.get(
            source.resolve()
        )
        is not None
    )


def test_module_reload_dependency_cycle_keeps_old_generation(
    tmp_path,
    monkeypatch,
):
    """
    Changing A's requires relationship from:

        A -> nothing
        B -> A

    into:

        A -> B
        B -> A

    creates a cycle.

    The candidate must be rejected while B remains completely untouched.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    facade = Facade()

    modules_dir = tmp_path / "workspace" / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    a_source = modules_dir / "a.py"
    b_source = modules_dir / "b.py"

    a_source.write_text(
        "# placeholder",
        encoding="utf-8",
    )

    b_source.write_text(
        "# placeholder",
        encoding="utf-8",
    )

    class AOld(Module):
        id = "a"

        def serialize_state(self):
            return {
                "value": 1,
            }

    class ANew(Module):
        id = "a"
        requires = ("b",)

    class B(Module):
        id = "b"
        requires = ("a",)

    old_a = facade._register_module_class(
        AOld,
        source=str(a_source),
        source_fingerprint=(1, 1),
        imported_module_name="old-a",
    )

    b_record = facade._register_module_class(
        B,
        source=str(b_source),
        source_fingerprint=(1, 1),
        imported_module_name="old-b",
    )

    facade._rebuild_dependency_graph()

    b_record.state = ModuleState.RUNNING

    b_instance = b_record.instance
    b_dependencies = b_instance.dependencies

    async def fake_run_module(
        record,
        started=None,
    ):
        record.state = ModuleState.RUNNING

        if started is not None:
            started.set()

    facade._run_module = fake_run_module

    run(
        hot_reload(
            facade,
            old=old_a,
            cls=ANew,
            imported_name="candidate-a",
            fingerprint=(2, 2),
        )
    )

    assert facade.modules["a"] is old_a

    assert facade.modules["b"] is (
        b_record
    )

    assert b_record.instance is (
        b_instance
    )

    assert b_instance.dependencies is (
        b_dependencies
    )

    assert b_record.state is (
        ModuleState.RUNNING
    )


# ============================================================================
# Local Tool reload transaction
# ============================================================================


def test_local_tool_invalid_candidate_keeps_old_provider(
    tmp_path,
    monkeypatch,
):
    """
    A malformed replacement must not replace the current provider.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    tool_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    tool_dir.mkdir(
        parents=True
    )

    source = (
        tool_dir / "example.py"
    )

    _write_tool(
        source,
        provider_id="example",
        body="""
    @tool
    def echo(self, text: str) -> str:
        return "v1:" + text
""",
    )

    runtime = ProviderRuntime()

    run(
        runtime._scan_locals()
    )

    old_provider = runtime.get_provider(
        "example"
    )

    assert old_provider is not None

    # Invalid candidate: unannotated parameter.
    _write_tool(
        source,
        provider_id="example",
        body="""
    @tool
    def echo(self, text) -> str:
        return "broken:" + text
""",
    )

    run(
        runtime._scan_locals()
    )

    new_provider = runtime.get_provider(
        "example"
    )

    assert new_provider is (
        old_provider
    )


def test_local_tool_id_collision_keeps_old_provider(
    tmp_path,
    monkeypatch,
):
    """
    Two files are independently owned.

    Changing alpha.py so that it claims beta's provider id must fail
    without removing the existing alpha provider.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    tool_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    tool_dir.mkdir(
        parents=True
    )

    alpha = (
        tool_dir / "alpha.py"
    )

    beta = (
        tool_dir / "beta.py"
    )

    _write_tool(
        alpha,
        provider_id="alpha",
        body="""
    @tool
    def echo(self, text: str) -> str:
        return "alpha:" + text
""",
    )

    _write_tool(
        beta,
        provider_id="beta",
        body="""
    @tool
    def echo(self, text: str) -> str:
        return "beta:" + text
""",
    )

    runtime = ProviderRuntime()

    run(
        runtime._scan_locals()
    )

    alpha_before = (
        runtime.get_provider(
            "alpha"
        )
    )

    beta_before = (
        runtime.get_provider(
            "beta"
        )
    )

    assert alpha_before is not None
    assert beta_before is not None

    # alpha.py now tries to claim beta.
    _write_tool(
        alpha,
        provider_id="beta",
        body="""
    @tool
    def echo(self, text: str) -> str:
        return "collision:" + text
""",
    )

    run(
        runtime._scan_locals()
    )

    # Failed reload must preserve both existing providers.
    assert (
        runtime.get_provider("alpha")
        is alpha_before
    )

    assert (
        runtime.get_provider("beta")
        is beta_before
    )


def test_removed_local_tool_does_not_remove_unrelated_provider(
    tmp_path,
    monkeypatch,
):
    """
    Removing one provider file must only remove that provider.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    tool_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    tool_dir.mkdir(
        parents=True
    )

    alpha = (
        tool_dir / "alpha.py"
    )

    beta = (
        tool_dir / "beta.py"
    )

    _write_tool(
        alpha,
        provider_id="alpha",
        body="""
    @tool
    def echo(self, text: str) -> str:
        return "alpha:" + text
""",
    )

    _write_tool(
        beta,
        provider_id="beta",
        body="""
    @tool
    def echo(self, text: str) -> str:
        return "beta:" + text
""",
    )

    runtime = ProviderRuntime()

    run(
        runtime._scan_locals()
    )

    beta_before = (
        runtime.get_provider(
            "beta"
        )
    )

    assert beta_before is not None

    alpha.unlink()

    run(
        runtime._scan_locals()
    )

    assert (
        runtime.get_provider("alpha")
        is None
    )

    assert (
        runtime.get_provider("beta")
        is beta_before
    )


# ============================================================================
# Skill reload transaction
# ============================================================================


def test_skill_invalid_candidate_keeps_old_record(
    tmp_path,
    monkeypatch,
):
    """
    Invalid replacement content must leave the old Skill visible.
    """

    import nan_itself.skills.registry as registry_module

    root = (
        tmp_path / "workspace" / "skills"
    )

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    alpha = (
        root / "alpha"
    )

    alpha.mkdir(
        parents=True
    )

    _write_valid_skill(
        alpha / "SKILL.md",
        "alpha",
    )

    def fake_read_metadata(
        skill_root,
    ):
        value = (
            skill_root / "SKILL.md"
        ).read_text(
            encoding="utf-8"
        ).strip()

        if value == "INVALID":
            raise ValueError(
                "invalid skill"
            )

        return SkillMetadata(
            name=value,
            description=value,
            source=(
                skill_root / "SKILL.md"
            ),
            frontmatter={},
        )

    monkeypatch.setattr(
        registry_module,
        "read_metadata",
        fake_read_metadata,
    )

    runtime = SkillRuntime()

    runtime.refresh()

    old_record = (
        runtime._registry.get_record(
            "alpha"
        )
    )

    assert old_record is not None

    (alpha / "SKILL.md").write_text(
        "INVALID",
        encoding="utf-8",
    )

    runtime.refresh()

    assert (
        runtime._registry.get_record(
            "alpha"
        )
        is old_record
    )

    assert (
        runtime.get_metadata("alpha")
        is old_record.metadata
    )


def test_skill_name_collision_keeps_old_record(
    tmp_path,
    monkeypatch,
):
    """
    Strict 1:1 + transactional semantics:

        alpha/SKILL.md -> name=alpha
        beta/SKILL.md  -> name=beta

    Changing alpha's file to name=beta must be rejected.

    Neither Skill should disappear or be replaced.
    """

    import nan_itself.skills.registry as registry_module

    root = (
        tmp_path / "workspace" / "skills"
    )

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    alpha = (
        root / "alpha"
    )

    beta = (
        root / "beta"
    )

    alpha.mkdir(
        parents=True
    )

    beta.mkdir(
        parents=True
    )

    _write_valid_skill(
        alpha / "SKILL.md",
        "alpha",
    )

    _write_valid_skill(
        beta / "SKILL.md",
        "beta",
    )

    def fake_read_metadata(
        skill_root,
    ):
        value = (
            skill_root / "SKILL.md"
        ).read_text(
            encoding="utf-8"
        ).strip()

        return SkillMetadata(
            name=value,
            description=value,
            source=(
                skill_root / "SKILL.md"
            ),
            frontmatter={},
        )

    monkeypatch.setattr(
        registry_module,
        "read_metadata",
        fake_read_metadata,
    )

    runtime = SkillRuntime()

    runtime.refresh()

    alpha_before = (
        runtime._registry.get_record(
            "alpha"
        )
    )

    beta_before = (
        runtime._registry.get_record(
            "beta"
        )
    )

    assert alpha_before is not None
    assert beta_before is not None

    # Same file now claims another Skill's name.
    (alpha / "SKILL.md").write_text(
        "beta",
        encoding="utf-8",
    )

    runtime.refresh()

    # Transaction must reject the change and preserve both old records.
    assert (
        runtime._registry.get_record(
            "alpha"
        )
        is alpha_before
    )

    assert (
        runtime._registry.get_record(
            "beta"
        )
        is beta_before
    )

    assert runtime.names() == (
        "alpha",
        "beta",
    )


def test_skill_removed_file_does_not_touch_other_skill(
    tmp_path,
    monkeypatch,
):
    """
    Removing one Skill directory must not mutate unrelated Skills.
    """

    import nan_itself.skills.registry as registry_module

    root = (
        tmp_path / "workspace" / "skills"
    )

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    alpha = (
        root / "alpha"
    )

    beta = (
        root / "beta"
    )

    alpha.mkdir(
        parents=True
    )

    beta.mkdir(
        parents=True
    )

    _write_valid_skill(
        alpha / "SKILL.md",
        "alpha",
    )

    _write_valid_skill(
        beta / "SKILL.md",
        "beta",
    )

    def fake_read_metadata(
        skill_root,
    ):
        value = (
            skill_root / "SKILL.md"
        ).read_text(
            encoding="utf-8"
        ).strip()

        return SkillMetadata(
            name=value,
            description=value,
            source=(
                skill_root / "SKILL.md"
            ),
            frontmatter={},
        )

    monkeypatch.setattr(
        registry_module,
        "read_metadata",
        fake_read_metadata,
    )

    runtime = SkillRuntime()

    runtime.refresh()

    beta_before = (
        runtime._registry.get_record(
            "beta"
        )
    )

    assert beta_before is not None

    import shutil

    shutil.rmtree(alpha)

    runtime.refresh()

    assert (
        runtime.get_metadata("alpha")
        is None
    )

    assert (
        runtime._registry.get_record(
            "beta"
        )
        is beta_before
    )