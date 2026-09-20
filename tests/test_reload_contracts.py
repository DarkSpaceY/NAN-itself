from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.modules import deps as deps_module
from nan_itself.modules.runtime import Facade
from nan_itself.modules.model import (
    Module,
    ModuleState,
)
from nan_itself.modules.reload import (
    hot_reload,
)
from nan_itself.skills.model import (
    SkillMetadata,
)
from nan_itself.skills.runtime import (
    SkillRuntime,
)
from nan_itself.tools.runtime import (
    ProviderRuntime,
)


def run(coro):
    return asyncio.run(coro)


def result_text(result):
    return result.content[0].text


# ============================================================================
# Local Tool helpers
# ============================================================================


def _write_tool(
    path: Path,
    *,
    provider_id: str,
    value: str,
) -> None:
    path.write_text(
        f"""# @tool

class Provider(LocalToolProvider):
    id = {provider_id!r}

    @tool
    def echo(self, text: str) -> str:
        return {value!r} + ":" + text
""",
        encoding="utf-8",
    )


# ============================================================================
# Local Tool hot reload
# ============================================================================


def test_local_tool_reload_replaces_only_one_provider(
    tmp_path,
):
    tool_dir = (
        tmp_path / "tools"
    )

    tool_dir.mkdir()

    alpha = (
        tool_dir / "alpha.py"
    )

    beta = (
        tool_dir / "beta.py"
    )

    _write_tool(
        alpha,
        provider_id="alpha",
        value="v1",
    )

    _write_tool(
        beta,
        provider_id="beta",
        value="beta",
    )

    runtime = ProviderRuntime(
        workspace_local_dir=tool_dir,
    )

    run(
        runtime._scan_locals()
    )

    beta_before = (
        runtime.get_provider("beta")
    )

    first = run(
        runtime.call_tool(
            "alpha",
            "echo",
            {
                "text": "hello"
            },
        )
    )

    assert (
        result_text(first)
        == "v1:hello"
    )

    _write_tool(
        alpha,
        provider_id="alpha",
        value="v2",
    )

    run(
        runtime._scan_locals()
    )

    beta_after = (
        runtime.get_provider("beta")
    )

    # Reload alpha must not replace beta.
    assert (
        beta_after
        is beta_before
    )

    second = run(
        runtime.call_tool(
            "alpha",
            "echo",
            {
                "text": "hello"
            },
        )
    )

    assert (
        result_text(second)
        == "v2:hello"
    )


def test_local_tool_file_remains_one_to_one_when_provider_id_changes(
    tmp_path,
):
    tool_dir = (
        tmp_path / "tools"
    )

    tool_dir.mkdir()

    source = (
        tool_dir / "one.py"
    )

    _write_tool(
        source,
        provider_id="alpha",
        value="alpha",
    )

    runtime = ProviderRuntime(
        # Isolate the builtin root: builtin/tools/local now ships a
        # real provider ('search') and this test must only see its
        # own temporary source.
        builtin_tools_dir=(
            tmp_path / "builtin"
        ),
        workspace_local_dir=tool_dir,
    )

    run(
        runtime._scan_locals()
    )

    assert (
        runtime.get_provider("alpha")
        is not None
    )

    assert (
        runtime.get_provider("beta")
        is None
    )

    _write_tool(
        source,
        provider_id="beta",
        value="beta",
    )

    run(
        runtime._scan_locals()
    )

    # One file owns one provider only.
    assert (
        runtime.get_provider("alpha")
        is None
    )

    assert (
        runtime.get_provider("beta")
        is not None
    )

    assert (
        runtime.provider_names()
        == ("beta",)
    )


def test_removed_local_tool_disappears_completely(
    tmp_path,
):
    tool_dir = (
        tmp_path / "tools"
    )

    tool_dir.mkdir()

    source = (
        tool_dir / "one.py"
    )

    _write_tool(
        source,
        provider_id="alpha",
        value="alpha",
    )

    runtime = ProviderRuntime(
        workspace_local_dir=tool_dir,
    )

    run(
        runtime._scan_locals()
    )

    assert (
        runtime.get_provider("alpha")
        is not None
    )

    source.unlink()

    run(
        runtime._scan_locals()
    )

    assert (
        runtime.get_provider("alpha")
        is None
    )


# ============================================================================
# Skill hot reload
# ============================================================================


def test_skill_refresh_isolated_and_does_not_leave_stale_name(
    tmp_path,
    monkeypatch,
):
    import nan_itself.skills.registry as registry_module

    root = (
        tmp_path / "skills"
    )

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

    (alpha / "SKILL.md").write_text(
        "alpha",
        encoding="utf-8",
    )

    (beta / "SKILL.md").write_text(
        "beta",
        encoding="utf-8",
    )

    def fake_read_metadata(
        skill_root,
    ):
        name = (
            skill_root / "SKILL.md"
        ).read_text(
            encoding="utf-8"
        ).strip()

        return SkillMetadata(
            name=name,
            description=name,
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

    runtime = SkillRuntime(
        workspace_skills=root,
        builtin_skills=tmp_path / "builtin-skills",
    )

    runtime.refresh()

    assert runtime.names() == (
        "alpha",
        "beta",
    )

    beta_before = (
        runtime._registry.get_record(
            "beta"
        )
    )

    assert beta_before is not None

    # The same workspace directory now represents a new Skill.
    (alpha / "SKILL.md").write_text(
        "alpha-v2",
        encoding="utf-8",
    )

    runtime.refresh()

    assert runtime.names() == (
        "alpha-v2",
        "beta",
    )

    # Strict 1:1 source binding:
    # the old Skill record must disappear.
    assert (
        runtime.get_metadata("alpha")
        is None
    )

    beta_after = (
        runtime._registry.get_record(
            "beta"
        )
    )

    # Reloading alpha must not mutate beta.
    assert (
        beta_after
        is beta_before
    )

    assert (
        beta_after.generation
        == 0
    )


def test_skill_reload_preserves_generation(
    tmp_path,
    monkeypatch,
):
    import nan_itself.skills.registry as registry_module

    root = (
        tmp_path / "skills"
    )

    alpha = (
        root / "alpha"
    )

    alpha.mkdir(
        parents=True
    )

    (alpha / "SKILL.md").write_text(
        "alpha",
        encoding="utf-8",
    )

    def fake_read_metadata(
        skill_root,
    ):
        return SkillMetadata(
            name="alpha",
            description="alpha",
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

    runtime = SkillRuntime(
        workspace_skills=root,
        builtin_skills=tmp_path / "builtin-skills",
    )

    runtime.refresh()

    first = (
        runtime._registry.get_record(
            "alpha"
        )
    )

    assert first is not None
    assert first.generation == 0

    (alpha / "SKILL.md").write_text(
        "alpha-v2",
        encoding="utf-8",
    )

    runtime.refresh()

    second = (
        runtime._registry.get_record(
            "alpha"
        )
    )

    assert second is not None
    assert second.generation == 1
    assert second is not first


def test_removed_skill_disappears_completely(
    tmp_path,
    monkeypatch,
):
    import nan_itself.skills.registry as registry_module

    root = (
        tmp_path / "skills"
    )

    alpha = (
        root / "alpha"
    )

    alpha.mkdir(
        parents=True
    )

    (alpha / "SKILL.md").write_text(
        "alpha",
        encoding="utf-8",
    )

    def fake_read_metadata(
        skill_root,
    ):
        return SkillMetadata(
            name="alpha",
            description="alpha",
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

    runtime = SkillRuntime(
        workspace_skills=root,
        builtin_skills=tmp_path / "builtin-skills",
    )

    runtime.refresh()

    assert (
        runtime.get_metadata("alpha")
        is not None
    )

    import shutil

    shutil.rmtree(alpha)

    runtime.refresh()

    assert (
        runtime.get_metadata("alpha")
        is None
    )

    assert (
        "alpha"
        not in runtime.names()
    )


# ============================================================================
# Module hot reload
# ============================================================================


def test_facade_workspace_module_reload_call_signature(
    tmp_path,
):
    """
    Facade -> hot_reload integration test.

    This test enters the actual reload path rather than calling
    hot_reload() directly.
    """

    workspace = (
        tmp_path / "modules"
    )

    workspace.mkdir(
        parents=True
    )

    source = (
        workspace / "example.py"
    )

    source.write_text(
        """# @module
import asyncio

class Example(Module):
    id = "example"

    async def start(self):
        await asyncio.Event().wait()
""",
        encoding="utf-8",
    )

    facade = Facade(
        workspace_modules=workspace,
        data_dir=(
            tmp_path / "data"
        ),
    )

    first_fp = facade._fingerprint(
        source
    )

    run(
        facade._load_or_reload_file(
            source,
            first_fp,
        )
    )

    first = facade._find_record_by_source(
        source
    )

    assert first is not None

    source.write_text(
        """# @module
import asyncio

class Example(Module):
    id = "example"

    async def start(self):
        await asyncio.Event().wait()

    async def query(self, turn):
        return "reloaded"
""",
        encoding="utf-8",
    )

    second_fp = facade._fingerprint(
        source
    )

    # This must enter hot_reload() successfully.
    run(
        facade._load_or_reload_file(
            source,
            second_fp,
        )
    )

    second = facade._find_record_by_source(
        source
    )

    assert second is not None
    assert second.generation == 1
    assert second.instance is not first.instance


def test_module_reload_does_not_rebind_unrelated_module(
    tmp_path,
    monkeypatch,
):
    """
    Strict independence contract.

    reload(A) must not rebind B.

    In particular B must preserve:
        - ModuleRecord identity
        - instance identity
        - task identity
        - generation
        - dependency mapping identity

    The final assertion additionally ensures that B's
    dependencies are not even rebound.
    """

    facade = Facade(
        workspace_modules=(
            tmp_path / "modules"
        ),
        data_dir=(
            tmp_path / "data"
        ),
    )

    a_source = (
        tmp_path / "a.py"
    )

    b_source = (
        tmp_path / "b.py"
    )

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
        requires = ("b",)

        def serialize_state(
            self,
        ):
            return {
                "counter": 7
            }

    class ANew(Module):
        id = "a"
        requires = ("b",)

        def restore_state(
            self,
            state,
        ):
            self.restored = state

    class B(Module):
        id = "b"

    old_a = (
        facade._register_module_class(
            AOld,
            source=str(a_source),
            source_fingerprint=(
                1,
                1,
            ),
            imported_module_name="old-a",
        )
    )

    b_record = (
        facade._register_module_class(
            B,
            source=str(b_source),
            source_fingerprint=(
                1,
                1,
            ),
            imported_module_name="old-b",
        )
    )

    facade._rebuild_dependency_graph()

    b_instance = (
        b_record.instance
    )

    b_task = object()

    b_record.task = b_task

    b_dependencies = (
        b_instance.dependencies
    )

    b_generation = (
        b_record.generation
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

    facade._run_module = (
        fake_run_module
    )

    run(
        hot_reload(
            facade,
            old=old_a,
            cls=ANew,
            imported_name="candidate-a",
            fingerprint=(
                2,
                2,
            ),
        )
    )

    new_a = facade.modules["a"]

    # ------------------------------------------------------------------
    # A changed exactly as expected.
    # ------------------------------------------------------------------

    assert (
        new_a.instance
        is not old_a.instance
    )

    assert (
        new_a.generation
        == old_a.generation + 1
    )

    assert (
        new_a.data
        is old_a.data
    )

    assert (
        facade.dataspaces["a"]
        is old_a.data
    )

    assert (
        new_a.instance.restored
        == {
            "counter": 7
        }
    )

    # ------------------------------------------------------------------
    # B must be completely untouched.
    # ------------------------------------------------------------------

    assert (
        facade.modules["b"]
        is b_record
    )

    assert (
        b_record.instance
        is b_instance
    )

    assert (
        b_record.task
        is b_task
    )

    assert (
        b_record.generation
        == b_generation
    )

    assert (
        b_instance.dependencies
        is b_dependencies
    )

    # ------------------------------------------------------------------
    # Strongest isolation assertion.
    #
    # hot_reload() must bind ONLY the candidate:
    # _rebuild_dependency_graph(bind=False) must not re-bind other
    # modules, so "b" never appears in bind_calls.
    # ------------------------------------------------------------------

    assert "b" not in bind_calls