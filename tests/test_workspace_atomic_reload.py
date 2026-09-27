from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from nan_itself.modules.runtime import Facade
from nan_itself.modules.model import ModuleState
from nan_itself.skills.model import SkillMetadata
from nan_itself.skills.runtime import SkillRuntime
from nan_itself.tools import local as local_backend
from nan_itself.tools import mcp as mcp_backend
from nan_itself.tools.runtime import ProviderRuntime
from nan_itself.tools.spec import (
    PROVIDER_KIND_MCP,
    ProviderSpec,
)
from nan_itself.tools.provider import Provider
from nan_itself.utils import paths as _paths


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Helpers
# ============================================================================


class FakeStack:
    def __init__(self):
        self.owner_task = asyncio.current_task()
        self.close_task = None

    async def aclose(self):
        self.close_task = asyncio.current_task()


class FakeSession:
    def __init__(
        self,
        generation: str,
    ):
        self.generation = generation

        self.tools = [
            SimpleNamespace(
                name="echo",
                description="echo",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                        }
                    },
                    "required": [
                        "text"
                    ],
                    "additionalProperties": False,
                },
            )
        ]

    async def list_tools(self):
        return SimpleNamespace(
            tools=self.tools
        )

    async def call_tool(
        self,
        name,
        arguments,
    ):
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=(
                        f"{self.generation}:"
                        f"{arguments['text']}"
                    ),
                )
            ]
        )


def make_mcp_provider(
    spec: ProviderSpec,
    generation: str,
) -> Provider:
    session = FakeSession(
        generation
    )

    stack = FakeStack()

    return Provider(
        spec=spec,
        stack=stack,
        session=session,
        tools={
            tool.name: tool
            for tool in session.tools
        },
    )


# ============================================================================
# MCP parser
# ============================================================================


def test_workspace_mcp_is_strictly_one_file_one_provider():
    config = {
        "name": "github",
        "command": "fake-mcp",
    }

    specs = mcp_backend.parse_config(
        config,
        Path("/tmp/github.yaml"),
    )

    assert len(specs) == 1
    assert specs[0].name == "github"


def test_workspace_mcp_rejects_multi_provider_mapping():
    config = {
        "mcp_servers": {
            "github": {
                "command": "fake-github",
            },
            "playwright": {
                "command": "fake-playwright",
            },
        }
    }

    try:
        mcp_backend.parse_config(
            config,
            Path("/tmp/multi.yaml"),
        )
    except ValueError as exc:
        message = str(exc)
        assert "exactly one provider" in message
        assert "mcp_servers" in message
    else:
        raise AssertionError(
            "MCP accepted a multi-provider file"
        )


# ============================================================================
# MCP transient invalid file
# ============================================================================


def test_mcp_invalid_intermediate_file_keeps_old_provider(
    tmp_path,
    monkeypatch,
):
    """
    Simulate a typical editor write:

        valid v1
          ->
        half-written / invalid YAML
          ->
        valid v2

    The invalid intermediate state must not remove v1.

    Once the fingerprint changes again, v2 must be retried and installed.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    async def scenario():
        mcp_dir = tmp_path / "workspace" / "tools" / "mcps"
        mcp_dir.mkdir(parents=True, exist_ok=True)

        source = (
            mcp_dir / "example.yaml"
        )

        source.write_text(
            "\n".join(
                [
                    "name: example",
                    "command: fake-mcp",
                    "args:",
                    "  - v1",
                ]
            ),
            encoding="utf-8",
        )

        async def fake_connect(
            spec,
        ):
            return make_mcp_provider(
                spec,
                spec.args[0],
            )

        monkeypatch.setattr(
            mcp_backend,
            "connect",
            fake_connect,
        )

        runtime = ProviderRuntime()

        try:
            # ------------------------------------------------------
            # v1
            # ------------------------------------------------------

            await runtime._scan_mcps()

            old_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert old_provider is not None
            assert (
                old_provider.spec.args
                == ("v1",)
            )

            # ------------------------------------------------------
            # Broken intermediate file.
            # ------------------------------------------------------

            source.write_text(
                "name: example\ncommand:\n",
                encoding="utf-8",
            )

            await runtime._scan_mcps()

            # v1 must survive.
            current = (
                runtime.get_provider(
                    "example"
                )
            )

            assert (
                current
                is old_provider
            )

            # ------------------------------------------------------
            # Valid v2.
            # ------------------------------------------------------

            source.write_text(
                "\n".join(
                    [
                        "name: example",
                        "command: fake-mcp",
                        "args:",
                        "  - v2",
                    ]
                ),
                encoding="utf-8",
            )

            await runtime._scan_mcps()

            new_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert new_provider is not None

            assert (
                new_provider
                is not old_provider
            )

            assert (
                new_provider.spec.args
                == ("v2",)
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Local Tool transient invalid file
# ============================================================================


def test_local_tool_invalid_intermediate_file_keeps_old_provider(
    tmp_path,
    monkeypatch,
):
    """
    The Local Tool file may be temporarily syntactically invalid while
    an editor is writing it. The current provider must remain available
    until a valid replacement appears.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    async def scenario():
        local_dir = tmp_path / "workspace" / "tools" / "local"
        local_dir.mkdir(parents=True, exist_ok=True)

        source = (
            local_dir / "example.py"
        )

        source.write_text(
            """# @tool

class Provider(LocalToolProvider):
    id = "example"

    @tool
    def echo(self, text: str) -> str:
        return "v1:" + text
""",
            encoding="utf-8",
        )

        runtime = ProviderRuntime()

        try:
            await runtime._scan_locals()

            old_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert old_provider is not None

            source.write_text(
                """# @tool

class Provider(LocalToolProvider):
    id = "example"

    @tool
    def echo(self, text: str
""",
                encoding="utf-8",
            )

            await runtime._scan_locals()

            assert (
                runtime.get_provider(
                    "example"
                )
                is old_provider
            )

            source.write_text(
                """# @tool

class Provider(LocalToolProvider):
    id = "example"

    @tool
    def echo(self, text: str) -> str:
        return "v2:" + text
""",
                encoding="utf-8",
            )

            await runtime._scan_locals()

            new_provider = (
                runtime.get_provider(
                    "example"
                )
            )

            assert new_provider is not None

            assert (
                new_provider
                is not old_provider
            )

        finally:
            await runtime.stop()

    run(
        scenario()
    )


# ============================================================================
# Skill transient invalid file
# ============================================================================


def test_skill_invalid_intermediate_file_keeps_old_record(
    tmp_path,
    monkeypatch,
):
    """
    The old Skill remains registered while a changed SKILL.md is invalid.
    """

    import nan_itself.skills.registry as registry_module

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    root = (
        tmp_path / "workspace" / "skills"
    )

    skill_dir = (
        root / "example"
    )

    skill_dir.mkdir(
        parents=True
    )

    source = (
        skill_dir / "SKILL.md"
    )

    source.write_text(
        "v1",
        encoding="utf-8",
    )

    def fake_read_metadata(
        skill_root,
    ):
        value = (
            skill_root / "SKILL.md"
        ).read_text(
            encoding="utf-8"
        )

        if value == "INVALID":
            raise ValueError(
                "temporary invalid SKILL.md"
            )

        return SkillMetadata(
            name="example",
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
            "example"
        )
    )

    assert old_record is not None

    source.write_text(
        "INVALID",
        encoding="utf-8",
    )

    runtime.refresh()

    assert (
        runtime._registry.get_record(
            "example"
        )
        is old_record
    )

    source.write_text(
        "v2",
        encoding="utf-8",
    )

    runtime.refresh()

    new_record = (
        runtime._registry.get_record(
            "example"
        )
    )

    assert new_record is not None

    assert new_record is not old_record

    assert (
        new_record.generation
        == old_record.generation + 1
    )


# ============================================================================
# Module transient invalid file
# ============================================================================


def test_module_invalid_intermediate_file_keeps_old_generation(
    tmp_path,
    monkeypatch,
):
    """
    A Python Module source can be temporarily invalid during editing.

    Facade must keep the current generation until a valid source is seen.
    """

    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    async def scenario():
        workspace = (
            tmp_path / "workspace" / "modules"
        )

        workspace.mkdir(
            parents=True
        )

        source = (
            workspace / "example.py"
        )

        valid_v1 = """# @module
import asyncio

class Example(Module):
    id = "example"

    async def start(self):
        await asyncio.Event().wait()
"""

        valid_v2 = """# @module
import asyncio

class Example(Module):
    id = "example"

    async def start(self):
        await asyncio.Event().wait()

    async def ask(self, turn):
        return "v2"
"""

        source.write_text(
            valid_v1,
            encoding="utf-8",
        )

        facade = Facade()

        # Initial scan/register.
        await facade._scan_module_root(
            facade.workspace_modules
        )

        old = (
            facade._find_record_by_source(
                source
            )
        )

        assert old is not None
        assert old.generation == 0

        # Temporary incomplete Python source.
        source.write_text(
            """# @module

class Example(Module):
    id = "example"

    async def start(self)
""",
            encoding="utf-8",
        )

        await facade._scan_module_root(
            facade.workspace_modules
        )

        current = (
            facade._find_record_by_source(
                source
            )
        )

        # Broken candidate must not replace v1.
        assert (
            current
            is old
        )

        assert (
            current.generation
            == 0
        )

        # Restore valid source.
        source.write_text(
            valid_v2,
            encoding="utf-8",
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

        await facade._scan_module_root(
            facade.workspace_modules
        )

        new = (
            facade._find_record_by_source(
                source
            )
        )

        assert new is not None

        assert new is not old

        assert (
            new.generation
            == 1
        )

        # Cleanup candidate task.
        task = new.task

        if (
            task is not None
            and not task.done()
        ):
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

    run(
        scenario()
    )