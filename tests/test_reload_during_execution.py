from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.agent.core import CoreAgent
from nan_itself.agent.engine import StepEngine
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.tools import mcp as mcp_backend
from nan_itself.tools.provider import Provider
from nan_itself.tools.results import text_result
from nan_itself.tools.runtime import ProviderRuntime
from nan_itself.utils.llm import ToolCall


# ============================================================================
# Generic helpers
# ============================================================================


def make_response(
    *,
    content=None,
    tool_calls=None,
):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        model="fake-model",
        usage=None,
        provider="fake-provider",
        finish_reason="stop",
    )


def make_tool_call(
    call_id,
    name,
    arguments,
):
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
    )


async def wait_until(
    predicate,
    *,
    timeout=2.0,
    interval=0.005,
):
    deadline = (
        asyncio.get_running_loop().time()
        + timeout
    )

    while not predicate():
        if (
            asyncio.get_running_loop().time()
            >= deadline
        ):
            raise AssertionError(
                "Condition was not satisfied before timeout."
            )

        await asyncio.sleep(
            interval
        )


def request_text(
    request,
):
    return "\n".join(
        message.content
        for message in request.messages
        if isinstance(
            getattr(
                message,
                "content",
                None,
            ),
            str,
        )
    )


# ============================================================================
# Minimal Agent infrastructure
# ============================================================================


class EmptyProviders:
    def provider_names(self):
        return ()

    def get_provider(
        self,
        name,
    ):
        return None


class EmptySkills:
    def refresh(self):
        pass

    def catalog(self):
        return ()

    def names(self):
        return ()

    def activate(
        self,
        name,
    ):
        raise AssertionError(
            "Unexpected Skill activation."
        )


class QueueLLM:
    """
    Deterministic async LLM.

    `block_first=True` makes the first LLM request wait until
    `release_first` is set. This gives the test a deterministic
    "active execution" window in which a resource can be reloaded.
    """

    model = "fake-model"
    provider = "fake-provider"

    def __init__(
        self,
        responses,
        *,
        block_first=False,
    ):
        self.responses = list(
            responses
        )

        self.requests = []

        self.block_first = (
            block_first
        )

        self.first_request_started = (
            asyncio.Event()
        )

        self.release_first = (
            asyncio.Event()
        )

    async def generate_complete(
        self,
        request,
    ):
        self.requests.append(
            request
        )

        if (
            self.block_first
            and len(self.requests)
            == 1
        ):
            self.first_request_started.set()

            await self.release_first.wait()

        if not self.responses:
            raise AssertionError(
                "LLM received more requests than expected."
            )

        await asyncio.sleep(0)

        return self.responses.pop(0)


# ============================================================================
# Module reload during an active CoreAgent turn
# ============================================================================


MODULE_V1 = """
# @module

import asyncio


class ExampleModule(Module):
    id = "example"
    requires = ()

    async def start(self):
        self.data.publish(
            {
                "version": 1,
            }
        )

        await asyncio.sleep(3600)

    async def query(
        self,
        turn,
    ):
        return (
            "module-version:"
            f"{turn.data['example']['version']}"
        )
"""


MODULE_V2 = """
# @module

import asyncio


class ExampleModule(Module):
    id = "example"
    requires = ()

    async def start(self):
        self.data.publish(
            {
                "version": 2,
            }
        )

        await asyncio.sleep(3600)

    async def query(
        self,
        turn,
    ):
        return (
            "module-version:"
            f"{turn.data['example']['version']}"
        )
"""


@pytest.mark.asyncio
async def test_module_reload_during_active_core_turn_keeps_snapshot_generation_isolated(
    tmp_path,
):
    """
    Verify the actual turn-consistency contract:

        Turn 1 captures Module/DataSpace world v1
            ->
        Turn 1 is blocked inside the LLM
            ->
        Module hot reload creates generation 2
            ->
        Turn 1 completes using its already-captured v1 snapshot
            ->
        Turn 2 captures the new world and sees v2

    The test records the real Facade.query_snapshot() calls instead
    of assuming Facade has test-double fields.
    """

    workspace = (
        tmp_path / "modules"
    )

    workspace.mkdir()

    source = (
        workspace / "example.py"
    )

    source.write_text(
        MODULE_V1,
        encoding="utf-8",
    )

    modules = __import__(
        "nan_itself.modules",
        fromlist=["Facade"],
    ).Facade(
        workspace_modules=workspace,
        builtin_modules=(),
        data_dir=tmp_path / "data",
        scan_interval=60.0,
        retry_interval=60.0,
    )

    providers = EmptyProviders()
    skills = EmptySkills()

    persona = {
        "value": "test persona",
    }

    llm = QueueLLM(
        [
            make_response(
                content="turn one complete"
            ),
            make_response(
                content="turn two complete"
            ),
        ],
        block_first=True,
    )

    core = CoreAgent(
        llm=llm,
        modules=modules,
        providers=providers,
        skills=skills,
        persona_source=lambda: persona[
            "value"
        ],
        max_subagent_depth=3,
    )

    # ------------------------------------------------------------------
    # Track calls to the REAL Facade.query_snapshot().
    # ------------------------------------------------------------------

    query_calls = []

    original_query_snapshot = (
        modules.query_snapshot
    )

    async def tracked_query_snapshot(
        turn,
        snapshot,
        **kwargs,
    ):
        query_calls.append(
            (
                turn,
                snapshot,
            )
        )

        return await original_query_snapshot(
            turn,
            snapshot,
            **kwargs,
        )

    modules.query_snapshot = (
        tracked_query_snapshot
    )

    try:
        # --------------------------------------------------------------
        # Initial Module generation.
        # --------------------------------------------------------------

        await modules.start()

        await wait_until(
            lambda: (
                "example"
                in modules.modules
                and modules.modules[
                    "example"
                ].data.snapshot().get(
                    "version"
                )
                == 1
            )
        )

        old_record = (
            modules.modules["example"]
        )

        old_generation = (
            old_record.generation
        )

        old_dataspace = (
            old_record.data
        )

        # --------------------------------------------------------------
        # Start Turn 1.
        #
        # CoreAgent:
        #
        #   skills.refresh()
        #   persona_source()
        #   modules.snapshot()
        #   engine.execute()
        #
        # Then the fake LLM blocks.
        # --------------------------------------------------------------

        first_task = asyncio.create_task(
            core.run(
                "first turn"
            ),
            name="test-core-turn-1",
        )

        await asyncio.wait_for(
            llm.first_request_started.wait(),
            timeout=1.0,
        )

        # The Module query has already happened before the first LLM
        # request is blocked.
        assert (
            len(query_calls)
            == 1
        )

        first_turn, first_snapshot = (
            query_calls[0]
        )

        assert (
            first_snapshot[
                "example"
            ]["version"]
            == 1
        )

        assert (
            first_turn.data[
                "example"
            ]["version"]
            == 1
        )

        # The Module's own query projection was based on v1.
        assert (
            "module-version:1"
            in request_text(
                llm.requests[0]
            )
        )

        # --------------------------------------------------------------
        # Reload while Turn 1 is still blocked.
        # --------------------------------------------------------------

        source.write_text(
            MODULE_V2,
            encoding="utf-8",
        )

        await asyncio.sleep(
            0
        )

        new_fingerprint = (
            modules._fingerprint(
                source
            )
        )

        await asyncio.wait_for(
            modules._load_or_reload_workspace_file(
                source,
                new_fingerprint,
            ),
            timeout=1.0,
        )

        new_record = (
            modules.modules["example"]
        )

        assert (
            new_record
            is not old_record
        )

        assert (
            new_record.generation
            == old_generation + 1
        )

        # DataSpace is owned by the Module id, not its generation.
        assert (
            new_record.data
            is old_dataspace
        )

        # Give the replacement generation one scheduler turn.
        await asyncio.sleep(
            0
        )

        # --------------------------------------------------------------
        # Critical invariant:
        #
        # Turn 1's already captured snapshot cannot change.
        # --------------------------------------------------------------

        assert (
            first_snapshot[
                "example"
            ]["version"]
            == 1
        )

        assert (
            first_turn.data[
                "example"
            ]["version"]
            == 1
        )

        assert (
            "module-version:1"
            in request_text(
                llm.requests[0]
            )
        )

        # --------------------------------------------------------------
        # Finish Turn 1.
        # --------------------------------------------------------------

        llm.release_first.set()

        first_result = (
            await asyncio.wait_for(
                first_task,
                timeout=1.0,
            )
        )

        assert (
            first_result.content
            == "turn one complete"
        )

        # --------------------------------------------------------------
        # The live runtime should now expose v2.
        # --------------------------------------------------------------

        await wait_until(
            lambda: (
                modules.snapshot()[
                    "example"
                ].get(
                    "version"
                )
                == 2
            )
        )

        # --------------------------------------------------------------
        # Start Turn 2.
        #
        # This must create a completely new world snapshot.
        # --------------------------------------------------------------

        second_result = (
            await asyncio.wait_for(
                core.run(
                    "second turn"
                ),
                timeout=1.0,
            )
        )

        assert (
            second_result.content
            == "turn two complete"
        )

        # There must now have been exactly two Module query calls:
        #
        #   query_calls[0] -> Turn 1 / v1
        #   query_calls[1] -> Turn 2 / v2
        #
        assert (
            len(query_calls)
            == 2
        )

        second_turn, second_snapshot = (
            query_calls[1]
        )

        assert (
            second_snapshot[
                "example"
            ]["version"]
            == 2
        )

        assert (
            second_turn.data[
                "example"
            ]["version"]
            == 2
        )

        # The new LLM request was generated from v2.
        assert (
            "module-version:2"
            in request_text(
                llm.requests[1]
            )
        )

        # --------------------------------------------------------------
        # Final turn consistency check.
        # --------------------------------------------------------------

        assert (
            query_calls[0][1][
                "example"
            ]["version"]
            == 1
        )

        assert (
            query_calls[1][1][
                "example"
            ]["version"]
            == 2
        )

        assert (
            len(llm.requests)
            == 2
        )

    finally:
        llm.release_first.set()

        await modules.stop()

        await core.agent_runtime.shutdown()


# ============================================================================
# Skill refresh during an active Subagent execution
# ============================================================================


SKILL_V1 = """---
name: demo
description: Demo Skill
---

instructions-v1
"""


SKILL_V2 = """---
name: demo
description: Demo Skill
---

instructions-v2
"""


@pytest.mark.asyncio
async def test_skill_refresh_during_active_subagent_keeps_current_generation_immutable(
    tmp_path,
):
    """
    A loaded Skill is an immutable execution value.

    During an active Subagent:

        current execution Skill == v1

    The workspace Skill is refreshed:

        registry generation == v2

    The running execution must continue holding the original Skill object.

    A newly activated Skill must contain v2.
    """

    skills_root = (
        tmp_path / "skills"
    )

    skill_dir = (
        skills_root / "demo"
    )

    skill_dir.mkdir(
        parents=True
    )

    skill_file = (
        skill_dir / "SKILL.md"
    )

    skill_file.write_text(
        SKILL_V1,
        encoding="utf-8",
    )

    from nan_itself.skills import (
        SkillRuntime,
    )

    skills = SkillRuntime(
        workspace_skills=skills_root,
        builtin_skills=(),
    )

    skills.discover()

    skill_v1 = skills.activate(
        "demo"
    )

    assert (
        skill_v1.instructions
        == "instructions-v1"
    )

    runtime = AgentRuntime(
        max_subagent_depth=3
    )

    providers = EmptyProviders()

    class EmptyModules:
        async def query_snapshot(
            self,
            turn,
            snapshot,
            **kwargs,
        ):
            return []

        def deliver_turn(
            self,
            record,
        ):
            pass

    llm = QueueLLM(
        [
            make_response(
                content="first skill turn"
            ),
            make_response(
                content="second skill turn"
            ),
        ],
        block_first=True,
    )

    engine = StepEngine(
        llm=llm,
        modules=EmptyModules(),
        providers=providers,
        skills=skills,
        agent_runtime=runtime,
    )

    root = runtime.create_root(
        world={},
        task="root",
    )

    async def run_child(
        skill,
        *,
        user_input,
    ):
        async def worker(
            context,
        ):
            return await engine.execute(
                context=context,
                user_input=user_input,
                persona="test",
            )

        handle = runtime.dispatch(
            root,
            task=user_input,
            worker=worker,
            skill=skill,
        )

        return await handle.wait()

    try:
        # ----------------------------------------------------------
        # First execution uses v1.
        # ----------------------------------------------------------

        first_task = asyncio.create_task(
            run_child(
                skill_v1,
                user_input="first skill turn",
            )
        )

        await asyncio.wait_for(
            llm.first_request_started.wait(),
            timeout=1.0,
        )

        first_request = (
            llm.requests[0]
        )

        first_text = request_text(
            first_request
        )

        assert (
            "instructions-v1"
            in first_text
        )

        # Keep a direct reference to the active execution Skill.
        active_skill = skill_v1

        # ----------------------------------------------------------
        # Refresh workspace Skill while the execution is active.
        # ----------------------------------------------------------

        await asyncio.sleep(
            0.01
        )

        skill_file.write_text(
            SKILL_V2,
            encoding="utf-8",
        )

        skills.refresh()

        skill_v2 = skills.activate(
            "demo"
        )

        assert (
            skill_v2.generation
            > skill_v1.generation
        )

        assert (
            skill_v2.instructions
            == "instructions-v2"
        )

        # The old immutable execution value did not mutate.
        assert (
            active_skill.instructions
            == "instructions-v1"
        )

        assert (
            active_skill.generation
            == skill_v1.generation
        )

        # ----------------------------------------------------------
        # Finish the original Subagent execution.
        # ----------------------------------------------------------

        llm.release_first.set()

        first_result = (
            await asyncio.wait_for(
                first_task,
                timeout=2.0,
            )
        )

        assert (
            first_result.content
            == "first skill turn"
        )

        # Its already-built prompt still contains v1.
        assert (
            "instructions-v1"
            in request_text(
                llm.requests[0]
            )
        )

        # ----------------------------------------------------------
        # New Subagent execution receives refreshed Skill v2.
        # ----------------------------------------------------------

        second_result = await run_child(
            skill_v2,
            user_input="second skill turn",
        )

        assert (
            second_result.content
            == "second skill turn"
        )

        assert (
            len(llm.requests)
            == 2
        )

        second_text = request_text(
            llm.requests[1]
        )

        assert (
            "instructions-v2"
            in second_text
        )

        assert (
            "instructions-v1"
            not in second_text
        )

    finally:
        llm.release_first.set()

        await runtime.shutdown()


# ============================================================================
# Local Tool reload during an in-flight call
# ============================================================================


LOCAL_TOOL_V1_TEMPLATE = """
# @tool

import asyncio
from pathlib import Path


class Calculator(LocalToolProvider):
    id = "calc"

    @tool
    async def slow(self) -> str:
        Path(r"{marker}").write_text(
            "started",
            encoding="utf-8",
        )

        await asyncio.sleep(0.2)

        return "v1"
"""


LOCAL_TOOL_V2 = """
# @tool

class Calculator(LocalToolProvider):
    id = "calc"

    @tool
    async def slow(self) -> str:
        return "v2"
"""


@pytest.mark.asyncio
async def test_local_tool_reload_keeps_inflight_call_on_old_provider(
    tmp_path,
):
    """
    An in-flight Local Tool call is bound to the Provider instance that
    existed when the call started.

    Reloading the workspace file replaces the live provider.

    Therefore:

        old in-flight call -> v1
        new call           -> v2
    """

    source = (
        tmp_path / "calc.py"
    )

    marker = (
        tmp_path / "started.txt"
    )

    source.write_text(
        LOCAL_TOOL_V1_TEMPLATE.format(
            marker=str(
                marker
            )
        ),
        encoding="utf-8",
    )

    runtime = ProviderRuntime(
        builtin_config_path=(
            tmp_path / "missing.yaml"
        ),
        workspace_mcp_dir=(
            tmp_path / "mcps"
        ),
        workspace_local_dir=tmp_path,
        builtin_tools=(),
        scan_interval=60.0,
        tool_timeout=2.0,
    )

    try:
        await runtime.start()

        old_provider = (
            runtime.get_provider(
                "calc"
            )
        )

        assert (
            old_provider is not None
        )

        # ----------------------------------------------------------
        # Start in-flight old call.
        # ----------------------------------------------------------

        old_call = asyncio.create_task(
            runtime.call_tool(
                "calc",
                "slow",
                {},
            )
        )

        await wait_until(
            marker.exists
        )

        # ----------------------------------------------------------
        # Reload local tool while call is still running.
        # ----------------------------------------------------------

        source.write_text(
            LOCAL_TOOL_V2,
            encoding="utf-8",
        )

        await asyncio.sleep(
            0.01
        )

        fingerprint = (
            runtime._local_tracker.fingerprints.get(
                source.resolve()
            )
        )

        new_fingerprint = (
            runtime._local_tracker.fingerprints.get(
                source.resolve()
            )
        )

        assert (
            new_fingerprint
            == fingerprint
        )

        await runtime._reload_local_source(
            source,
            runtime._local_tracker.fingerprints.get(
                source.resolve(),
                (
                    source.stat().st_mtime_ns,
                    source.stat().st_size,
                ),
            ),
        )

        new_provider = (
            runtime.get_provider(
                "calc"
            )
        )

        assert (
            new_provider is not None
        )

        assert (
            new_provider
            is not old_provider
        )

        # ----------------------------------------------------------
        # Old in-flight call must still complete against v1.
        # ----------------------------------------------------------

        old_result = (
            await asyncio.wait_for(
                old_call,
                timeout=2.0,
            )
        )

        assert (
            old_result.isError
            is False
        )

        assert (
            old_result.content[0].text
            == "v1"
        )

        # ----------------------------------------------------------
        # New call must use the replacement provider.
        # ----------------------------------------------------------

        new_result = (
            await runtime.call_tool(
                "calc",
                "slow",
                {},
            )
        )

        assert (
            new_result.isError
            is False
        )

        assert (
            new_result.content[0].text
            == "v2"
        )

    finally:
        await runtime.stop()


# ============================================================================
# MCP reload during an in-flight call
# ============================================================================


class FakeMCPStack:
    def __init__(self):
        self.closed = False
        self.closed_by = None

    async def aclose(self):
        self.closed = True
        self.closed_by = (
            asyncio.current_task()
        )


class InflightMCPSession:
    def __init__(
        self,
        *,
        value,
        started=None,
        release=None,
    ):
        self.value = value
        self.started = started
        self.release = release

        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="lookup",
                    description="Lookup",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "key",
                        ],
                        "additionalProperties": False,
                    },
                )
            ]
        )

    async def call_tool(
        self,
        name,
        arguments,
    ):
        self.calls.append(
            (
                name,
                dict(arguments),
            )
        )

        if self.started is not None:
            self.started.set()

        if self.release is not None:
            await self.release.wait()

        return text_result(
            f"{self.value}:{arguments['key']}"
        )


def write_mcp_source(
    path: Path,
):
    path.write_text(
        """
name: example
command: fake-mcp
""",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_mcp_reload_keeps_inflight_call_on_old_provider_and_new_calls_on_new_provider(
    tmp_path,
    monkeypatch,
):
    """
    MCP reload must install the new Provider without destroying an
    in-flight call on the old Provider.

    The old worker is allowed to remain alive until its active call
    drains.

    Therefore during reload:

        runtime provider mapping -> NEW

        OLD call                  -> still running

    After release:

        old call -> old result
        new call -> new result
    """

    workspace = (
        tmp_path / "mcps"
    )

    workspace.mkdir()

    source = (
        workspace / "example.yaml"
    )

    write_mcp_source(
        source
    )

    old_started = (
        asyncio.Event()
    )

    old_release = (
        asyncio.Event()
    )

    sessions = []

    async def fake_connect(
        spec,
    ):
        generation = len(
            sessions
        ) + 1

        if generation == 1:
            session = InflightMCPSession(
                value="old",
                started=old_started,
                release=old_release,
            )

        else:
            session = InflightMCPSession(
                value="new",
            )

        sessions.append(
            session
        )

        result = (
            await session.list_tools()
        )

        return Provider(
            spec=spec,
            stack=FakeMCPStack(),
            session=session,
            tools={
                tool.name: tool
                for tool in result.tools
            },
        )

    monkeypatch.setattr(
        mcp_backend,
        "connect",
        fake_connect,
    )

    runtime = ProviderRuntime(
        builtin_config_path=(
            tmp_path / "missing.yaml"
        ),
        workspace_mcp_dir=workspace,
        workspace_local_dir=(
            tmp_path / "local"
        ),
        builtin_tools=(),
        scan_interval=60.0,
        tool_timeout=2.0,
    )

    try:
        await runtime.start()

        old_provider = (
            runtime.get_provider(
                "example"
            )
        )

        assert (
            old_provider is not None
        )

        # ----------------------------------------------------------
        # Start old in-flight call.
        # ----------------------------------------------------------

        old_call = asyncio.create_task(
            runtime.call_tool(
                "example",
                "lookup",
                {
                    "key": "x",
                },
            )
        )

        await asyncio.wait_for(
            old_started.wait(),
            timeout=1.0,
        )

        old_worker = (
            runtime._mcp_workers[
                "example"
            ]
        )

        assert (
            old_worker.active_calls
            == 1
        )

        # ----------------------------------------------------------
        # Start reload concurrently.
        # ----------------------------------------------------------

        reload_task = asyncio.create_task(
            runtime._reload_workspace_source(
                source,
                runtime._mcp_tracker.fingerprints.get(
                    source.resolve(),
                    (
                        source.stat().st_mtime_ns,
                        source.stat().st_size,
                    ),
                ),
            )
        )

        # The candidate must replace the live mapping before the old
        # worker is allowed to finish draining.
        await wait_until(
            lambda: (
                len(sessions)
                == 2
                and runtime.get_provider(
                    "example"
                )
                is not old_provider
            )
        )

        new_provider = (
            runtime.get_provider(
                "example"
            )
        )

        assert (
            new_provider is not None
        )

        assert (
            new_provider
            is not old_provider
        )

        # Reload cannot finish yet because old active_calls == 1.
        assert (
            not reload_task.done()
        )

        # ----------------------------------------------------------
        # New calls already use the new provider.
        # ----------------------------------------------------------

        new_result = (
            await runtime.call_tool(
                "example",
                "lookup",
                {
                    "key": "y",
                },
            )
        )

        assert (
            new_result.isError
            is False
        )

        assert (
            new_result.content[0].text
            == "new:y"
        )

        # ----------------------------------------------------------
        # Let old in-flight call drain.
        # ----------------------------------------------------------

        old_release.set()

        old_result = (
            await asyncio.wait_for(
                old_call,
                timeout=2.0,
            )
        )

        assert (
            old_result.isError
            is False
        )

        assert (
            old_result.content[0].text
            == "old:x"
        )

        await asyncio.wait_for(
            reload_task,
            timeout=2.0,
        )

        # ----------------------------------------------------------
        # Old worker is no longer the live worker.
        # ----------------------------------------------------------

        current_worker = (
            runtime._mcp_workers[
                "example"
            ]
        )

        assert (
            current_worker
            is not old_worker
        )

    finally:
        old_release.set()

        await runtime.stop()