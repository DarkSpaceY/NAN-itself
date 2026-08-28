"""
Round-2 enrichment for the agent package.

Locks mechanisms that survived the refactor untested: value-object
freezing, prompt assembly order, report truncation/consumption,
verb validation matrices, schema shapes, snapshot-once invariant,
subtree orphan dropping, tool-result serialization branches and
the loop's backoff rhythm.
"""

import asyncio
import time
from pathlib import Path

import pytest

from src.nan_itself.agent.engine import (
    StepEngine,
    _serialize_tool_result,
)
from src.nan_itself.agent.loop import (
    AgentLoop,
    Inbox,
)
from src.nan_itself.agent.model import (
    AgentResult,
    AgentTurn,
    ChildSubagent,
)
from src.nan_itself.agent.prompts import (
    build_messages,
    format_skill_section,
)
from src.nan_itself.agent.reports import (
    collect_finished_children,
)
from src.nan_itself.agent.role import (
    RolePolicy,
)
from src.nan_itself.agent.runtime import (
    AgentContext,
    AgentRuntime,
    SubagentLimitError,
)
from src.nan_itself.agent.verbs import (
    VERBS,
    ExecutionState,
)
from src.nan_itself.skills import (
    Skill,
    SkillMetadata,
    UnknownSkillError,
)
from src.nan_itself.tools import (
    ProviderRuntime,
)
from src.nan_itself.utils.llm import (
    Message,
)


def _skill(name: str) -> Skill:
    return Skill(
        metadata=SkillMetadata(
            name=name,
            description="d",
            source=Path("/tmp") / name / "SKILL.md",
            origin="workspace",
            frontmatter={"name": name},
        ),
        instructions=f"DO {name.upper()}",
        scripts=(),
        references=(),
        assets=(),
    )


# ============================================================================
# Value objects
# ============================================================================


def test_context_and_results_are_frozen():
    context = AgentContext(
        agent_hash="h",
        parent_hash=None,
        depth=0,
        task=None,
        skill=None,
        world={},
    )

    with pytest.raises(Exception):
        context.depth = 3

    result = AgentResult(
        content="x",
        messages=(),
        response=None,
    )

    with pytest.raises(Exception):
        result.content = "y"

    turn = AgentTurn(
        turn_id="t",
        agent_hash="h",
        depth=0,
        user_input="u",
        data={},
    )

    with pytest.raises(Exception):
        turn.user_input = "v"


@pytest.mark.asyncio
async def test_dispatch_metadata_reaches_child():
    runtime = AgentRuntime()

    root = runtime.create_root(
        world={"k": "v"},
        task="root",
    )

    seen = {}

    async def worker(child):
        seen.update(
            metadata=dict(child.metadata),
            world=child.world,
            depth=child.depth,
            parent=child.parent_hash,
        )

        class R:
            content = "done"

        return R()

    handle = runtime.dispatch(
        root,
        task="t",
        worker=worker,
        metadata={"origin": "enrichment"},
    )

    await handle.wait()

    assert seen["metadata"] == {
        "origin": "enrichment"
    }
    assert seen["world"] == {"k": "v"}
    assert seen["depth"] == 1
    assert seen["parent"] == root.agent_hash


def test_depth_limit_zero_blocks_immediately():
    runtime = AgentRuntime(max_subagent_depth=0)

    root = runtime.create_root(world={})

    async def worker(context):
        raise AssertionError("must not run")

    with pytest.raises(SubagentLimitError):
        runtime.dispatch(
            root,
            task="t",
            worker=worker,
        )


# ============================================================================
# Prompt assembly
# ============================================================================


def test_build_messages_orders_layers():
    persona = "PERSONA"

    current = [
        Message(role="user", content="hi")
    ]

    # Persona alone.
    messages = build_messages(
        persona=persona,
        skill_section=None,
        ambient_context=[],
        current=current,
    )

    assert len(messages) == 2
    assert messages[0].content == "PERSONA"

    # All layers stacked in order, XML-tagged.
    messages = build_messages(
        persona=persona,
        skill_section="active-skill-body",
        ambient_context=["note-1"],
        current=current,
        running_subagents="- id: abcd1234 | task: side job",
    )

    system = messages[0].content

    assert system.index("PERSONA") < system.index("<skills>")
    assert system.index("<skills>") < system.index(
        "<running_subagents>"
    )
    assert system.index("<running_subagents>") < system.index(
        "<module>"
    )
    assert "side job" in system
    assert "note-1" in system

    # An empty section body omits its container entirely.
    slim = build_messages(
        persona=persona,
        skill_section=None,
        ambient_context=[],
        current=current,
    )

    assert "<skills>" not in (slim[0].content or "")
    assert "<module>" not in (slim[0].content or "")

    # The current input stays outside the system message.
    assert messages[1].role == "user"
    assert messages[1].content == "hi"


def test_format_skill_sections_states():
    skill = _skill("alpha")

    # No active skill, empty catalog: body only, no container
    # (the container is added by build_messages).
    bare = format_skill_section(None, [])

    assert "No skill is currently active." in bare
    assert "[" not in bare and "]" not in bare

    # Active skill with catalog.
    full = format_skill_section(
        skill,
        [
            type(
                "M",
                (),
                {
                    "name": "alpha",
                    "description": "d",
                },
            )()
        ],
    )

    assert "Active skill: alpha" in full
    assert "DO ALPHA" in full
    assert "- alpha: d" in full
    assert "activate_skill" in full


# ============================================================================
# Reports
# ============================================================================


class _StubHandle:
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    @property
    def done(self):
        return True

    async def wait(self):
        if self._error is not None:
            raise self._error

        return self._value


def _child(task, handle):
    return ChildSubagent(
        id="abc12345",
        task=task,
        handle=handle,
    )


@pytest.mark.asyncio
async def test_collect_marks_children_consumed():
    children = [
        _child(
            "t",
            _StubHandle(
                value=AgentResult(
                    content="ok",
                    messages=(),
                    response=None,
                )
            ),
        )
    ]

    first = await collect_finished_children(children)

    assert len(first) == 1
    assert "status: completed" in first[0]

    # Consumed: the second sweep delivers nothing.
    second = await collect_finished_children(children)

    assert second == []


@pytest.mark.asyncio
async def test_long_task_preview_is_truncated():
    long_task = "x" * 500

    report = None

    children = [_child(long_task, _StubHandle(value=AgentResult(content="c", messages=(), response=None)))]

    reports = await collect_finished_children(children)

    report = reports[0]

    line = next(
        line
        for line in report.splitlines()
        if line.startswith("task:")
    )

    assert line.startswith("task: " + "x" * 200 + "...")
    assert len(line) < 300


# ============================================================================
# Verbs: validation matrices
# ============================================================================


class RecordingRuntime:
    def __init__(self):
        self.slept = []

    async def sleep(self, seconds):
        self.slept.append(seconds)
        return seconds, False


class FakeEngine:
    def __init__(self, skills=None):
        self.agent_runtime = RecordingRuntime()
        self.skills = skills


def _state(skill=None):
    return ExecutionState(
        active_skill=skill,
        persona="P",
    )


class _Call:
    def __init__(self, arguments):
        self.id = "c1"
        self.name = "?"
        self.arguments = arguments


class _Ctx:
    def __init__(self, depth=0):
        self.depth = depth


@pytest.mark.asyncio
async def test_sleep_argument_matrix():
    verb = VERBS["sleep"]

    engine = FakeEngine()

    cases = [
        ({}, "requires"),
        ({"seconds": "x"}, "number"),
        ({"seconds": -1}, ">= 0"),
    ]

    for arguments, fragment in cases:
        reply = await verb.execute(
            call=_Call(arguments),
            context=_Ctx(),
            state=_state(),
            engine=engine,
        )

        assert fragment in reply, arguments

    ok = await verb.execute(
        call=_Call({"seconds": 0.5}),
        context=_Ctx(),
        state=_state(),
        engine=engine,
    )

    assert "Waited" in ok
    assert engine.agent_runtime.slept == [0.5]


@pytest.mark.asyncio
async def test_dispatch_rejects_blank_task():
    verb = VERBS["dispatch_subagent"]

    reply = await verb.execute(
        call=_Call({"task": "   "}),
        context=_Ctx(),
        state=_state(),
        engine=FakeEngine(),
    )

    assert "non-empty" in reply


@pytest.mark.asyncio
async def test_activate_reports_available_skills():
    class FakeSkills:
        def names(self):
            return ("a", "b")

        def activate(self, name):
            if name not in ("a", "b"):
                raise UnknownSkillError(name)

            return _skill(name)

    verb = VERBS[ACTIVATE_NAME]

    engine = FakeEngine(skills=FakeSkills())

    missing = await verb.execute(
        call=_Call({}),
        context=_Ctx(depth=1),
        state=_state(),
        engine=engine,
    )

    assert "requires 'name'" in missing
    assert "a, b" in missing

    unknown = await verb.execute(
        call=_Call({"name": "zzz"}),
        context=_Ctx(depth=1),
        state=_state(),
        engine=engine,
    )

    assert "Unknown Skill 'zzz'" in unknown
    assert "a, b" in unknown

    state = _state()

    ok = await verb.execute(
        call=_Call({"name": "a"}),
        context=_Ctx(depth=1),
        state=state,
        engine=engine,
    )

    assert "activated" in ok
    assert state.active_skill.name == "a"


from src.nan_itself.agent.verbs import (
    ACTIVATE_SKILL_TOOL_NAME as ACTIVATE_NAME,
)


def test_verb_schemas_are_tight():
    for name in ("sleep", "dispatch_subagent", ACTIVATE_NAME):
        schema = VERBS[name].definition().input_schema

        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
        assert schema["required"], name


# ============================================================================
# Engine invariants
# ============================================================================


class CountingModules:
    def __init__(self):
        self.query_calls = 0
        self.turn_records: list = []

    def snapshot(self):
        return {}

    def deliver_turn(self, record):
        self.turn_records.append(record)

    async def query_snapshot(self, turn, world):
        self.query_calls += 1

        return []


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def generate_complete(self, request):
        self.requests.append(request)

        await asyncio.sleep(0)

        return self.responses.pop(0)


def _text_response(text):
    class R:
        content = text
        tool_calls = []
        finish_reason = "stop"

    return R()


def _call_response(call):
    class R:
        content = ""
        tool_calls = [call]
        finish_reason = "tool_calls"

    return R()


class _MinimalSkills:
    def refresh(self):
        pass

    def catalog(self):
        return ()

    def names(self):
        return ()

    def activate(self, name):
        raise UnknownSkillError(name)


def _make_engine(llm, modules):
    runtime = AgentRuntime()

    return StepEngine(
        llm=llm,
        modules=modules,
        providers=ProviderRuntime(),
        skills=_MinimalSkills(),
        agent_runtime=runtime,
    ), runtime


def _root(engine_runtime, world=None):
    return engine_runtime.create_root(
        world=world or {},
        skill=None,
        task="t",
    )


@pytest.mark.asyncio
async def test_ambient_queried_once_per_execution():
    modules = CountingModules()

    from src.nan_itself.utils.llm import (
        ToolCall,
    )

    llm = ScriptedLLM([
        _call_response(
            ToolCall(
                id="s1",
                name="sleep",
                arguments={"seconds": 0},
            )
        ),
        _text_response("done"),
    ])

    engine, runtime = _make_engine(llm, modules)

    await engine.execute(
        context=_root(runtime),
        user_input="go",
        persona="P",
    )

    # Multi-step turn still queries the world exactly once:
    # one consistent snapshot for the whole execution.
    assert modules.query_calls == 1


@pytest.mark.asyncio
async def test_subtree_orphans_are_dropped():
    released = asyncio.Event()
    child_started = asyncio.Event()

    class GatedDeep(StepEngine):
        async def execute(self, **kwargs):
            if kwargs["context"].depth == 2:
                child_started.set()

                await released.wait()

                return AgentResult(
                    content="deep done",
                    messages=(),
                    response=None,
                )

            return await super().execute(**kwargs)

    modules = CountingModules()

    from src.nan_itself.utils.llm import (
        ToolCall,
    )

    # Root: dispatch A then finish. A: dispatch B then finish.
    root_llm = ScriptedLLM([
        _call_response(
            ToolCall(
                id="da",
                name="dispatch_subagent",
                arguments={"task": "A"},
            )
        ),
        _text_response("root out"),
    ])

    inner_llm = ScriptedLLM([
        _call_response(
            ToolCall(
                id="db",
                name="dispatch_subagent",
                arguments={"task": "B"},
            )
        ),
        _text_response("A out"),
    ])

    engine, runtime = _make_engine(root_llm, modules)

    gated = GatedDeep(
        llm=inner_llm,
        modules=modules,
        providers=ProviderRuntime(),
        skills=_MinimalSkills(),
        agent_runtime=runtime,
    )

    # Route depth-1 executions through the gated engine by
    # wrapping dispatch at the root: easiest is to swap the
    # engine's own recursion target for depth-1 workers.
    original_execute = engine.execute

    parked = []

    async def wrapped(**kwargs):
        if kwargs["context"].depth >= 1:
            return await gated.execute(**kwargs)

        return await original_execute(**kwargs)

    engine.execute = wrapped  # type: ignore[method-assign]

    result = await engine.execute(
        context=_root(runtime),
        user_input="go",
        persona="P",
        report_sink=parked.append,
    )

    assert result.content == "root out"

    await asyncio.wait_for(
        child_started.wait(),
        timeout=2.0,
    )

    # A finished while root was still stepping? Either way only
    # MAIN-tree orphans may be parked. B (depth-2, gated) belongs
    # to a dead subtree and must be dropped silently.
    released.set()

    await asyncio.sleep(0.05)

    flat = [
        report
        for batch in parked
        for report in batch
    ]

    assert len(flat) == 1
    assert "task: A\n" in flat[0]
    assert "deep done" not in flat[0]
    assert not any("task: B" in r for r in flat)


def test_serialize_tool_result_branches():
    assert (
        _serialize_tool_result("raw")
        == "raw"
    )

    dumped = _serialize_tool_result({"a": 1})

    assert '"a"' in dumped and "1" in dumped

    class Weird:
        def __repr__(self):
            return "<weird>"

    # default=str stringifies, then json quotes it.
    assert (
        _serialize_tool_result(Weird())
        == '"<weird>"'
    )



@pytest.mark.asyncio
async def test_running_subagents_section_tracks_children():
    child_started = asyncio.Event()

    released = asyncio.Event()

    class GatedDeep(StepEngine):
        async def execute(self, **kwargs):
            if kwargs["context"].depth == 1:
                child_started.set()

                await released.wait()

                return AgentResult(
                    content="child done",
                    messages=(),
                    response=None,
                )

            return await super().execute(**kwargs)

    modules = CountingModules()

    from src.nan_itself.utils.llm import (
        ToolCall,
    )

    llm = ScriptedLLM([
        _call_response(
            ToolCall(
                id="d",
                name="dispatch_subagent",
                arguments={"task": "watch me work"},
            )
        ),
        _text_response("noted"),
    ])

    engine, runtime = _make_engine(llm, modules)

    gated = GatedDeep(
        llm=ScriptedLLM([]),
        modules=modules,
        providers=ProviderRuntime(),
        skills=_MinimalSkills(),
        agent_runtime=runtime,
    )

    original_execute = engine.execute

    async def wrapped(**kwargs):
        if kwargs["context"].depth >= 1:
            kwargs.pop("report_sink", None)

            return await gated.execute(**kwargs)

        return await original_execute(**kwargs)

    engine.execute = wrapped  # type: ignore[method-assign]

    parked: list[str] = []

    result = await engine.execute(
        context=_root(runtime),
        user_input="go",
        persona="P",
        report_sink=parked.append,
    )

    # Before any dispatch: no such section.
    first_system = (
        llm.requests[0].messages[0].content or ""
    )

    assert "<running_subagents>" not in first_system

    # While the gated child is in flight, the very next step
    # sees it.
    second_system = (
        llm.requests[1].messages[0].content or ""
    )

    assert "<running_subagents>" in second_system

    running_lines = [
        line
        for line in second_system.splitlines()
        if line.startswith("- id:")
    ]

    assert len(running_lines) == 1
    assert "watch me work" in running_lines[0]

    assert result.content == "noted"

    # Turn ended with the child still running: main-tree orphan
    # gets parked once it finishes.
    await asyncio.wait_for(child_started.wait(), timeout=2.0)

    released.set()

    await asyncio.sleep(0.05)

    flat = [r for batch in parked for r in batch]

    assert len(flat) == 1
    assert "child done" in flat[0]




# ============================================================================
# TurnRecord delivery through the engine
# ============================================================================


@pytest.mark.asyncio
async def test_turn_record_fields_on_success():
    modules = CountingModules()

    llm = ScriptedLLM([
        _text_response("final words"),
    ])

    engine, runtime = _make_engine(llm, modules)

    root = runtime.create_root(
        world={"k": "v"},
        task=None,
    )

    await engine.execute(
        context=root,
        user_input="hello there",
        persona="P",
    )

    for _ in range(200):
        if len(modules.turn_records) == 1:
            break

        await asyncio.sleep(0.005)

    assert len(modules.turn_records) == 1

    record = modules.turn_records[0]

    # Seen side.
    assert record.agent_hash == root.agent_hash
    assert record.parent_hash is None
    assert record.depth == 0
    assert record.user_input == "hello there"
    assert record.world == {"k": "v"}

    # Returned side.
    assert record.reply == "final words"
    assert record.error is None

    assert record.started_at <= record.ended_at


@pytest.mark.asyncio
async def test_turn_record_captures_error():
    class ExplodingLLM:
        async def generate_complete(self, request):
            raise RuntimeError("backend exploded")

    modules = CountingModules()

    engine, runtime = _make_engine(ExplodingLLM(), modules)

    with pytest.raises(RuntimeError):
        await engine.execute(
            context=_root(runtime),
            user_input="go",
            persona="P",
        )

    for _ in range(200):
        if len(modules.turn_records) == 1:
            break

        await asyncio.sleep(0.005)

    record = modules.turn_records[0]

    assert record.reply is None
    assert "RuntimeError" in record.error
    assert "backend exploded" in record.error


@pytest.mark.asyncio
async def test_slow_on_turn_does_not_block_execution(tmp_path):
    """
    Feeding discipline, enforced from the other side: the engine
    must not wait for any module's on_turn handler.
    """
    import src.nan_itself.modules as modules_pkg

    release = asyncio.Event()

    entered = asyncio.Event()

    class SlowFeeder(modules_pkg.Module):
        id = "slow-feeder"

        def __init__(self):
            self.records: list = []

        async def start(self):
            while True:
                await asyncio.sleep(10)

        async def on_turn(self, record):
            entered.set()

            await release.wait()

            self.records.append(record)

    from pathlib import Path as _Path

    facade = modules_pkg.Facade(
        workspace_modules=_Path(tmp_path) / "ws",
        builtin_modules=(SlowFeeder,),
        data_dir=_Path(tmp_path) / "data",
        scan_interval=0.03,
        retry_interval=0.03,
    )

    await facade.start()

    try:
        llm = ScriptedLLM([
            _text_response("quick"),
        ])

        runtime = AgentRuntime()

        # The engine points at the real facade: only registered
        # modules receive deliveries.
        engine = StepEngine(
            llm=llm,
            modules=facade,
            providers=ProviderRuntime(),
            skills=_MinimalSkills(),
            agent_runtime=runtime,
        )

        t0 = time.monotonic()

        result = await engine.execute(
            context=_root(runtime),
            user_input="go",
            persona="P",
            report_sink=None,
        )

        elapsed = time.monotonic() - t0

        # Execution returned without waiting for the feeder.
        assert result.content == "quick"
        assert elapsed < 0.15

        ok = await _eventually(
            lambda: entered.is_set()
        )

        assert ok

        release.set()

    finally:
        release.set()

        await facade.stop()


# ============================================================================
# Loop backoff rhythm
# ============================================================================


class FlakyAgent:
    def __init__(self, failures):
        self.failures = failures
        self.runs = []
        self._pending_reports = []

    def has_pending_reports(self):
        return bool(self._pending_reports)

    def drain_pending_reports(self):
        out = list(self._pending_reports)

        self._pending_reports.clear()

        return out

    async def run(self, inputs):
        self.runs.append(inputs)

        if self.failures > 0:
            self.failures -= 1

            raise RuntimeError("flaky")

        return AgentResult(
            content="ok",
            messages=(),
            response=None,
        )


@pytest.mark.asyncio
async def test_backoff_retries_same_batch_then_resets():
    agent = FlakyAgent(failures=2)

    inbox = Inbox()

    inbox.put("batch-x")

    loop = AgentLoop(
        agent,
        inbox,
        turn_grace=0.1,
        backoff=(0.02, 0.04),
    )

    task = asyncio.create_task(loop.run_forever())

    await asyncio.wait_for(
        _eventually(lambda: len(agent.runs) == 3),
        timeout=2.0,
    )

    loop.request_stop()

    await asyncio.wait_for(task, timeout=2.0)

    # Same batch retried until success.
    assert agent.runs == ["batch-x", "batch-x", "batch-x"]

    # Two backoffs elapsed: 20ms + 40ms at minimum.
    assert loop.cycles == 1


async def _eventually(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if predicate():
            return True

        await asyncio.sleep(0.005)

    raise AssertionError("condition not met in time")


@pytest.mark.asyncio
async def test_stop_interrupts_backoff_quickly():
    agent = FlakyAgent(failures=1)

    inbox = Inbox()

    inbox.put("x")

    # Long backoff: stopping must not wait for it.
    loop = AgentLoop(
        agent,
        inbox,
        turn_grace=0.1,
        backoff=(30.0,),
    )

    task = asyncio.create_task(loop.run_forever())

    started = time.monotonic()

    await asyncio.wait_for(
        _eventually(lambda: len(agent.runs) == 1),
        timeout=2.0,
    )

    loop.request_stop()

    await asyncio.wait_for(task, timeout=2.0)

    elapsed = time.monotonic() - started

    assert elapsed < 5.0
