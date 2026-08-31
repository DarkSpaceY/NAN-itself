
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nan_itself.agent.engine import StepEngine
from nan_itself.agent.model import AgentResult
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.agent.verbs import (
    ActivateSkillVerb,
    DispatchVerb,
    ExecutionState,
    SleepVerb,
)
from nan_itself.skills import UnknownSkillError
from nan_itself.tools.results import text_result
from nan_itself.utils.llm import ToolCall


# ============================================================================
# Async helpers
# ============================================================================


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Generic test doubles
# ============================================================================


class FakeModules:
    def __init__(self):
        self.queries = 0
        self.turns = []

    async def query_snapshot(
        self,
        turn,
        snapshot,
        **kwargs,
    ):
        self.queries += 1
        self.turns.append(
            turn
        )
        return []

    def deliver_turn(
        self,
        record,
    ):
        pass


class FakeSkills:
    def __init__(
        self,
        skills=None,
    ):
        self._skills = dict(
            skills or {}
        )

        self.refresh_calls = 0
        self.activate_calls = []

    def refresh(self):
        self.refresh_calls += 1

    def catalog(self):
        return tuple(
            skill.metadata
            for skill in self._skills.values()
            if hasattr(skill, "metadata")
        )

    def names(self):
        return tuple(
            self._skills
        )

    def activate(
        self,
        name,
    ):
        self.activate_calls.append(
            name
        )

        if name not in self._skills:
            raise UnknownSkillError(
                name
            )

        return self._skills[name]


class EmptyProviders:
    def provider_names(self):
        return ()

    def get_provider(
        self,
        name,
    ):
        return None


class FakeLLM:
    model = "fake-model"
    provider = "fake-provider"

    def __init__(
        self,
        responses,
    ):
        self.responses = list(
            responses
        )

        self.requests = []

    async def generate_complete(
        self,
        request,
    ):
        self.requests.append(
            request
        )

        if not self.responses:
            raise AssertionError(
                "LLM received more requests than expected"
            )

        return self.responses.pop(0)


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


def make_engine(
    *,
    llm,
    runtime=None,
    skills=None,
):
    modules = FakeModules()

    providers = EmptyProviders()

    if runtime is None:
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

    if skills is None:
        skills = FakeSkills()

    engine = StepEngine(
        llm=llm,
        modules=modules,
        providers=providers,
        skills=skills,
        agent_runtime=runtime,
    )

    return (
        engine,
        modules,
        skills,
        runtime,
    )


def make_context(
    runtime,
    *,
    depth=0,
    skill=None,
    task="test",
):
    return runtime.create_root(
        world={
            "state": {
                "value": 1,
            }
        },
        skill=skill,
        task=task,
    )


# ============================================================================
# SleepVerb
# ============================================================================


def test_sleep_verb_real_wait():
    async def scenario():
        runtime = AgentRuntime()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        started = (
            asyncio.get_running_loop().time()
        )

        result = await SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": 0.02,
                }
            ),
            context=None,
            state=state,
            engine=SimpleNamespace(
                agent_runtime=runtime
            ),
        )

        elapsed = (
            asyncio.get_running_loop().time()
            - started
        )

        assert result == (
            "Waited 0.02 seconds."
        )

        assert elapsed >= 0.015

    run(
        scenario()
    )


def test_sleep_verb_real_interrupt():
    async def scenario():
        runtime = AgentRuntime()

        interrupt = asyncio.Event()

        runtime.set_interrupt_event(
            interrupt
        )

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        async def trigger():
            await asyncio.sleep(
                0.02
            )
            interrupt.set()

        trigger_task = asyncio.create_task(
            trigger()
        )

        result = await SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": 5,
                }
            ),
            context=None,
            state=state,
            engine=SimpleNamespace(
                agent_runtime=runtime
            ),
        )

        await trigger_task

        assert result.startswith(
            "Sleep interrupted after "
        )

        assert (
            "new input arrived"
            in result
        )

    run(
        scenario()
    )


def test_sleep_verb_zero_seconds_is_real_noop():
    async def scenario():
        runtime = AgentRuntime()

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        result = await SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": 0,
                }
            ),
            context=None,
            state=state,
            engine=SimpleNamespace(
                agent_runtime=runtime
            ),
        )

        assert result == (
            "Waited 0 seconds."
        )

    run(
        scenario()
    )


# ============================================================================
# ActivateSkillVerb
# ============================================================================


def test_activate_skill_changes_execution_state():
    async def scenario():
        skill = SimpleNamespace(
            name="research"
        )

        skills = FakeSkills(
            {
                "research": skill,
            }
        )

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        engine = SimpleNamespace(
            skills=skills
        )

        result = await ActivateSkillVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "name": "research",
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )

        assert result == (
            "Skill 'research' activated. It takes "
            "effect from your next step."
        )

        assert (
            state.active_skill
            is skill
        )

        assert (
            skills.activate_calls
            == ["research"]
        )

    run(
        scenario()
    )


def test_activate_skill_unknown_skill_does_not_mutate_state():
    async def scenario():
        original = SimpleNamespace(
            name="original"
        )

        skills = FakeSkills(
            {
                "research": SimpleNamespace(
                    name="research"
                )
            }
        )

        state = ExecutionState(
            active_skill=original,
            persona="test",
        )

        engine = SimpleNamespace(
            skills=skills
        )

        result = await ActivateSkillVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "name": "missing",
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )

        assert (
            "Unknown Skill 'missing'."
            in result
        )

        assert (
            state.active_skill
            is original
        )

    run(
        scenario()
    )


# ============================================================================
# Skill -> Dispatch inheritance
# ============================================================================


def test_activate_skill_then_dispatch_child_with_current_skill():
    """
    Real execution-state flow:

        activate_skill
            ->
        state.active_skill
            ->
        dispatch_subagent
            ->
        AgentContext.skill
    """

    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        root = make_context(
            runtime
        )

        research = SimpleNamespace(
            name="research"
        )

        state = ExecutionState(
            active_skill=None,
            persona="test-persona",
        )

        skills = FakeSkills(
            {
                "research": research,
            }
        )

        observed = {}

        async def child_execute(
            *,
            context,
            user_input,
            persona,
        ):
            observed["context"] = context

            return AgentResult(
                content="child complete",
                messages=(),
                response=SimpleNamespace(
                    content="child complete"
                ),
            )

        engine = SimpleNamespace(
            skills=skills,
            agent_runtime=runtime,
            execute=child_execute,
        )

        # Step 1: activate.
        await ActivateSkillVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "name": "research"
                }
            ),
            context=root,
            state=state,
            engine=engine,
        )

        assert (
            state.active_skill
            is research
        )

        # Step 2: dispatch using current state.
        dispatch_result = (
            await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "research child"
                    }
                ),
                context=root,
                state=state,
                engine=engine,
            )
        )

        assert (
            "Subagent dispatched."
            in dispatch_result
        )

        assert (
            len(state.children)
            == 1
        )

        child = state.children[0]

        child_result = (
            await child.handle.wait()
        )

        assert (
            child_result.content
            == "child complete"
        )

        child_context = (
            observed["context"]
        )

        assert (
            child_context.skill
            is research
        )

        assert (
            child_context.parent_hash
            == root.agent_hash
        )

        assert (
            child_context.world
            is root.world
        )

    run(
        scenario()
    )


# ============================================================================
# DispatchVerb -> actual child execution -> report
# ============================================================================


def test_dispatch_verb_child_executes_and_report_is_formatable():
    """
    The dispatch Verb must produce a real ChildSubagent whose handle
    can be awaited and whose result can be converted into the standard
    [Subagent Report] message.
    """

    from nan_itself.agent.reports import (
        format_child_report,
    )

    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        root = make_context(
            runtime,
            task="parent",
        )

        state = ExecutionState(
            active_skill=None,
            persona="parent-persona",
        )

        observed = {}

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            observed["context"] = context
            observed["user_input"] = user_input
            observed["persona"] = persona

            return AgentResult(
                content="research result",
                messages=(),
                response=SimpleNamespace(
                    content="research result"
                ),
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=execute_child,
        )

        dispatch_result = (
            await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "research the topic",
                    }
                ),
                context=root,
                state=state,
                engine=engine,
            )
        )

        assert (
            "Subagent dispatched."
            in dispatch_result
        )

        child = state.children[0]

        report = (
            await format_child_report(
                child
            )
        )

        assert (
            report.startswith(
                "[Subagent Report]\n"
            )
        )

        assert (
            "task: research the topic"
            in report
        )

        assert (
            "status: completed"
            in report
        )

        assert (
            "research result"
            in report
        )

        assert (
            observed["user_input"]
            == "research the topic"
        )

        assert (
            observed["persona"]
            == "parent-persona"
        )

        assert (
            observed["context"].skill
            is None
        )

        assert (
            observed["context"].world
            is root.world
        )

    run(
        scenario()
    )


def test_dispatch_child_failure_becomes_failed_report():
    from nan_itself.agent.reports import (
        format_child_report,
    )

    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        root = make_context(
            runtime
        )

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            raise RuntimeError(
                "child exploded"
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=execute_child,
        )

        await DispatchVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "task": "failing child"
                }
            ),
            context=root,
            state=state,
            engine=engine,
        )

        child = state.children[0]

        report = (
            await format_child_report(
                child
            )
        )

        assert (
            "status: failed"
            in report
        )

        assert (
            "child exploded"
            in report
        )

    run(
        scenario()
    )


# ============================================================================
# StepEngine: Sleep as an actual model-selected Verb
# ============================================================================


def test_step_engine_can_execute_sleep_then_continue():
    llm = FakeLLM(
        [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "sleep-1",
                        "sleep",
                        {
                            "seconds": 0,
                        },
                    )
                ]
            ),
            make_response(
                content="continued",
            ),
        ]
    )

    engine, modules, skills, runtime = (
        make_engine(
            llm=llm
        )
    )

    context = make_context(
        runtime,
        task="sleep test",
    )

    result = run(
        engine.execute(
            context=context,
            user_input="take a tiny pause",
            persona="test",
        )
    )

    assert (
        result.content
        == "continued"
    )

    assert (
        len(llm.requests)
        == 2
    )


# ============================================================================
# StepEngine: activate_skill as a model-selected Verb
# ============================================================================


def test_main_agent_cannot_activate_skill():
    """
    Main Agent is depth=0.

    activate_skill is intentionally unavailable to the Main Agent.
    The model may hallucinate the verb, but SkillRuntime.activate()
    must never be called.
    """

    metadata = SimpleNamespace(
        name="research",
        description="Research skill",
        origin="workspace",
        source=None,
        frontmatter={},
    )

    skill = SimpleNamespace(
        name="research",
        description="Research skill",
        metadata=metadata,
        instructions="Use careful source analysis.",
        resources=[],
        scripts=(),
        references=(),
        assets=(),
        generation=1,
    )

    skills = FakeSkills(
        {
            "research": skill,
        }
    )

    llm = FakeLLM(
        [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "skill-1",
                        "activate_skill",
                        {
                            "name": "research",
                        },
                    )
                ]
            ),
            make_response(
                content="continued",
            ),
        ]
    )

    engine, modules, _, runtime = (
        make_engine(
            llm=llm,
            skills=skills,
        )
    )

    context = make_context(
        runtime,
        depth=0,
    )

    result = run(
        engine.execute(
            context=context,
            user_input="switch to research",
            persona="test",
        )
    )

    assert (
        result.content
        == "continued"
    )

    # Main Agent is forbidden from activating Skills.
    assert (
        skills.activate_calls
        == []
    )

    assert len(
        llm.requests
    ) == 2

    tool_results = [
        message.content
        for request in llm.requests
        for message in request.messages
        if getattr(
            message,
            "role",
            None,
        ) == "tool"
    ]

    assert any(
        "only available to Subagents"
        in text
        for text in tool_results
    )


def test_subagent_can_activate_skill_then_finish():
    async def scenario():
        metadata = SimpleNamespace(
            name="research",
            description="Research skill",
            origin="workspace",
            source=None,
            frontmatter={},
        )

        skill = SimpleNamespace(
            name="research",
            description="Research skill",
            metadata=metadata,
            instructions=(
                "Use careful source analysis."
            ),
            resources=[],
            scripts=(),
            references=(),
            assets=(),
            generation=1,
        )

        skills = FakeSkills(
            {
                "research": skill,
            }
        )

        llm = FakeLLM(
            [
                make_response(
                    tool_calls=[
                        make_tool_call(
                            "skill-1",
                            "activate_skill",
                            {
                                "name": "research",
                            },
                        )
                    ]
                ),
                make_response(
                    content="research mode active",
                ),
            ]
        )

        engine, modules, _, runtime = (
            make_engine(
                llm=llm,
                skills=skills,
            )
        )

        root = make_context(
            runtime,
            depth=0,
            task="root",
        )

        observed = {}

        async def worker(
            context,
        ):
            observed["context"] = context

            return await engine.execute(
                context=context,
                user_input="switch to research",
                persona="test",
            )

        try:
            handle = runtime.dispatch(
                root,
                task="research task",
                worker=worker,
            )

            child_result = (
                await handle.wait()
            )

            assert (
                child_result.content
                == "research mode active"
            )

            assert (
                skills.activate_calls
                == ["research"]
            )

            child_context = (
                observed["context"]
            )

            assert (
                child_context.depth
                == 1
            )

            assert (
                child_context.parent_hash
                == root.agent_hash
            )

            assert len(
                llm.requests
            ) == 2

            second_system = (
                llm.requests[1]
                .messages[0]
                .content
            )

            assert (
                "research"
                in second_system
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


# ============================================================================
# StepEngine: dispatch -> child -> parent report -> final answer
# =====================================================================
def test_step_engine_dispatches_child_and_parent_receives_report():
    """
    Full Agent-facing dispatch loop:

        parent step 1
            ->
        dispatch_subagent
            ->
        parent step 2
            ->
        sleep(0) yields the event loop
            ->
        child executes and completes
            ->
        parent step 3 starts
            ->
        collect_finished_children()
            ->
        [Subagent Report] injected
            ->
        parent final answer

    IMPORTANT:

    collect_finished_children() runs at the BEGINNING of a StepEngine
    iteration. It does not run again after an LLM response returns.

    Therefore the test deliberately introduces a second step whose
    Verb yields to the event loop. That gives the child time to finish
    before the following iteration begins and collects the report.
    """

    class DispatchLLM:
        model = "fake-model"
        provider = "fake-provider"

        def __init__(self):
            self.requests = []

            self.parent_steps = 0
            self.child_calls = 0

        async def generate_complete(
            self,
            request,
        ):
            self.requests.append(
                request
            )

            # ------------------------------------------------------
            # Real LLM calls are asynchronous.
            #
            # Give already-created Subagent Tasks a chance to run.
            # ------------------------------------------------------

            await asyncio.sleep(0)

            user_texts = [
                message.content
                for message in request.messages
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
            ]

            joined = "\n".join(
                user_texts
            )

            # ======================================================
            # CHILD
            # ======================================================

            if (
                "child task"
                in joined
                and "[Subagent Report]"
                not in joined
            ):
                self.child_calls += 1

                return make_response(
                    content="child result"
                )

            # ======================================================
            # PARENT
            # ======================================================

            if (
                "parent task"
                in joined
            ):
                has_report = (
                    "[Subagent Report]"
                    in joined
                )

                # Once the report exists, the parent finishes.
                if has_report:
                    return make_response(
                        content=(
                            "parent received "
                            "child result"
                        )
                    )

                self.parent_steps += 1

                # Parent step 1:
                # dispatch exactly one child.
                if (
                    self.parent_steps
                    == 1
                ):
                    return make_response(
                        tool_calls=[
                            make_tool_call(
                                "dispatch-1",
                                "dispatch_subagent",
                                {
                                    "task": "child task"
                                },
                            )
                        ]
                    )

                # Parent step 2:
                # deliberately yield through a real Verb.
                #
                # This gives the child time to finish before the
                # next StepEngine iteration starts.
                if (
                    self.parent_steps
                    == 2
                ):
                    return make_response(
                        tool_calls=[
                            make_tool_call(
                                "sleep-1",
                                "sleep",
                                {
                                    "seconds": 0,
                                },
                            )
                        ]
                    )

                raise AssertionError(
                    "Parent advanced past the expected "
                    "dispatch/sleep sequence without "
                    "receiving a Subagent report."
                )

            raise AssertionError(
                "Unexpected LLM request:\n"
                + joined
            )

    async def scenario():
        llm = DispatchLLM()

        engine, modules, skills, runtime = (
            make_engine(
                llm=llm
            )
        )

        context = make_context(
            runtime,
            task="parent task",
        )

        try:
            result = await engine.execute(
                context=context,
                user_input="parent task",
                persona="parent persona",
            )

            # ------------------------------------------------------
            # Final parent response.
            # ------------------------------------------------------

            assert (
                result.content
                == "parent received "
                "child result"
            )

            # ------------------------------------------------------
            # Exactly one child.
            # ------------------------------------------------------

            assert (
                llm.child_calls
                == 1
            )

            assert (
                llm.parent_steps
                == 2
            )

            children = [
                agent
                for agent in runtime.agents()
                if (
                    agent.parent_hash
                    == context.agent_hash
                )
            ]

            assert (
                len(children)
                == 1
            )

            child_context = children[0]

            assert (
                child_context.task
                == "child task"
            )

            assert (
                child_context.depth
                == 1
            )

            assert (
                child_context.parent_hash
                == context.agent_hash
            )

            # The entire dispatch tree uses the same turn world.
            assert (
                child_context.world
                is context.world
            )

            # ------------------------------------------------------
            # Verify that a report was actually injected.
            # ------------------------------------------------------

            report_requests = []

            for request in llm.requests:
                contains_report = any(
                    (
                        "[Subagent Report]"
                        in (
                            message.content
                            or ""
                        )
                    )
                    for message in request.messages
                    if getattr(
                        message,
                        "role",
                        None,
                    )
                    == "user"
                    and isinstance(
                        getattr(
                            message,
                            "content",
                            None,
                        ),
                        str,
                    )
                )

                if contains_report:
                    report_requests.append(
                        request
                    )

            assert (
                report_requests
            )

            report_text = "\n".join(
                message.content
                for message in (
                    report_requests[-1].messages
                )
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
                and "[Subagent Report]"
                in (
                    message.content
                    or ""
                )
            )

            assert (
                "[Subagent Report]"
                in report_text
            )

            assert (
                "id:"
                in report_text
            )

            assert (
                "task: child task"
                in report_text
            )

            assert (
                "status: completed"
                in report_text
            )

            assert (
                "child result"
                in report_text
            )

            # ------------------------------------------------------
            # Parent's final request must contain the report.
            # ------------------------------------------------------

            final_request = (
                report_requests[-1]
            )

            final_user_text = "\n".join(
                message.content
                for message in final_request.messages
                if getattr(
                    message,
                    "role",
                    None,
                )
                == "user"
                and isinstance(
                    getattr(
                        message,
                        "content",
                        None,
                    ),
                    str,
                )
            )

            assert (
                "parent task"
                in final_user_text
            )

            assert (
                "[Subagent Report]"
                in final_user_text
            )

            assert (
                "child result"
                in final_user_text
            )

            # ------------------------------------------------------
            # No leaked Subagent Tasks.
            #
            # The child should already be finished by the time the
            # parent receives its report.
            # ------------------------------------------------------

            assert (
                runtime.active_subagent_count
                == 0
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


# ============================================================================
# Dispatch depth enforcement through the Verb
# ============================================================================


def test_dispatch_verb_respects_runtime_depth_limit():
    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=0
        )

        root = make_context(
            runtime
        )

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=lambda **kwargs: None,
        )

        result = await DispatchVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "task": "too deep",
                }
            ),
            context=root,
            state=state,
            engine=engine,
        )

        assert (
            "Maximum Subagent depth exceeded"
            in result
        )

        assert (
            state.children
            == []
        )

    run(
        scenario()
    )


# ============================================================================
# Cross-Verb execution state isolation
# ============================================================================


def test_execution_state_is_not_shared_between_independent_executions():
    first = ExecutionState(
        active_skill=SimpleNamespace(
            name="first"
        ),
        persona="one",
    )

    second = ExecutionState(
        active_skill=None,
        persona="two",
    )

    assert (
        first.active_skill
        is not second.active_skill
    )

    first.active_skill = (
        SimpleNamespace(
            name="changed"
        )
    )

    assert (
        second.active_skill
        is None
    )

    assert first.persona == "one"
    assert second.persona == "two"