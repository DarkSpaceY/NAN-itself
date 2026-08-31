from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nan_itself.agent.engine import StepEngine
from nan_itself.agent.verbs import (
    ACTIVATE_SKILL_TOOL_NAME,
    DISPATCH_SUBAGENT_TOOL_NAME,
    SLEEP_TOOL_NAME,
    ActivateSkillVerb,
    DispatchVerb,
    ExecutionState,
    SleepVerb,
    VERBS,
)
from nan_itself.tools.results import text_result
from nan_itself.utils.llm import ToolCall


def run(coro):
    return asyncio.run(coro)


# ============================================================================
# Fake infrastructure
# ============================================================================


class FakeModules:
    """
    Minimal Module facade required by StepEngine.execute().
    """

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


class FakeLLM:
    model = "fake-model"
    provider = "fake-provider"

    def __init__(
        self,
        responses,
    ):
        self.responses = list(responses)
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
                "LLM asked for more responses than expected"
            )

        return self.responses.pop(0)


class FakeProviderRuntime:
    """
    Minimal ProviderRuntime implementation for AgentToolView.
    """

    def __init__(self):
        self.called = []

        self.providers = {
            "calc": SimpleNamespace(
                tools={
                    "add": SimpleNamespace(
                        name="add",
                        description="Add two numbers.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "x": {
                                    "type": "number",
                                },
                                "y": {
                                    "type": "number",
                                },
                            },
                            "required": [
                                "x",
                                "y",
                            ],
                            "additionalProperties": False,
                        },
                    )
                }
            )
        }

    def provider_names(self):
        return tuple(
            self.providers
        )

    def get_provider(
        self,
        name,
    ):
        return self.providers.get(
            name
        )

    async def refresh_provider_tools(
        self,
        name,
    ):
        assert name in self.providers

    async def call_tool(
        self,
        provider_name,
        tool_name,
        arguments,
    ):
        self.called.append(
            (
                provider_name,
                tool_name,
                dict(arguments),
            )
        )

        value = (
            arguments["x"]
            + arguments["y"]
        )

        return text_result(
            str(value)
        )


def tool_call(
    call_id,
    name,
    arguments,
) -> ToolCall:
    """
    Construct the exact ToolCall model expected by Message.
    """
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
    )


def llm_response(
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


# ============================================================================
# Verb registry / definitions
# ============================================================================


def test_all_verbs_are_registered_once():
    assert set(VERBS) == {
        SLEEP_TOOL_NAME,
        DISPATCH_SUBAGENT_TOOL_NAME,
        ACTIVATE_SKILL_TOOL_NAME,
    }

    names = [
        verb.definition().name
        for verb in VERBS.values()
    ]

    assert len(names) == len(
        set(names)
    )


def test_verb_schemas_require_expected_arguments():
    assert (
        SleepVerb()
        .definition()
        .input_schema["required"]
        == ["seconds"]
    )

    assert (
        DispatchVerb()
        .definition()
        .input_schema["required"]
        == ["task"]
    )

    assert (
        ActivateSkillVerb()
        .definition()
        .input_schema["required"]
        == ["name"]
    )


# ============================================================================
# SleepVerb
# ============================================================================


def test_sleep_verb_rejects_missing_seconds():
    class Runtime:
        async def sleep(
            self,
            seconds,
        ):
            raise AssertionError(
                "sleep must not be called"
            )

    engine = SimpleNamespace(
        agent_runtime=Runtime()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        SleepVerb().execute(
            call=SimpleNamespace(
                arguments={}
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert result == (
        "sleep requires 'seconds'."
    )


def test_sleep_verb_rejects_non_number():
    class Runtime:
        async def sleep(
            self,
            seconds,
        ):
            raise AssertionError(
                "sleep must not be called"
            )

    engine = SimpleNamespace(
        agent_runtime=Runtime()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": "1",
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert result == (
        "'seconds' must be a number."
    )


def test_sleep_verb_rejects_negative():
    class Runtime:
        async def sleep(
            self,
            seconds,
        ):
            raise AssertionError(
                "sleep must not be called"
            )

    engine = SimpleNamespace(
        agent_runtime=Runtime()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": -1,
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert result == (
        "'seconds' must be >= 0."
    )


def test_sleep_verb_calls_runtime():
    class Runtime:
        def __init__(self):
            self.calls = []

        async def sleep(
            self,
            seconds,
        ):
            self.calls.append(
                seconds
            )
            return (
                seconds,
                False,
            )

    runtime = Runtime()

    engine = SimpleNamespace(
        agent_runtime=runtime
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": 1.5,
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert runtime.calls == [
        1.5
    ]

    assert result == (
        "Waited 1.5 seconds."
    )


def test_sleep_verb_reports_interrupt():
    class Runtime:
        async def sleep(
            self,
            seconds,
        ):
            return (
                0.7,
                True,
            )

    engine = SimpleNamespace(
        agent_runtime=Runtime()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        SleepVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "seconds": 10,
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert result == (
        "Sleep interrupted after 0.7s: "
        "new input arrived. End your turn now so it "
        "can be processed."
    )


# ============================================================================
# DispatchVerb
# ============================================================================


def test_dispatch_verb_rejects_empty_task():
    class Runtime:
        def dispatch(
            self,
            *args,
            **kwargs,
        ):
            raise AssertionError(
                "dispatch must not be called"
            )

    engine = SimpleNamespace(
        agent_runtime=Runtime()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        DispatchVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "task": "   "
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert (
        "non-empty 'task'"
        in result
    )


def test_dispatch_verb_registers_child():
    class Runtime:
        def __init__(self):
            self.calls = []

        def dispatch(
            self,
            context,
            *,
            task,
            worker,
            skill=None,
            metadata=None,
        ):
            self.calls.append(
                {
                    "context": context,
                    "task": task,
                    "worker": worker,
                    "skill": skill,
                    "metadata": metadata,
                }
            )

            return SimpleNamespace(
                agent_hash="abcdefgh12345678",
                depth=2,
            )

    runtime = Runtime()

    import nan_itself.agent.verbs as verbs_module

    original_child_type = (
        verbs_module.ChildSubagent
    )

    verbs_module.ChildSubagent = (
        lambda *,
        id,
        task,
        handle: SimpleNamespace(
            id=id,
            task=task,
            handle=handle,
            reported=False,
        )
    )

    try:
        active_skill = SimpleNamespace(
            name="research"
        )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=lambda **kwargs: None,
        )

        state = ExecutionState(
            active_skill=active_skill,
            persona="test",
        )

        context = SimpleNamespace(
            agent_hash="parent"
        )

        result = run(
            DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "do something",
                    }
                ),
                context=context,
                state=state,
                engine=engine,
            )
        )

    finally:
        verbs_module.ChildSubagent = (
            original_child_type
        )

    assert len(
        runtime.calls
    ) == 1

    call = runtime.calls[0]

    assert (
        call["context"]
        is context
    )

    assert (
        call["task"]
        == "do something"
    )

    assert callable(
        call["worker"]
    )

    assert (
        call["skill"]
        is active_skill
    )

    assert (
        call["metadata"]
        is None
    )

    assert (
        "Subagent dispatched."
        in result
    )

    assert len(
        state.children
    ) == 1

    assert (
        state.children[0].id
        == "abcdefgh"
    )

    assert (
        state.children[0].task
        == "do something"
    )


def test_dispatch_verb_passes_none_skill_when_execution_has_no_skill():
    class Runtime:
        def __init__(self):
            self.calls = []

        def dispatch(
            self,
            context,
            *,
            task,
            worker,
            skill=None,
            metadata=None,
        ):
            self.calls.append(
                {
                    "context": context,
                    "task": task,
                    "worker": worker,
                    "skill": skill,
                    "metadata": metadata,
                }
            )

            return SimpleNamespace(
                agent_hash="abcdefgh12345678",
                depth=2,
            )

    runtime = Runtime()

    import nan_itself.agent.verbs as verbs_module

    original_child_type = (
        verbs_module.ChildSubagent
    )

    verbs_module.ChildSubagent = (
        lambda *,
        id,
        task,
        handle: SimpleNamespace(
            id=id,
            task=task,
            handle=handle,
            reported=False,
        )
    )

    try:
        engine = SimpleNamespace(
            agent_runtime=runtime,
            execute=lambda **kwargs: None,
        )

        state = ExecutionState(
            active_skill=None,
            persona="test",
        )

        context = SimpleNamespace(
            agent_hash="parent"
        )

        result = run(
            DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "do something",
                    }
                ),
                context=context,
                state=state,
                engine=engine,
            )
        )

    finally:
        verbs_module.ChildSubagent = (
            original_child_type
        )

    assert (
        len(runtime.calls)
        == 1
    )

    assert (
        runtime.calls[0]["skill"]
        is None
    )

    assert (
        "Subagent dispatched."
        in result
    )


# ============================================================================
# ActivateSkillVerb
# ============================================================================


def test_activate_skill_rejects_missing_name():
    class Skills:
        def names(self):
            return (
                "research",
            )

    engine = SimpleNamespace(
        skills=Skills()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        ActivateSkillVerb().execute(
            call=SimpleNamespace(
                arguments={}
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert (
        "activate_skill requires 'name'."
        in result
    )

    assert (
        "research"
        in result
    )


def test_activate_skill_updates_execution_state():
    skill = SimpleNamespace(
        name="research"
    )

    class Skills:
        def names(self):
            return (
                "research",
            )

        def activate(
            self,
            name,
        ):
            assert name == (
                "research"
            )
            return skill

    engine = SimpleNamespace(
        skills=Skills()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        ActivateSkillVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "name": "research",
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert (
        state.active_skill
        is skill
    )

    assert result == (
        "Skill 'research' activated. It takes "
        "effect from your next step."
    )


def test_activate_skill_unknown_name_returns_error():
    from nan_itself.skills import (
        UnknownSkillError,
    )

    class Skills:
        def names(self):
            return (
                "research",
                "coding",
            )

        def activate(
            self,
            name,
        ):
            raise UnknownSkillError(
                name
            )

    engine = SimpleNamespace(
        skills=Skills()
    )

    state = ExecutionState(
        active_skill=None,
        persona="test",
    )

    result = run(
        ActivateSkillVerb().execute(
            call=SimpleNamespace(
                arguments={
                    "name": "missing",
                }
            ),
            context=None,
            state=state,
            engine=engine,
        )
    )

    assert (
        "Unknown Skill 'missing'."
        in result
    )

    assert (
        "research"
        in result
    )

    assert (
        "coding"
        in result
    )

    assert (
        state.active_skill
        is None
    )


# ============================================================================
# Agent -> route -> tool -> final response
# ============================================================================


def test_step_engine_routes_then_calls_tool_then_finishes():
    llm = FakeLLM(
        [
            llm_response(
                tool_calls=[
                    tool_call(
                        "1",
                        "route",
                        {
                            "provider_name": "calc",
                        },
                    )
                ]
            ),
            llm_response(
                tool_calls=[
                    tool_call(
                        "2",
                        "add",
                        {
                            "x": 2,
                            "y": 3,
                        },
                    )
                ]
            ),
            llm_response(
                content="done"
            ),
        ]
    )

    providers = FakeProviderRuntime()

    engine = StepEngine(
        llm=llm,
        modules=FakeModules(),
        providers=providers,
        skills=SimpleNamespace(
            catalog=lambda: ()
        ),
        agent_runtime=SimpleNamespace(),
    )

    context = SimpleNamespace(
        skill=None,
        world={},
        agent_hash="agent1",
        parent_hash=None,
        depth=0,
        task=None,
    )

    result = run(
        engine.execute(
            context=context,
            user_input=(
                "please calculate 2 + 3"
            ),
            persona="test",
        )
    )

    assert result.content == (
        "done"
    )

    assert providers.called == [
        (
            "calc",
            "add",
            {
                "x": 2,
                "y": 3,
            },
        )
    ]

    assert len(
        llm.requests
    ) == 3

    first_tool_names = {
        tool.name
        for tool in llm.requests[0].tools
    }

    second_tool_names = {
        tool.name
        for tool in llm.requests[1].tools
    }

    # Before routing only `route` is visible.
    assert (
        "route"
        in first_tool_names
    )

    assert (
        "add"
        not in first_tool_names
    )

    # After routing the whole provider block is visible.
    assert (
        "add"
        in second_tool_names
    )

    roles = [
        message.role
        for message in result.messages
    ]

    assert roles == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]