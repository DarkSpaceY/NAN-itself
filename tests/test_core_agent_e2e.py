from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.agent.core import Agent
from nan_itself.agent.engine import StepEngine
from nan_itself.agent.prompts import render_turn
from nan_itself.tools.results import text_result
from nan_itself.utils.llm import ToolCall


# ============================================================================
# Generic helpers
# ============================================================================


def write_persona(
    tmp_path,
    text="persona",
):
    """
    Every Agent re-reads its persona file at each turn boundary, so
    each test needs a real file to point it at.
    """
    path = tmp_path / "persona.md"

    path.write_text(text, encoding="utf-8")

    return path


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


def reply_text(turn):
    """
    The turn's reply text (empty when the turn never produced
    one).
    """
    return turn.reply or ""


def turn_messages(turn):
    """
    The turn's message run, rendered by the prompt assembly.
    """
    return list(render_turn(turn))


def tool_messages(turn):
    """
    The turn's tool-result messages, rendered by the prompt
    assembly.
    """
    return [
        message
        for message in render_turn(turn)
        if getattr(message, "role", None) == "tool"
    ]


# ============================================================================
# Fake Module runtime
# ============================================================================


def _load_inbox():
    """
    Load the real InboxModule from its hot-reload file.
    """
    from nan_itself.modules.loading import import_module_class

    path = (
        Path(__file__)
        .resolve()
        .parents[1]
        / "builtin"
        / "modules"
        / "inbox.py"
    )

    cls, _, _ = import_module_class(path)

    return cls()


class FakeModules:
    def __init__(
        self,
        initial_value=1,
    ):
        self.value = initial_value

        self.snapshot_calls = 0
        self.snapshot_values = []

        self.query_snapshot_calls = 0
        self.query_snapshots = []
        self.query_turns = []

        self.delivered_turns = []

        self.inbox = None

    def get(
        self,
        module_id: str,
    ):
        return self.inbox

    def snapshot(self):
        self.snapshot_calls += 1

        snapshot = {
            "state": {
                "value": self.value,
            }
        }

        self.snapshot_values.append(
            snapshot
        )

        return snapshot

    async def query_snapshot(
        self,
        turn,
        **kwargs,
    ):
        self.query_snapshot_calls += 1

        self.query_turns.append(
            turn
        )

        self.query_snapshots.append(
            turn.world
        )

        ambient = []

        # Real inbox module: the observation carries whatever
        # is queued (user input).
        if self.inbox is not None:
            body = await self.inbox.ask(
                turn
            )

            if body:
                ambient.append(
                    body
                )

        return ambient

    def deliver_turn(
        self,
        record,
    ):
        self.delivered_turns.append(
            record
        )


# ============================================================================
# Fake Skill runtime
# ============================================================================


class FakeSkills:
    def __init__(self):
        self.refresh_calls = 0

    def refresh(self):
        self.refresh_calls += 1

    def catalog(self):
        return ()

    def names(self):
        return ()


# ============================================================================
# Fake Provider runtime
# ============================================================================


class FakeProviderRuntime:
    """
    Deterministic ProviderRuntime exposing the surface consumed by
    the tool verbs: resolve_tool() and call_tool().
    """

    def __init__(self):
        self.providers = {
            "calc": SimpleNamespace(
                spec=SimpleNamespace(
                    name="calc",
                ),
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

        self.calls = []
        self.refresh_calls = []

    def provider_names(self):
        return tuple(
            self.providers
        )

    async def resolve_tool(
        self,
        name,
    ):
        """
        Resolve a 'provider/tool' composite name.
        """
        provider_name, _, tool_name = (
            name.partition("/")
        )

        provider = self.providers.get(
            provider_name
        )

        if provider is None or not tool_name:
            return None

        tool = provider.tools.get(
            tool_name
        )

        if tool is None:
            return None

        return (
            provider,
            tool,
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
        self.refresh_calls.append(
            name
        )

    async def call_tool(
        self,
        provider_name,
        tool_name,
        arguments,
    ):
        self.calls.append(
            (
                provider_name,
                tool_name,
                dict(arguments),
            )
        )

        if (
            provider_name == "calc"
            and tool_name == "add"
        ):
            return text_result(
                str(
                    arguments["x"]
                    + arguments["y"]
                )
            )

        raise AssertionError(
            "Unexpected tool call: "
            f"{provider_name}.{tool_name}"
        )

    async def start(self):
        pass

    async def stop(self):
        pass


# ============================================================================
# Fake LLM
# ============================================================================


class QueueLLM:
    """
    Simple ordered LLM response source.

    Every request is retained so the test can inspect the actual
    messages and tool definitions produced by StepEngine.
    """

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

        await asyncio.sleep(0)

        if not self.responses:
            raise AssertionError(
                "LLM received more requests than expected."
            )

        return self.responses.pop(0)


# ============================================================================
# Agent construction
# ============================================================================


def make_agent(
    *,
    llm,
    modules,
    tools,
    skills,
):
    return Agent(
        engine=StepEngine(
            llm=llm,
        ),
        modules=modules,
        tools=tools,
        skills=skills,
        max_subagent_depth=3,
    )


async def cancel_children(agent):
    """
    Stop and cancel any live child agent. These tests drive
    Agent.run() directly, so there is no loop task to await.
    """
    for child in list(agent.children):
        child.request_stop()

        await Agent._cancel_and_suppress(
            child._task
        )


# ============================================================================
# Real Agent -> StepEngine -> route -> tool -> final
# ============================================================================


@pytest.mark.asyncio
async def test_agent_runs_real_route_tool_final_chain(tmp_path):
    modules = FakeModules(
        initial_value=7
    )

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    write_persona(
        tmp_path,
        "persona-v1",
    )

    llm = QueueLLM(
        [
            # ------------------------------------------------------
            # Step 1:
            # invoke the provider tool directly
            # ------------------------------------------------------

            make_response(
                tool_calls=[
                    make_tool_call(
                        "add-1",
                        "invoke_tool",
                        {
                            "name": "calc/add",
                            "arguments": {
                                "x": 20,
                                "y": 22,
                            },
                        },
                    )
                ]
            ),

            # ------------------------------------------------------
            # Step 2:
            # final
            # ------------------------------------------------------

            make_response(
                content="42"
            ),
        ]
    )

    agent = make_agent(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
    )

    try:
        # Single-step turn 1: it runs the invoke_tool call and
        # its result is written back onto the turn.
        first = await agent.run()

        first_rendered = turn_messages(first)

        assert first_rendered[-1].role == "tool"

        assert first_rendered[-1].content

        # Turn 2: the observation is rebuilt and the model
        # finalizes with the tool result in history.
        result = await agent.run()

        # ----------------------------------------------------------
        # Final reply of the turn.
        # ----------------------------------------------------------

        assert reply_text(result) == "42"

        # ----------------------------------------------------------
        # Agent turn boundary.
        # ----------------------------------------------------------

        assert (
            modules.snapshot_calls
            == 2
        )

        assert (
            skills.refresh_calls
            == 2
        )

        # ----------------------------------------------------------
        # World snapshot crosses into Turn / StepEngine.
        #
        # Do not assume the world is directly rendered into the
        # final LLM prompt. The Agent passes it into StepEngine,
        # which passes it to modules.query_snapshot().
        # ----------------------------------------------------------

        assert (
            modules.query_snapshot_calls
            == 2
        )

        assert (
            modules.query_snapshots[0][
                "state"
            ]["value"]
            == 7
        )

        assert (
            modules.query_turns[0].world[
                "state"
            ]["value"]
            == 7
        )

        # ----------------------------------------------------------
        # Persona is rendered into the system message.
        # ----------------------------------------------------------

        first_request = (
            llm.requests[0]
        )

        assert (
            "persona-v1"
            in first_request.messages[0].content
        )

        # ----------------------------------------------------------
        # Tool exposure: the 8 verbs are always visible; provider
        # tools are never exposed directly (invoke_tool only).
        # ----------------------------------------------------------

        first_tools = {
            tool.name
            for tool in first_request.tools
        }

        assert (
            "invoke_tool"
            in first_tools
        )

        assert (
            "add"
            not in first_tools
        )

        # ----------------------------------------------------------
        # Every request sees the same full verb set.
        # ----------------------------------------------------------

        second_request = (
            llm.requests[1]
        )

        second_tools = {
            tool.name
            for tool in second_request.tools
        }

        assert (
            "invoke_tool"
            in second_tools
        )

        assert (
            "add"
            not in second_tools
        )

        # ----------------------------------------------------------
        # Real provider call happened.
        # ----------------------------------------------------------

        assert (
            providers.calls
            == [
                (
                    "calc",
                    "add",
                    {
                        "x": 20,
                        "y": 22,
                    },
                )
            ]
        )

        # ----------------------------------------------------------
        # Tool result reached the model.
        #
        # The same historical tool message is carried into later
        # requests, so deduplicate by tool_call_id.
        # ----------------------------------------------------------

        tool_messages_by_id = {}

        for request in llm.requests:
            for message in request.messages:
                if (
                    getattr(
                        message,
                        "role",
                        None,
                    )
                    != "tool"
                ):
                    continue

                call_id = getattr(
                    message,
                    "tool_call_id",
                    None,
                )

                if call_id is None:
                    continue

                tool_messages_by_id[
                    call_id
                ] = message

        tool_messages = list(
            tool_messages_by_id.values()
        )

        assert (
            len(tool_messages)
            == 1
        )

        assert any(
            '"text":"42"'
            in (
                message.content
                or ""
            )
            for message in tool_messages
        )

        # ----------------------------------------------------------
        # Completed turn is delivered to Modules.
        # ----------------------------------------------------------

        assert (
            len(
                modules.delivered_turns
            )
            == 2
        )

        # The second turn delivered the final reply, recorded
        # structurally on the turn's reply field.
        record = (
            modules.delivered_turns[
                -1
            ]
        )

        assert (
            record.reply
            == "42"
        )

        assert (
            record.error
            is None
        )

    finally:
        await cancel_children(agent)


# ============================================================================
# Agent turn boundary: world + persona + skills
# ============================================================================


@pytest.mark.asyncio
async def test_agent_refreshes_skill_persona_and_world_each_turn(tmp_path):
    modules = FakeModules(
        initial_value=1
    )

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    write_persona(
        tmp_path,
        "persona-v1",
    )

    llm = QueueLLM(
        [
            make_response(
                content="first"
            ),
            make_response(
                content="second"
            ),
        ]
    )

    agent = make_agent(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
    )

    try:
        first = await agent.run()

        # Persona edits hot-reload: the file is re-read per turn.
        write_persona(
            tmp_path,
            "persona-v2",
        )

        modules.value = 2

        second = await agent.run()

        assert reply_text(first) == "first"

        assert reply_text(second) == "second"

        # ----------------------------------------------------------
        # Exactly one refresh/snapshot per turn.
        # ----------------------------------------------------------

        assert (
            skills.refresh_calls
            == 2
        )

        assert (
            modules.snapshot_calls
            == 2
        )

        # ----------------------------------------------------------
        # Turn 1 world remains v1.
        # Turn 2 sees v2.
        # ----------------------------------------------------------

        first_context_world = (
            modules
            .snapshot_values[0]
        )

        second_context_world = (
            modules
            .snapshot_values[1]
        )

        assert (
            first_context_world[
                "state"
            ]["value"]
            == 1
        )

        assert (
            second_context_world[
                "state"
            ]["value"]
            == 2
        )

        # ----------------------------------------------------------
        # Persona is injected independently on every turn.
        # The production StepEngine gets it from the Agent's Turn.
        # We verify it from the actual system prompt.
        # ----------------------------------------------------------

        first_persona = (
            llm.requests[0]
            .messages[0]
            .content
        )

        second_persona = (
            llm.requests[1]
            .messages[0]
            .content
        )

        assert (
            "persona-v1"
            in first_persona
        )

        assert (
            "persona-v2"
            in second_persona
        )

        # ----------------------------------------------------------
        # History accumulates across turns: the second request
        # replays the first turn's messages (cleared only when
        # the history character limit is exceeded).
        # ----------------------------------------------------------

        first_request_users = [
            message.content
            for message in llm.requests[0].messages
            if getattr(
                message,
                "role",
                None,
            )
            == "user"
        ]

        second_request_users = [
            message.content
            for message in llm.requests[1].messages
            if getattr(
                message,
                "role",
                None,
            )
            == "user"
        ]

        assert (
            first_request_users
            == [""]
        )

        assert (
            second_request_users
            == ["", ""]
        )

        # ----------------------------------------------------------
        # The first turn's assistant reply is replayed in the
        # second request.
        # ----------------------------------------------------------

        assert any(
            getattr(
                message,
                "role",
                None,
            )
            == "assistant"
            and (message.content or "")
            == "first"
            for message in llm.requests[
                1
            ].messages
        )

    finally:
        await cancel_children(agent)


# ============================================================================
# Late subagent report -> next Agent turn
# ============================================================================


@pytest.mark.asyncio
async def test_late_subagent_report_is_harvested_into_the_next_turn(tmp_path):
    """
    This is the complete late-report path:

        Agent.run(root)
            ->
        real StepEngine
            ->
        SpawnVerb -> Agent.spawn
            ->
        the root's turn ends before the child reports
            ->
        the child finishes; its report settles on the child
            ->
        Agent.run(root) harvests it
            ->
        the report is folded into the root's own observation
            (the inbox is never involved)
    """

    modules = FakeModules(
        initial_value=5
    )

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    write_persona(
        tmp_path,
        "persona",
    )

    child_started = asyncio.Event()
    release_child = asyncio.Event()

    class LateReportLLM:
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

            await asyncio.sleep(0)

            user_inputs = [
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

            # ------------------------------------------------------
            # Child execution.
            #
            # The first user message of a child turn is the task
            # supplied to spawn(), wrapped in <task>.
            # ------------------------------------------------------

            if (
                user_inputs
                and "<task>" in user_inputs[0]
                and "late child" in user_inputs[0]
                and "<subagent_report>" not in user_inputs[0]
            ):
                self.child_calls += 1

                child_started.set()

                # First child turn: blocked, then replies with
                # plain text -- which does NOT end a subagent
                # task anymore; the loop starts a new turn.
                if self.child_calls == 1:
                    await release_child.wait()

                    return make_response(
                        content="late child result"
                    )

                # Second child turn: submit the report via
                # the finish tool; only this ends the task.
                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "finish-1",
                            "finish",
                            {
                                "report": "late child result",
                            },
                        )
                    ]
                )

            # ------------------------------------------------------
            # Parent execution.
            # ------------------------------------------------------

            if not user_inputs:
                raise AssertionError(
                    "Parent request has no user messages."
                )

            # A later root turn carries the harvested child report as
            # a user message; the report is recognized by the
            # <subagent_report> frame.
            has_report = any(
                "<subagent_report>"
                in (
                    message.content
                    or ""
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

            # ------------------------------------------------------
            # First parent turn.
            #
            # It dispatches the child and then immediately finalizes.
            # Child is intentionally still blocked.
            # ------------------------------------------------------

            if (
                not has_report
                and self.parent_steps
                == 0
            ):
                self.parent_steps += 1

                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "spawn-1",
                            "spawn",
                            {
                                "task": "late child",
                            },
                        )
                    ]
                )

            if (
                not has_report
                and self.parent_steps
                == 1
            ):
                self.parent_steps += 1

                return make_response(
                    content="parent finished early"
                )

            # ------------------------------------------------------
            # Next parent turn.
            #
            # The report has now been harvested onto the
            # parent's turn and reaches the StepEngine rendered
            # into the observation.
            # ------------------------------------------------------

            if has_report:
                report_text = "\n".join(
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
                )

                assert (
                    "<subagent_report>"
                    in report_text
                )

                assert (
                    "late child"
                    in report_text
                )

                assert (
                    "late child result"
                    in report_text
                )

                return make_response(
                    content="report received"
                )

            raise AssertionError(
                f"Unexpected parent state: "
                f"steps={self.parent_steps} "
                f"child={self.child_calls} "
                f"has_report={has_report} "
                f"inputs={user_inputs!r}"
            )

    llm = LateReportLLM()

    # A real InboxModule is running: the report must NOT travel
    # through it, so its emptiness is part of the assertion.
    modules.inbox = _load_inbox()

    agent = make_agent(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
    )

    try:
        # ----------------------------------------------------------
        # Root turn.
        #
        # Child must start but remain blocked.
        # ----------------------------------------------------------

        first = await agent.run()

        # Single-step turn: it runs the spawn call and its
        # result is written back onto the turn.
        first_rendered = turn_messages(first)

        assert first_rendered[-1].role == "tool"

        assert (
            "Subagent spawned"
            in first_rendered[-1].content
        )

        await asyncio.wait_for(
            child_started.wait(),
            timeout=1.0,
        )

        # At this point the child is still running and still
        # listed; no report exists yet.
        assert len(agent.children) == 1

        assert not modules.inbox._items

        # ----------------------------------------------------------
        # Second turn: no report yet, so the root just wraps up.
        # ----------------------------------------------------------

        second = await agent.run()

        assert (
            reply_text(second)
            == "parent finished early"
        )

        # ----------------------------------------------------------
        # Now let the child complete.
        # ----------------------------------------------------------

        release_child.set()

        child = agent.children[0]

        # Wait until the child's report has settled on the child.
        for _ in range(200):
            if child.done and child.report is not None:
                break

            await asyncio.sleep(
                0
            )

        assert child.report is not None

        assert child.report.status == "completed"

        # The report never travelled through the inbox.
        assert not modules.inbox._items

        # ----------------------------------------------------------
        # Next turn: the harvest folds the report into this
        # turn's observation.
        # ----------------------------------------------------------

        third = await agent.run()

        assert (
            reply_text(third)
            == "report received"
        )

        # The harvested child left the tree.
        assert agent.children == []

        # ----------------------------------------------------------
        # Verify report reached the actual StepEngine request.
        # ----------------------------------------------------------

        report_requests = []

        for request in llm.requests:
            has_report = any(
                "<subagent_report>"
                in (
                    message.content
                    or ""
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

            if has_report:
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
        )

        assert (
            "task: late child"
            in report_text
        )

        assert (
            "status: completed"
            in report_text
        )

        assert (
            "late child result"
            in report_text
        )

    finally:
        release_child.set()

        await cancel_children(agent)


# ============================================================================
# Agent.run returns the completed Turn
# ============================================================================


@pytest.mark.asyncio
async def test_run_returns_the_turn_whose_reply_is_the_model_text(tmp_path):
    """
    Agent.run() hands back the completed Turn: the model's reply
    is recorded on the turn's reply field, and that same Turn is
    what reaches the Modules.
    """
    modules = FakeModules()

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    write_persona(
        tmp_path,
        "persona",
    )

    llm = QueueLLM(
        [
            make_response(
                content="the answer"
            )
        ]
    )

    agent = make_agent(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
    )

    try:
        turn = await agent.run()

        assert turn is agent.last_turn

        assert reply_text(turn) == "the answer"

        rendered = turn_messages(turn)

        assert rendered[-1].role == "assistant"

        assert (
            rendered[-1].content
            == "the answer"
        )

        # The same Turn was delivered to the Modules.
        assert (
            modules.delivered_turns[-1]
            is turn
        )

    finally:
        await cancel_children(agent)


@pytest.mark.asyncio
async def test_tool_results_are_written_back_to_the_same_turn(tmp_path):
    """
    A turn that ends with tool calls runs its verbs and writes
    each result back positionally onto THAT SAME Turn, paired
    with the assistant's calls; rendering turns them into tool
    messages right after the assistant message.
    """
    modules = FakeModules()

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    write_persona(
        tmp_path,
        "persona",
    )

    llm = QueueLLM(
        [
            make_response(
                tool_calls=[
                    make_tool_call(
                        "add-1",
                        "invoke_tool",
                        {
                            "name": "calc/add",
                            "arguments": {
                                "x": 2,
                                "y": 3,
                            },
                        },
                    )
                ]
            ),
        ]
    )

    agent = make_agent(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
    )

    try:
        turn = await agent.run()

        # observation + assistant(tool_calls) + tool, rendered
        rendered = turn_messages(turn)

        assert len(rendered) == 3

        assert rendered[1].role == "assistant"

        assert [
            call.name
            for call in rendered[1].tool_calls
        ] == ["invoke_tool"]

        assert rendered[2].role == "tool"

        assert (
            rendered[2].tool_call_id
            == "add-1"
        )

        assert (
            '"text":"5"'
            in rendered[2].content
        )

        # The provider tool really ran.
        assert providers.calls == [
            (
                "calc",
                "add",
                {
                    "x": 2,
                    "y": 3,
                },
            )
        ]

        # The delivered Turn carries the results too.
        assert (
            tool_messages(
                modules.delivered_turns[-1]
            )
            == [rendered[2]]
        )

    finally:
        await cancel_children(agent)


# ============================================================================
# Failure visibility on the delivered Turn
# ============================================================================


class FailingEngine:
    """
    StepEngine double whose model call always fails: the Turn that
    reaches the Modules must still show the failure.
    """

    def __init__(self, exc):
        self.exc = exc

    async def step(self, turn):
        raise self.exc


def make_failing_agent(
    *,
    tmp_path,
    exc,
):
    modules = FakeModules()

    write_persona(
        tmp_path,
        "persona",
    )

    agent = make_agent(
        llm=QueueLLM([]),
        modules=modules,
        tools=FakeProviderRuntime(),
        skills=FakeSkills(),
    )

    agent.engine = FailingEngine(exc)

    return agent, modules


@pytest.mark.asyncio
async def test_step_failure_is_visible_on_the_delivered_turn(
    tmp_path,
):
    agent, modules = make_failing_agent(
        tmp_path=tmp_path,
        exc=RuntimeError("boom"),
    )

    try:
        with pytest.raises(RuntimeError):
            await agent.run()

        record = modules.delivered_turns[-1]

        assert (
            record.error
            == "RuntimeError: boom"
        )

        assert record.ended_at is not None

        # A failed turn never becomes the next turn's history
        # source: last_turn only advances on success.
        assert agent.last_turn is None

    finally:
        await cancel_children(agent)


@pytest.mark.asyncio
async def test_cancelled_step_is_visible_on_the_delivered_turn(
    tmp_path,
):
    agent, modules = make_failing_agent(
        tmp_path=tmp_path,
        exc=asyncio.CancelledError(),
    )

    try:
        with pytest.raises(asyncio.CancelledError):
            await agent.run()

        record = modules.delivered_turns[-1]

        assert record.error == "cancelled"

        assert record.ended_at is not None

    finally:
        await cancel_children(agent)