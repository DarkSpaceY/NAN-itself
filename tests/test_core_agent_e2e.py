from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nan_itself.agent.core import CoreAgent
from nan_itself.agent.engine import StepEngine
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.tools.results import text_result
from nan_itself.utils.llm import ToolCall


# ============================================================================
# Generic helpers
# ============================================================================


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
        # is queued (user input, parked reports).
        if self.inbox is not None:
            body = await self.inbox.query(
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
# CoreAgent construction
# ============================================================================


def make_core(
    *,
    llm,
    modules,
    tools,
    skills,
    persona,
):
    runtime = AgentRuntime(
        max_subagent_depth=3
    )

    core = CoreAgent(
        llm=llm,
        modules=modules,
        tools=tools,
        skills=skills,
        persona_source=lambda: persona[
            "value"
        ],
        max_subagent_depth=3,
    )

    # Make the CoreAgent use our deterministic runtime so tests can
    # inspect the dispatch tree directly.
    core.agent_runtime = runtime

    core.engine = StepEngine(
        llm=llm,
        modules=modules,
        tools=tools,
        skills=skills,
        agent_runtime=runtime,
    )

    return (
        core,
        runtime,
    )


# ============================================================================
# Real CoreAgent -> StepEngine -> route -> tool -> final
# ============================================================================


@pytest.mark.asyncio
async def test_core_agent_runs_real_route_tool_final_chain():
    modules = FakeModules(
        initial_value=7
    )

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    persona = {
        "value": "persona-v1"
    }

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

    core, runtime = make_core(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
        persona=persona,
    )

    try:
        # Single-step turn 1: it ends with the invoke_tool call.
        first = await core.run()

        assert first.content is None

        # Turn 2: the observation is rebuilt and the model
        # finalizes with the tool result in history.
        result = await core.run()

        # ----------------------------------------------------------
        # Final Agent result.
        # ----------------------------------------------------------

        assert (
            result.content
            == "42"
        )

        # ----------------------------------------------------------
        # CoreAgent turn boundary.
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
        # final LLM prompt. CoreAgent passes it into StepEngine,
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

        # The second turn delivered the final reply. Turn has
        # no reply field: the reply is the last assistant
        # message in the message flow.
        record = (
            modules.delivered_turns[
                -1
            ]
        )

        assert any(
            getattr(
                message,
                "role",
                None,
            )
            == "assistant"
            and (message.content or "")
            == "42"
            for message in record.messages
        )

        assert (
            record.error
            is None
        )

    finally:
        await runtime.shutdown()


# ============================================================================
# CoreAgent turn boundary: world + persona + skills
# ============================================================================


@pytest.mark.asyncio
async def test_core_agent_refreshes_skill_persona_and_world_each_turn():
    modules = FakeModules(
        initial_value=1
    )

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    persona = {
        "value": "persona-v1"
    }

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

    core, runtime = make_core(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
        persona=persona,
    )

    try:
        first = await core.run()

        persona[
            "value"
        ] = "persona-v2"

        modules.value = 2

        second = await core.run()

        assert (
            first.content
            == "first"
        )

        assert (
            second.content
            == "second"
        )

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
        # The production StepEngine gets it from CoreAgent.
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
        await runtime.shutdown()


# ============================================================================
# Late Subagent report -> next CoreAgent turn
# ============================================================================


@pytest.mark.asyncio
async def test_late_subagent_report_is_parked_and_injected_on_next_turn():
    """
    This is the complete late-report path:

        CoreAgent.run(parent)
            ->
        real StepEngine
            ->
        dispatch_subagent
            ->
        parent finishes before child
            ->
        late report is parked
            ->
        CoreAgent.run(next turn)
            ->
        pending report is drained into seed_reports
            ->
        StepEngine receives the report in the new turn
    """

    modules = FakeModules(
        initial_value=5
    )

    providers = FakeProviderRuntime()

    skills = FakeSkills()

    persona = {
        "value": "persona"
    }

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
            # The first user message of a child turn is exactly the
            # task supplied to spawn().
            # ------------------------------------------------------

            if (
                user_inputs
                and user_inputs[0]
                == "late child"
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

            # A later CoreAgent turn contains the original new input
            # as the first user message. The parked report is another
            # user message containing REPORT_PREFIX.
            has_report = any(
                "[Subagent Report]"
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
            # The report has now been parked and supplied to the
            # StepEngine through seed_reports.
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
                    "[Subagent Report]"
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

    # A real InboxModule is required: the engine parks late
    # subagent reports into it and the next observation drains it.
    modules.inbox = _load_inbox()

    core, runtime = make_core(
        llm=llm,
        modules=modules,
        tools=providers,
        skills=skills,
        persona=persona,
    )

    try:
        # ----------------------------------------------------------
        # Parent turn.
        #
        # Child must start but remain blocked.
        # ----------------------------------------------------------

        first = await core.run()

        # Single-step turn: it ends with the spawn call.
        assert first.content is None

        await asyncio.wait_for(
            child_started.wait(),
            timeout=1.0,
        )

        # At this point the child is still running and no
        # report has been parked yet.
        assert (
            runtime.active_subagent_count
            == 1
        )

        assert not modules.inbox._items

        # ----------------------------------------------------------
        # Second turn: the inbox is still empty, so the parent
        # just wraps up.
        # ----------------------------------------------------------

        second = await core.run()

        assert (
            second.content
            == "parent finished early"
        )

        # ----------------------------------------------------------
        # Now let child complete.
        # ----------------------------------------------------------

        release_child.set()

        # Wait until the background archival task has parked
        # the formatted report into the inbox.
        for _ in range(100):
            if modules.inbox._items:
                break

            await asyncio.sleep(
                0
            )

        assert modules.inbox._items

        assert (
            runtime.active_subagent_count
            == 0
        )

        # ----------------------------------------------------------
        # Next turn: the observation now carries the report.
        # ----------------------------------------------------------

        third = await core.run()

        assert (
            third.content
            == "report received"
        )

        # The report was drained by this turn's observation.
        assert not modules.inbox._items

        # ----------------------------------------------------------
        # Verify report reached the actual StepEngine request.
        # ----------------------------------------------------------

        report_requests = []

        for request in llm.requests:
            has_report = any(
                "[Subagent Report]"
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

        await runtime.shutdown()