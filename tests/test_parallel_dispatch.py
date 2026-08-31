from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nan_itself.agent.engine import StepEngine
from nan_itself.agent.runtime import AgentRuntime
from nan_itself.agent.verbs import (
    DispatchVerb,
    ExecutionState,
)
from nan_itself.tools.view import AgentToolView
from nan_itself.utils.llm import ToolCall


# ============================================================================
# Async helper
# ============================================================================


def run(coro):
    return asyncio.run(coro)


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
# Test infrastructure
# ============================================================================


class FakeModules:
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


class FakeSkills:
    def refresh(self):
        pass

    def catalog(self):
        return ()

    def names(self):
        return ()


class FakeProviders:
    """
    No real provider is needed for dispatch tests.

    AgentToolView still gets constructed by StepEngine, so the object
    exposes the minimum ProviderRuntime-like interface.
    """

    def provider_names(self):
        return ()

    def get_provider(
        self,
        name,
    ):
        return None


# ============================================================================
# Direct DispatchVerb concurrency
# ============================================================================


def test_dispatch_verb_starts_multiple_children_in_parallel():
    """
    Dispatching two independent Subagents must not wait for either child.

    Both workers are blocked behind separate release Events.

        dispatch A
        dispatch B
             ↓
        A started
        B started
             ↓
        release A/B
             ↓
        A completed
        B completed
    """

    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        parent = runtime.create_root(
            world={
                "state": {
                    "value": 1,
                }
            },
            task="parent",
        )

        state = ExecutionState(
            active_skill=None,
            persona="parent",
        )

        started_a = asyncio.Event()
        started_b = asyncio.Event()

        release_a = asyncio.Event()
        release_b = asyncio.Event()

        results = {}

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            if user_input == "task-a":
                started_a.set()
                await release_a.wait()

                results["a"] = context

                return SimpleNamespace(
                    content="result-a"
                )

            if user_input == "task-b":
                started_b.set()
                await release_b.wait()

                results["b"] = context

                return SimpleNamespace(
                    content="result-b"
                )

            raise AssertionError(
                f"Unexpected child task: {user_input}"
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            skills=FakeSkills(),
            execute=execute_child,
        )

        try:
            first = await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "task-a",
                    }
                ),
                context=parent,
                state=state,
                engine=engine,
            )

            second = await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "task-b",
                    }
                ),
                context=parent,
                state=state,
                engine=engine,
            )

            assert (
                "Subagent dispatched."
                in first
            )

            assert (
                "Subagent dispatched."
                in second
            )

            assert (
                len(state.children)
                == 2
            )

            child_a = state.children[0]
            child_b = state.children[1]

            assert (
                child_a.task
                == "task-a"
            )

            assert (
                child_b.task
                == "task-b"
            )

            # ------------------------------------------------------
            # Both children must start before either is released.
            # ------------------------------------------------------

            await asyncio.wait_for(
                asyncio.gather(
                    started_a.wait(),
                    started_b.wait(),
                ),
                timeout=1.0,
            )

            assert (
                runtime.active_subagent_count
                == 2
            )

            assert (
                not child_a.handle.done
            )

            assert (
                not child_b.handle.done
            )

            # ------------------------------------------------------
            # Release both independently.
            # ------------------------------------------------------

            release_a.set()
            release_b.set()

            result_a, result_b = (
                await asyncio.gather(
                    child_a.handle.wait(),
                    child_b.handle.wait(),
                )
            )

            assert (
                result_a.content
                == "result-a"
            )

            assert (
                result_b.content
                == "result-b"
            )

            assert (
                runtime.active_subagent_count
                == 0
            )

            # ------------------------------------------------------
            # Distinct identities.
            # ------------------------------------------------------

            assert (
                results["a"].agent_hash
                != results["b"].agent_hash
            )

            assert (
                results["a"].parent_hash
                == parent.agent_hash
            )

            assert (
                results["b"].parent_hash
                == parent.agent_hash
            )

            # Same dispatch-tree world.
            assert (
                results["a"].world
                is parent.world
            )

            assert (
                results["b"].world
                is parent.world
            )

        finally:
            await runtime.shutdown()

    run(
        scenario()
    )


# ============================================================================
# Multiple dispatches in one actual StepEngine response
# ============================================================================


def test_step_engine_multiple_dispatch_calls_start_parallel_children_and_collect_both_reports():
    """
    Real StepEngine integration.

    The Parent's first model response contains TWO tool calls:

        dispatch_subagent("task-a")
        dispatch_subagent("task-b")

    Both are handled by the same StepEngine iteration.

    The next parent model request waits until both children complete.
    The parent then executes sleep(0), yielding to the next StepEngine
    boundary.

    At that next boundary:

        collect_finished_children()

    must inject both reports before the final Parent response.
    """

    class ParallelLLM:
        model = "fake-model"
        provider = "fake-provider"

        def __init__(self):
            self.requests = []

            self.parent_dispatch_response_used = (
                False
            )

            self.parent_sync_response_used = (
                False
            )

            self.child_completions = 0

            self.children_finished = (
                asyncio.Event()
            )

        async def generate_complete(
            self,
            request,
        ):
            self.requests.append(
                request
            )

            # Every real LLM request has an async boundary.
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

            if not user_inputs:
                raise AssertionError(
                    "LLM request contained no user messages."
                )

            first_user_input = (
                user_inputs[0]
            )

            # ======================================================
            # CHILD A
            # ======================================================

            if (
                first_user_input
                == "task-a"
            ):
                self.child_completions += 1

                if (
                    self.child_completions
                    == 2
                ):
                    self.children_finished.set()

                return make_response(
                    content="result-a"
                )

            # ======================================================
            # CHILD B
            # ======================================================

            if (
                first_user_input
                == "task-b"
            ):
                self.child_completions += 1

                if (
                    self.child_completions
                    == 2
                ):
                    self.children_finished.set()

                return make_response(
                    content="result-b"
                )

            # ======================================================
            # PARENT
            # ======================================================

            if (
                first_user_input
                != "parent task"
            ):
                raise AssertionError(
                    "Unexpected initial user input: "
                    f"{first_user_input!r}"
                )

            has_report = any(
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

            # ------------------------------------------------------
            # Final Parent step:
            # both reports must already have been injected.
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
                    and "[Subagent Report]"
                    in (
                        message.content
                        or ""
                    )
                )

                assert (
                    "result-a"
                    in report_text
                )

                assert (
                    "result-b"
                    in report_text
                )

                return make_response(
                    content=(
                        "received both child results"
                    )
                )

            # ------------------------------------------------------
            # Parent first step:
            # dispatch BOTH children in one model response.
            # ------------------------------------------------------

            if not self.parent_dispatch_response_used:
                self.parent_dispatch_response_used = True

                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "dispatch-a",
                            "dispatch_subagent",
                            {
                                "task": "task-a",
                            },
                        ),
                        make_tool_call(
                            "dispatch-b",
                            "dispatch_subagent",
                            {
                                "task": "task-b",
                            },
                        ),
                    ]
                )

            # ------------------------------------------------------
            # Parent second model step:
            #
            # Wait until both children have actually completed.
            #
            # The Parent then executes sleep(0), so the following
            # StepEngine iteration is where report collection happens.
            # ------------------------------------------------------

            if not self.parent_sync_response_used:
                self.parent_sync_response_used = True

                try:
                    await asyncio.wait_for(
                        self.children_finished.wait(),
                        timeout=1.0,
                    )

                except asyncio.TimeoutError as exc:
                    raise AssertionError(
                        "Both Subagents did not finish before "
                        "the Parent's synchronization step."
                    ) from exc

                return make_response(
                    tool_calls=[
                        make_tool_call(
                            "yield-1",
                            "sleep",
                            {
                                "seconds": 0,
                            },
                        )
                    ]
                )

            raise AssertionError(
                "Parent requested another model response "
                "without receiving the child reports."
            )

    async def scenario():
        llm = ParallelLLM()

        modules = FakeModules()
        providers = FakeProviders()
        skills = FakeSkills()

        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        engine = StepEngine(
            llm=llm,
            modules=modules,
            providers=providers,
            skills=skills,
            agent_runtime=runtime,
        )

        parent = runtime.create_root(
            world={
                "state": {
                    "value": 1,
                }
            },
            task="parent task",
        )

        try:
            result = await engine.execute(
                context=parent,
                user_input="parent task",
                persona="parent persona",
            )

            assert (
                result.content
                == "received both child results"
            )

            # ------------------------------------------------------
            # Both children really executed.
            # ------------------------------------------------------

            assert (
                llm.child_completions
                == 2
            )

            children = [
                agent
                for agent in runtime.agents()
                if (
                    agent.parent_hash
                    == parent.agent_hash
                )
            ]

            assert (
                len(children)
                == 2
            )

            assert {
                child.task
                for child in children
            } == {
                "task-a",
                "task-b",
            }

            assert all(
                child.depth == 1
                for child in children
            )

            # ------------------------------------------------------
            # Same world across both children.
            # ------------------------------------------------------

            assert all(
                child.world
                is parent.world
                for child in children
            )

            # ------------------------------------------------------
            # Parent made exactly one initial multi-dispatch
            # response and one synchronization response.
            # ------------------------------------------------------

            assert (
                llm.parent_dispatch_response_used
                is True
            )

            assert (
                llm.parent_sync_response_used
                is True
            )

            # ------------------------------------------------------
            # Both reports must have been injected into a model
            # request before the final response.
            # ------------------------------------------------------

            report_requests = []

            for request in llm.requests:
                has_report = any(
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

                if has_report:
                    report_requests.append(
                        request
                    )

            assert (
                report_requests
            )

            final_report_text = "\n".join(
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
                "task: task-a"
                in final_report_text
            )

            assert (
                "task: task-b"
                in final_report_text
            )

            assert (
                "result-a"
                in final_report_text
            )

            assert (
                "result-b"
                in final_report_text
            )

            # No Subagent remains live after the reports have been
            # collected.
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
# Parallel children remain independently awaitable
# ============================================================================


def test_one_parallel_child_failure_does_not_destroy_other_child():
    """
    Child executions are independent Tasks.

    If one fails, the other remains independently awaitable.
    """

    async def scenario():
        runtime = AgentRuntime(
            max_subagent_depth=3
        )

        parent = runtime.create_root(
            world={
                "value": 1,
            }
        )

        state = ExecutionState(
            active_skill=None,
            persona="parent",
        )

        sibling_started = asyncio.Event()
        release_sibling = asyncio.Event()

        async def execute_child(
            *,
            context,
            user_input,
            persona,
        ):
            if user_input == "failing-child":
                await asyncio.sleep(0)

                raise RuntimeError(
                    "child failed"
                )

            if user_input == "healthy-child":
                sibling_started.set()

                await release_sibling.wait()

                return SimpleNamespace(
                    content="healthy result"
                )

            raise AssertionError(
                user_input
            )

        engine = SimpleNamespace(
            agent_runtime=runtime,
            skills=FakeSkills(),
            execute=execute_child,
        )

        try:
            await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "failing-child",
                    }
                ),
                context=parent,
                state=state,
                engine=engine,
            )

            await DispatchVerb().execute(
                call=SimpleNamespace(
                    arguments={
                        "task": "healthy-child",
                    }
                ),
                context=parent,
                state=state,
                engine=engine,
            )

            failing = (
                state.children[0]
            )

            healthy = (
                state.children[1]
            )

            await asyncio.wait_for(
                sibling_started.wait(),
                timeout=1.0,
            )

            with pytest.raises(
                RuntimeError,
                match="child failed",
            ):
                await failing.handle.wait()

            assert (
                not healthy.handle.done
            )

            release_sibling.set()

            healthy_result = (
                await healthy.handle.wait()
            )

            assert (
                healthy_result.content
                == "healthy result"
            )

            assert (
                runtime.active_subagent_count
                == 0
            )

        finally:
            release_sibling.set()

            await runtime.shutdown()

    run(
        scenario()
    )