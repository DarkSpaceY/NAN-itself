"""
Global sink: the process-wide `sink` singleton replaces
StreamSink reference threading.

Verified here:

    - every emitted event carries the envelope on the top level
      (`t` / `id` / optional `boot_id`) and an always-object
      `content` payload
    - the sink is a pure transport layer: `content` is passed
      through verbatim, identity keys included -- the caller
      supplies `agent_hash` / `parent_hash` / `depth` when it has
      an agent context and omits them entirely otherwise
    - agent emitters carry their own identity explicitly at every
      call site, so a spawned subagent's events never ride the
      root's identity
    - the user_input echo carries text / mid inside content, with
      boot_id on the envelope (no longer inside content)
    - without an attached bus emit() is a no-op
    - reset() detaches the bus and clears the boot_id
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nan_itself.agent.core import Agent
from nan_itself.agent.engine import StepEngine
from nan_itself.agent.prompts import render_turn
from nan_itself.events import (
    EventBus,
    sink,
)
from nan_itself.modules.model import Turn


# ============================================================================
# Hygiene: the singleton must never leak between tests
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_sink():
    yield

    sink.reset()


# ============================================================================
# Identity travels in content
# ============================================================================


def test_caller_supplied_identity_lands_in_content():
    bus = EventBus()

    sink.attach(bus)

    record_id = sink.emit(
        "record_started",
        content={
            "kind": "agent",
            "name": "some task",
            "agent_hash": "hash-a",
            "parent_hash": "hash-parent",
            "depth": 2,
        },
    )

    assert record_id

    event = bus.history()[-1]

    # Envelope on the top level; content is always an object.
    assert event["t"] == "record_started"

    assert event["id"] == record_id

    assert isinstance(event["content"], dict)

    # The caller-supplied identity rides through verbatim; the
    # sink neither adds nor rewrites fields.
    assert event["content"]["agent_hash"] == "hash-a"

    assert (
        event["content"]["parent_hash"]
        == "hash-parent"
    )

    assert event["content"]["depth"] == 2

    assert event["content"]["kind"] == "agent"


def test_emission_without_identity_omits_the_keys():
    bus = EventBus()

    sink.attach(bus)

    sink.emit(
        "status",
        content={"state": "working"},
    )

    event = bus.history()[-1]

    assert event["t"] == "status"

    assert event["content"] == {"state": "working"}

    # The caller supplied no identity: the three keys are absent,
    # not null.
    assert "agent_hash" not in event["content"]

    assert "parent_hash" not in event["content"]

    assert "depth" not in event["content"]


def test_ids_stay_unique_across_emissions():
    bus = EventBus()

    sink.attach(bus)

    first = sink.emit(
        "record_started",
        content={
            "kind": "agent",
            "name": "one",
        },
    )

    second = sink.emit(
        "record_started",
        content={
            "kind": "agent",
            "name": "two",
        },
    )

    assert first

    assert second

    assert first != second


# ============================================================================
# No-op without a bus
# ============================================================================


def test_unattached_sink_is_a_noop():
    assert not sink.attached

    assert (
        sink.emit(
            "record_started",
            content={
                "kind": "agent",
                "name": "x",
            },
        )
        == ""
    )

    sink.emit(
        "status",
        content={"state": "working"},
    )

    sink.emit(
        "output_done",
        id="whatever",
        content={},
    )


# ============================================================================
# reset()
# ============================================================================


def test_reset_detaches_bus_and_boot_id():
    bus = EventBus()

    sink.attach(bus, boot_id="boot-x")

    sink.reset()

    assert not sink.attached

    # A fresh attach starts clean: no stale boot_id leaks into
    # events.
    sink.attach(bus)

    sink.emit(
        "status",
        content={"state": "working"},
    )

    event = bus.history()[-1]

    assert "boot_id" not in event


# ============================================================================
# Agent integration: call sites carry their own identity
# ============================================================================


class StreamLLM:
    """
    Streaming double: one text chunk, then done. Retains the
    request for assertions and signals when it was called.
    """

    model = "fake-model"

    provider = "openai"

    def __init__(self):
        self.requests = []
        self.called = asyncio.Event()

    async def generate(self, request):
        self.requests.append(request)

        self.called.set()

        yield SimpleNamespace(
            kind="text",
            text="hi",
            tool_call=None,
        )

        yield SimpleNamespace(
            kind="done",
            usage=None,
            finish_reason="stop",
            tool_call=None,
        )


class RecordingModules:
    def __init__(self):
        self.delivered = []

    def snapshot(self):
        return {}

    async def query_snapshot(
        self,
        turn,
        *,
        on_start=None,
        on_result=None,
    ):
        if on_start is not None:
            on_start("state")

        if on_result is not None:
            on_result(
                "state",
                "value: 1",
                0.0,
                False,
            )

        return ["value: 1"]

    def deliver_turn(self, turn):
        self.delivered.append(turn)


def test_agent_run_stamps_events_with_its_own_identity():
    """
    A spawned (depth-1) agent's whole turn -- module query records
    and the LLM output stream -- must reach the bus stamped with
    that agent's own hash, parent hash and depth. Every emit call
    site carries them explicitly from the emitting agent, so the
    child's events never ride the root's identity.
    """

    async def scenario():
        bus = EventBus()

        sink.attach(bus)

        modules = RecordingModules()

        llm = StreamLLM()

        skills = SimpleNamespace(
            refresh=lambda: None,
        )

        root = Agent(
            engine=StepEngine(
                llm=llm,
            ),
            modules=modules,
            tools=SimpleNamespace(),
            skills=skills,
            autonomous_interval=10.0,
        )

        child = root.spawn("child task")

        await asyncio.wait_for(
            llm.called.wait(),
            timeout=1.0,
        )

        # Let the in-flight turn finish emitting.
        for _ in range(100):
            if any(
                event["t"] == "output_done"
                for event in bus.history()
            ):
                break

            await asyncio.sleep(0)

        child._task.cancel()

        await asyncio.gather(
            child._task,
            return_exceptions=True,
        )

        assert child.depth == 1

        assert len(modules.delivered) == 1

        events = bus.history()

        assert events

        # Every event of the child's turn carries the child's own
        # identity inside content -- never the root's.
        for event in events:
            assert (
                event["content"]["agent_hash"]
                == child.agent_hash
            )

            assert (
                event["content"]["parent_hash"]
                == root.agent_hash
            )

            assert event["content"]["depth"] == 1

        kinds = [
            event["t"]
            for event in events
        ]

        assert "record_started" in kinds

        assert "output_started" in kinds

        assert "output_done" in kinds

    asyncio.run(scenario())


def test_engine_step_without_bus_uses_complete_generation():
    """
    Without an attached bus the engine must not stream: it calls
    generate_complete() exactly like the pre-sink era, which the
    non-streaming test doubles rely on.
    """

    class CompleteLLM:
        model = "fake-model"

        provider = "openai"

        def __init__(self):
            self.requests = []

            self.streamed = False

        async def generate_complete(
            self,
            request,
        ):
            self.requests.append(request)

            return SimpleNamespace(
                content="hi",
                tool_calls=[],
                model="fake-model",
                usage=None,
                provider="openai",
                finish_reason="stop",
            )

        async def generate(
            self,
            request,
        ):
            self.streamed = True

            yield SimpleNamespace(
                kind="done",
                usage=None,
                finish_reason="stop",
                tool_call=None,
            )

    async def scenario():
        assert not sink.attached

        llm = CompleteLLM()

        engine = StepEngine(
            llm=llm,
        )

        turn = Turn(
            agent_hash="hash-c",
            parent_hash=None,
            depth=0,
            task=None,
            world={},
            persona="p",
            # A bare-string report passes through unframed.
            reports=("observation",),
        )

        result = await engine.step(turn)

        # The step records the assistant's outcome structurally
        # (reply); rendering yields the observation first, then
        # the assistant reply.
        assert [
            message.content
            for message in render_turn(result)
        ] == ["observation", "hi"]

        assert not llm.streamed

        assert len(llm.requests) == 1

    asyncio.run(scenario())


# ============================================================================
# User input echo
# ============================================================================


def test_user_input_echo_carries_text_and_mid_in_content():
    bus = EventBus()

    sink.attach(bus, boot_id="boot123")

    sink.emit(
        "user_input",
        content={"text": "hello", "mid": "m1"},
    )

    event = bus.history()[-1]

    assert event["t"] == "user_input"

    assert event["content"]["text"] == "hello"

    assert event["content"]["mid"] == "m1"

    # boot_id rides the envelope now, never the content.
    assert event["boot_id"] == "boot123"

    assert "boot_id" not in event["content"]

    # The id is sink-assigned like every other event's.
    assert event["id"]

    # An attach without a boot_id simply omits the envelope field.
    sink.reset()

    sink.attach(bus)

    sink.emit(
        "user_input",
        content={"text": "second"},
    )

    event = bus.history()[-1]

    assert event["t"] == "user_input"

    assert event["content"]["text"] == "second"

    assert "boot_id" not in event

    assert "mid" not in event["content"]
