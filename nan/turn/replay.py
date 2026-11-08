"""
Replaying: the ONLY place a stored Turn becomes UI events.

A restarted core rebuilds each agent's timeline from its snapshot,
so the event ring needs no durable copy. The shapes mirror the
live emissions in nan/agent/engine.py and nan/agent/core.py: a
committed turn is complete, so replaying it is deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from nan.events import sink
from nan.turn.model import InboxMessage, Turn


def replay_turns(turns: Sequence[Turn], *, pending: Iterable[InboxMessage] = ()) -> None:
    """
    Publish `turns` oldest first, then the inputs no turn has
    observed yet. Replayed events carry the turn's timestamps and
    the current boot id, so the frontend rebuilds them like live
    history.
    """
    chars = 0
    epoch: int | None = None

    for index, turn in enumerate(turns, start=1):
        if turn.epoch != epoch:
            chars = 0
            epoch = turn.epoch

        chars = _replay_turn(turn, index=index, history_chars=chars)

    for message in pending:
        if message.kind == "user":
            sink.emit("user_input", ts=message.received_at, content={"text": message.text})


def _replay_turn(turn: Turn, *, index: int, history_chars: int) -> int:
    """Emit one turn and return the history size it leaves behind."""
    committed = turn.phase == "committed"

    identity = {"agent_id": turn.agent_id, "depth": turn.depth}
    llm = turn.llm
    call_started = llm.ended_at if llm is not None else turn.started_at
    started = llm.started_at if llm is not None else turn.started_at

    for message in turn.inbox:
        if message.kind == "user":
            sink.emit("user_input", ts=message.received_at, content={"text": message.text})

    # The live turn_started is emitted when the turn opens, so only
    # a committed turn re-announces it. A resident turn is announced
    # by the resume path (core.py) and must not be replayed twice.
    if committed:
        sink.emit(
            "turn_started",
            ts=turn.started_at,
            content={
                "agent_id": turn.agent_id,
                "turn_index": index,
                "inbox_count": len(turn.inbox),
            },
        )

    # The inject phase only runs after the inputs were observed.
    if turn.phase != "observed":
        sink.emit(
            "injections_collected",
            ts=turn.started_at,
            content={
                "agent_id": turn.agent_id,
                "services": [
                    {
                        "service_id": injection.service_id,
                        "ok": injection.ok,
                        "duration_s": injection.ended_at - injection.started_at,
                        "text_len": len(injection.text),
                    }
                    for injection in turn.injections
                ],
            },
        )

    reasoning = "".join(block.text for block in turn.thinking)

    if reasoning:
        thinking_id = sink.emit("thinking_started", ts=started, content=identity)

        sink.emit(
            "thinking_delta", id=thinking_id, ts=started, content={**identity, "text": reasoning}
        )

        sink.emit("thinking_done", id=thinking_id, ts=started, content=identity)

    if turn.reply:
        stream_id = sink.emit("output_started", ts=started, content=identity)

        sink.emit(
            "output_delta", id=stream_id, ts=call_started, content={**identity, "text": turn.reply}
        )

        sink.emit(
            "output_cancelled" if turn.verb_calls else "output_done",
            id=stream_id,
            ts=call_started,
            content=identity,
        )

    results = {result.call_id: result for result in turn.verb_results}

    for call in turn.verb_calls:
        result = results.get(call.id)

        sink.emit(
            "verb_started",
            ts=result.started_at if result is not None else call_started,
            content={"agent_id": turn.agent_id, "verb": call.name, "arguments": call.arguments},
        )

        # A malformed call never produced a result the live stream
        # reported: the adapter's reason was its failure.
        if call.error is not None:
            sink.emit(
                "verb_failed",
                ts=call_started,
                content={
                    "agent_id": turn.agent_id,
                    "verb": call.name,
                    "error": {"type": "malformed_tool_call", "message": call.error},
                },
            )

            continue

        # A resident turn may still owe results; the interrupted
        # calls stay open and the live turn_interrupted closes them.
        if result is None:
            continue

        sink.emit(
            "verb_result",
            ts=result.ended_at,
            content={"agent_id": turn.agent_id, "verb": call.name, "text": result.output},
        )

        sink.emit(
            "verb_done",
            ts=result.ended_at,
            content={
                "agent_id": turn.agent_id,
                "verb": call.name,
                "duration_s": result.ended_at - result.started_at,
            },
        )

    if not committed:
        return history_chars

    if turn.rendered is not None:
        history_chars += sum(
            len(message.content or "")
            + sum(len(block.text) + len(block.data or "") for block in message.thinking)
            for message in turn.rendered
        )

    sink.emit(
        "turn_done",
        ts=turn.ended_at if turn.ended_at is not None else call_started,
        content={
            "agent_id": turn.agent_id,
            "turn_index": index,
            "usage": (
                llm.usage.model_dump() if llm is not None and llm.usage is not None else None
            ),
            "latency_s": (llm.ended_at - llm.started_at if llm is not None else 0.0),
            "retries": (llm.retries if llm is not None else 0),
            "calls": len(turn.verb_calls),
            "history_chars": history_chars,
        },
    )

    return history_chars
