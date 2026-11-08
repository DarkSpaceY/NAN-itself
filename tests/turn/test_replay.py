"""Replay: a stored Turn rebuilds the UI events it produced."""

from __future__ import annotations

from nan.events import sink
from nan.llm import ThinkingBlock, VerbCall
from nan.turn import InboxMessage, LLMCall, Turn, VerbResult, freeze_turn, replay_turns


def _types() -> list[str]:
    return [event["t"] for event in sink.history()]


def test_replay_rebuilds_a_committed_turn() -> None:
    sink.reset(boot_id="test-boot")

    turn = freeze_turn(
        Turn(
            agent_id="r0",
            parent_id=None,
            depth=0,
            origin="first",
            phase="committed",
            started_at=1.0,
            ended_at=5.0,
            inbox=(InboxMessage.from_user("hi"),),
            reply="done",
            thinking=(ThinkingBlock(text="why"),),
            verb_calls=(VerbCall(id="c1", name="shell", arguments={"command": "ls"}),),
            verb_results=(
                VerbResult(
                    call_id="c1",
                    name="shell",
                    output="ok",
                    started_at=3.0,
                    ended_at=4.0,
                ),
            ),
            llm=LLMCall(model="m", started_at=2.0, ended_at=4.5),
        )
    )

    replay_turns([turn])

    assert _types() == [
        "user_input",
        "turn_started",
        "injections_collected",
        "thinking_started",
        "thinking_delta",
        "thinking_done",
        "output_started",
        "output_delta",
        "output_cancelled",
        "verb_started",
        "verb_result",
        "verb_done",
        "turn_done",
    ]

    events = sink.history()

    assert events[1]["ts"] == 1.0
    assert events[7]["content"]["text"] == "done"
    assert events[9]["ts"] == 3.0
    assert events[10]["ts"] == 4.0
    assert events[-1]["ts"] == 5.0
    assert events[-1]["content"]["calls"] == 1
    assert events[-1]["content"]["history_chars"] > 0


def test_replay_leaves_a_resident_turn_to_the_resume_path() -> None:
    sink.reset(boot_id="test-boot")

    turn = Turn(
        agent_id="r0",
        parent_id=None,
        depth=0,
        phase="thought",
        started_at=1.0,
        reply="ok",
        verb_calls=(VerbCall(id="c1", name="sleep", arguments={}),),
        llm=LLMCall(model="m", started_at=2.0, ended_at=3.0),
    )

    replay_turns([turn])

    # No turn_started (the resume path announces it) and no
    # turn_done (the turn is still open); the call stays open.
    assert _types() == [
        "injections_collected",
        "output_started",
        "output_delta",
        "output_cancelled",
        "verb_started",
    ]


def test_replay_publishes_only_user_inputs_still_waiting() -> None:
    sink.reset(boot_id="test-boot")

    replay_turns(
        [],
        pending=(
            InboxMessage.from_user("typed"),
            InboxMessage.from_parent("task", source="r0"),
        ),
    )

    assert [event["content"]["text"] for event in sink.history()] == ["typed"]
