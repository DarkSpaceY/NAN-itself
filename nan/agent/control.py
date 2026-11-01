"""
Control: the agent loop as a fold over one event stream.

The loop holds no branching of its own. Each iteration derives the
current Status from the agent, folds ONE event through
`transition`, and performs the returned Action:

    event = await ports.next()        inbox message or latch change
    action = transition(status, event, interval=...)
    event = await perform(action)     run a turn / wait / sleep

Death, pause, shutdown, the open turn and the ReACT continuation
are decided here, once: the loop has no `if death` / `if paused`
scattered between awaits.

`event` is why the fold ran:

    start      the loop's first step
    input      the inbox has a message to observe
    change     a control latch changed (pause / resume / exit)
    turn_done  the agent finished one turn task

`action` is what the loop must do next:

    stop       leave the fold and finalize (dispose or suspend)
    wait       block for the next event
    run        open a new turn
    resume     continue the open turn from its phase
    sleep      the autonomous-interval pause, then fold again
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nan.turn import Death

Action = Literal["stop", "wait", "run", "resume", "sleep"]

Event = Literal["start", "input", "change", "turn_done"]


@dataclass(frozen=True)
class Status:
    """
    Everything a transition may read: the three latches, whether a
    turn is open (opened, not ended), and whether the last closed
    turn called verbs (the ReACT continuation is owed).
    """

    death: Death | None = None
    stopping: bool = False  # shutdown: persist, no death
    paused: bool = False
    open_turn: bool = False
    called: bool = False


def transition(status: Status, event: Event, *, interval: float) -> Action:
    """The one place (status, event) becomes an action."""
    if status.death is not None or status.stopping:
        return "stop"

    if status.paused:
        return "wait"

    if status.open_turn:
        return "resume"

    if event == "turn_done":
        if interval > 0:
            return "sleep"

        # Results the model has not seen yet continue the ReACT
        # cycle without new input.
        return "run" if status.called else "wait"

    if event == "start":
        # Event-driven agents wait for their first observation;
        # interval agents take a turn immediately.
        return "run" if interval > 0 else "wait"

    return "run"
