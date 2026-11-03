"""
Control: the agent loop as a fold over its own state.

The loop holds no branching of its own. Each iteration derives the
current Status from the agent, folds it once through `transition`,
and performs the returned Action. Death, pause, shutdown, the
resident turn, the ReACT continuation and the autonomous cadence
are decided here, once: the loop has no `if death` / `if paused`
between awaits, and no second place asks what is pending.

`action` is what the loop must do next:

    stop    leave the fold and finalize (dispose or suspend)
    wait    block until a latch changes or input arrives
    run     open a new turn
    resume  continue the open turn from its phase
    pace    wait out `autonomous_interval`, then open a turn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from nan.turn import Death

Action = Literal["stop", "wait", "run", "resume", "pace"]


@dataclass(frozen=True)
class Status:
    """
    Everything a transition may read: the three latches, whether a
    turn is open (opened, not ended), whether the last closed turn
    called verbs (the ReACT continuation is owed), whether input
    waits to be observed, and whether the cadence is served.
    """

    death: Death | None = None
    suspend: bool = False  # shutdown: persist, no death
    paused: bool = False
    open_turn: bool = False
    owed_step: bool = False
    input_ready: bool = False
    paced: bool = True  # a turn may open now


def transition(status: Status, *, interval: float) -> Action:
    """The one place (status, interval) becomes an action."""
    if status.death is not None or status.suspend:
        return "stop"

    if status.paused:
        return "wait"

    # The open turn owns the agent: it continues where it stands.
    if status.open_turn:
        return "resume"

    # An interval agent is paced entirely by the cadence: sleep, then
    # run, then sleep again. Pending input waits for the interval.
    if interval > 0:
        return "run" if status.paced else "pace"

    # Results the model has not seen yet continue the ReACT cycle
    # without new input.
    if status.input_ready or status.owed_step:
        return "run"

    return "wait"
