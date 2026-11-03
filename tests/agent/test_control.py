"""Control: the one (status, interval) -> action table."""

from __future__ import annotations

import pytest

from nan.agent.control import Status, transition
from nan.turn import Death


def test_death_stops_first() -> None:
    status = Status(death=Death("killed"), paused=True, open_turn=True, input_ready=True)

    assert transition(status, interval=0.0) == "stop"


def test_shutdown_stops() -> None:
    status = Status(suspend=True, paused=True, open_turn=True, input_ready=True)

    assert transition(status, interval=0.0) == "stop"


def test_pause_waits_before_everything_but_exit() -> None:
    status = Status(paused=True, open_turn=True, input_ready=True, owed_step=True, paced=False)

    assert transition(status, interval=0.0) == "wait"
    assert transition(status, interval=1.0) == "wait"


def test_open_turn_resumes_from_its_phase() -> None:
    status = Status(open_turn=True, input_ready=True)

    assert transition(status, interval=0.0) == "resume"
    assert transition(status, interval=1.0) == "resume"


def test_interval_agent_runs_then_paces() -> None:
    assert transition(Status(paced=True), interval=1.0) == "run"
    assert transition(Status(paced=False), interval=1.0) == "pace"


def test_interval_ignores_pending_input() -> None:
    """An interval agent never skips its cadence because input waits."""
    status = Status(paced=False, input_ready=True, owed_step=True)

    assert transition(status, interval=1.0) == "pace"


def test_event_driven_runs_on_input_or_owed_step() -> None:
    assert transition(Status(input_ready=True), interval=0.0) == "run"
    assert transition(Status(owed_step=True), interval=0.0) == "run"


def test_idle_event_driven_agent_waits() -> None:
    assert transition(Status(), interval=0.0) == "wait"


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_non_positive_interval_is_event_driven(interval: float) -> None:
    assert transition(Status(paced=False), interval=interval) == "wait"
