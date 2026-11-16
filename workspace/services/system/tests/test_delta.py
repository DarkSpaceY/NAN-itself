"""
Delta/state tests: one full snapshot per epoch, changed fields after.

The probe's samplers are not started here; facts are placed on the
service directly so each turn is deterministic.
"""

from __future__ import annotations

import main as system_mod
import pytest
from nan_service_kit import HubLink
from nan_service_kit.testing import StubHub


@pytest.fixture(autouse=True)
def fixed_minute(monkeypatch):
    """Keep the `now` diff field stable across calls."""
    monkeypatch.setattr(system_mod, "_minute_key", lambda: "12:00")


def make_service() -> system_mod.SystemService:
    config = system_mod.load_config()

    stub = StubHub()

    return system_mod.SystemService(
        config,
        link=HubLink(config, transport=stub.transport()),
    )


def facts(**overrides) -> dict:
    base = {
        "now": 1000.0,
        "identity": {"hostname": "h", "model": "m", "os": "macOS", "arch": "arm64"},
        "cpu": 10.0,
        "mem": 40.0,
        "swap": None,
        "disk": 50.0,
        "disk_read_bps": None,
        "disk_write_bps": None,
        "battery": None,
        "charging": None,
        "uptime": 100.0,
        "self_pid": 1,
        "self_cpu": 1.0,
        "self_mem_mb": 10,
        "top_procs": [],
        "idle": 0.0,
        "work_seconds": 0.0,
        "focus": None,
        "locked": None,
        "dnd": None,
        "assertions": None,
        "events": [],
        "notes": [],
    }

    base.update(overrides)

    return base


def turn(
    *,
    agent: str = "a",
    started: float = 1.0,
    origin: str = "first",
    epoch: int = 0,
    depth: int = 0,
) -> dict:
    return {
        "agent_id": agent,
        "started_at": started,
        "origin": origin,
        "epoch": epoch,
        "depth": depth,
    }


def test_first_turn_is_full_then_unchanged_is_silent():
    service = make_service()

    service._facts = facts()

    first = service.contribution(turn())

    assert "- host: m macOS arm64" in first

    assert "- note: snapshot for this epoch" in first

    assert "cpu 10%" in first

    assert service.contribution(turn(started=2.0, origin="normal")) == ""


def test_only_the_changed_band_is_sent():
    service = make_service()

    service._facts = facts()

    service.contribution(turn())

    service._facts = facts(cpu=20.0)

    assert service.contribution(turn(started=2.0, origin="normal")) == "- cpu: 20%"


def test_mem_pressure_replaces_the_band_line():
    service = make_service()

    service._facts = facts(mem=84.0)

    service.contribution(turn())

    service._facts = facts(mem=85.0)

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- pressure: mem 85% (high)"
    )

    service._facts = facts(mem=95.0)

    assert service.contribution(turn(started=3.0, origin="normal")) == (
        "- pressure: mem 95% (critical)"
    )

    service._facts = facts(mem=70.0)

    assert service.contribution(turn(started=4.0, origin="normal")) == (
        "- pressure: mem normal (70%)"
    )


def test_disk_pressure_uses_the_same_suppression():
    service = make_service()

    service._facts = facts(disk=89.0)

    service.contribution(turn())

    service._facts = facts(disk=90.0)

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- pressure: disk 90% (high)"
    )


def test_battery_low_and_charging_labels():
    service = make_service()

    service._facts = facts(battery=25, charging=False)

    service.contribution(turn())

    service._facts = facts(battery=20, charging=False)

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- battery: 20% low (on battery)"
    )

    service._facts = facts(battery=19, charging=False)

    assert service.contribution(turn(started=3.0, origin="normal")) == (
        "- battery: 19% low (on battery)"
    )

    service._facts = facts(battery=19, charging=True)

    assert service.contribution(turn(started=4.0, origin="normal")) == (
        "- battery: 19% charging"
    )


def test_duplicate_turn_replays_the_same_text():
    service = make_service()

    service._facts = facts()

    first = service.contribution(turn())

    assert service.contribution(turn()) == first

    assert service.contribution(turn(started=2.0, origin="normal")) == ""


def test_agents_have_independent_baselines():
    service = make_service()

    service._facts = facts()

    assert "- note:" in service.contribution(turn(agent="a"))

    assert "- note:" in service.contribution(turn(agent="b"))

    assert service.contribution(turn(agent="a", started=2.0, origin="normal")) == ""


def test_context_cleared_sends_a_new_snapshot():
    service = make_service()

    service._facts = facts()

    service.contribution(turn())

    assert "- note:" in service.contribution(
        turn(started=2.0, origin="context_cleared"),
    )


def test_child_first_turn_skips_notes_from_before_it_existed():
    service = make_service()

    service._facts = facts(notes=[(1.0, "old"), (5.0, "new")])

    child = service.contribution(
        turn(agent="child", started=3.0, origin="first", depth=1),
    )

    assert "old" not in child

    assert "new" in child

    root = service.contribution(
        turn(agent="root", started=3.0, origin="first", depth=0),
    )

    assert "old" in root

    assert "new" in root
