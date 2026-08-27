"""
System module contracts: deterministic facts only.

psutil, platform shims and the clock are all stubbed; every
event path is driven by direct state manipulation. Query output
must contain zero heuristic conclusions.
"""

import time

import pytest

from src.nan_itself.modules.builtin.system import (
    SystemModule,
)
from src.nan_itself.utils.system import (
    SystemProbe,
)


class Clock:
    """Independent wall/mono clocks: sleep jumps become testable."""

    def __init__(self, wall=1_000_000.0, mono=100.0):
        self.wall = wall

        self.mono_v = mono

    def now(self):
        return self.wall

    def mono(self):
        return self.mono_v

    def advance(self, wall=0.0, mono=0.0):
        self.wall += wall

        self.mono_v += mono


BASE_METRICS = {
    "cpu": 30.0,
    "mem": 60.0,
    "swap": 5.0,
    "disk": 40.0,
    "battery": 80,
    "charging": True,
    "net_up": True,
    "uptime": 3600.0,
    "self_cpu": 1.0,
    "self_mem_mb": 100,
}


class Holder:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        return self.value


def make_probe(
    clock=None,
    idle=5.0,
    focus=None,
    metrics=None,
    procs=None,
    ssid="Home",
    displays=1,
    metrics_fn=None,
    procs_fn=None,
):
    clock = clock or Clock()

    probe = SystemProbe(
        now=clock.now,
        mono=clock.mono,
        idle_fn=idle if callable(idle) else Holder(idle),
        focus_fn=focus if callable(focus) else Holder(focus),
        ssid_fn=ssid if callable(ssid) else Holder(ssid),
        displays_fn=displays if callable(displays) else Holder(displays),
        metrics_fn=metrics_fn or (lambda: dict(metrics or BASE_METRICS)),
        procs_fn=procs_fn or (lambda cache: list(procs or [])),
    )

    return probe, clock


def facts_without_events(facts):
    return {k: v for k, v in facts.items() if k != "events"}


# ======================================================================
# Probe: snapshot + deterministic events
# ======================================================================


def test_sample_carries_facts():
    probe, _ = make_probe(idle=12.0, focus=("Code", "audio.py"))

    facts = probe.sample()

    assert facts["idle"] == 12.0

    assert facts["focus"] == ("Code", "audio.py")

    assert facts["cpu"] == 30.0

    assert facts["battery"] == 80


def test_wake_event_on_wall_clock_jump():
    probe, clock = make_probe()

    probe.sample()

    # System slept: wall jumped, monotonic barely moved.
    clock.advance(wall=3600.0, mono=1.0)

    probe.sample()

    assert any("woke" in text for _, text in probe.events)

    # Normal cadence: no duplicate.
    clock.advance(wall=5.0, mono=5.0)

    probe.sample()

    assert sum("woke" in text for _, text in probe.events) == 1


def test_ssid_change_event():
    holder = Holder("Home")

    probe, clock = make_probe(ssid=holder)

    probe.sample()

    holder.value = "Office-5G"

    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert any('wifi -> "Office-5G"' in t for _, t in probe.events)


def test_display_change_event():
    holder = Holder(1)

    probe, clock = make_probe(displays=holder)

    probe.sample()

    holder.value = 2

    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert any("display +1" in t for _, t in probe.events)


def test_disk_threshold_event_with_cooldown():
    metrics = dict(BASE_METRICS, disk=50.0)

    holder = Holder(metrics)

    probe, clock = make_probe(metrics_fn=holder)

    probe.sample()

    metrics["disk"] = 91.0

    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert sum("disk 91%" in t for _, t in probe.events) == 1

    # Within cooldown: no repeat.
    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert sum("disk 91%" in t for _, t in probe.events) == 1


def test_runaway_process_needs_three_consecutive_samples():
    proc = [(123, "Xcode", 95.0)]

    holder = Holder(proc)

    probe, clock = make_probe(procs_fn=holder)

    for _ in range(2):
        probe.sample()

        clock.advance(wall=5.0, mono=5.0)

    assert list(probe.events) == []

    probe.sample()

    assert any("runaway: Xcode" in t for _, t in probe.events)

    # Stays loud but never re-fires.
    probe.sample()

    assert sum("runaway" in t for _, t in probe.events) == 1

    proc[0] = (123, "Xcode", 10.0)

    probe.sample()

    # Recovered: tracker drops it; a new burst re-fires.
    proc[0] = (123, "Xcode", 95.0)

    for _ in range(3):
        probe.sample()

        clock.advance(wall=5.0, mono=5.0)

    assert sum("runaway" in t for _, t in probe.events) == 2


def test_work_accumulates_and_resets_on_away():
    probe, clock = make_probe(idle=5.0)

    probe.sample()

    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert probe.work_seconds == pytest.approx(10.0)

    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert probe.work_seconds == pytest.approx(20.0)

    # Away: idle >= 300 resets the counter.
    probe.idle_fn = Holder(400.0)

    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert probe.work_seconds == 0.0


# ======================================================================
# Module: rendering (territory + facts only)
# ======================================================================


def make_module(facts):
    module = SystemModule()

    module._latest = facts

    return module


def base_facts(**overrides):
    facts = {
        "now": time.time(),
        "cpu": 30.0,
        "mem": 60.0,
        "swap": 5.0,
        "disk": 40.0,
        "battery": 80,
        "charging": True,
        "net_up": True,
        "uptime": 22500.0,
        "self_cpu": 2.1,
        "self_mem_mb": 310,
        "idle": 12.0,
        "work_seconds": 1500.0,
        "focus": ("Code", "audio.py"),
        "events": [(time.time(), "woke 14:02")],
    }

    facts.update(overrides)

    return facts


def run_query(module):
    import asyncio

    return asyncio.run(module.query(turn=None))


def test_query_renders_single_territory_with_facts():
    module = make_module(base_facts())

    rendered = run_query(module)

    assert rendered.startswith("[System]\n")

    assert rendered.count("[System]") == 1

    assert "- now:" in rendered

    assert "- focus: Code — audio.py" in rendered

    assert "- idle: 12s" in rendered

    assert "work: 25m" in rendered

    assert "battery 80% charging" in rendered

    assert "- self: nan cpu 2.1% mem 310MB" in rendered

    assert "- events: woke 14:02" in rendered


def test_query_lines_are_all_facts():
    module = make_module(base_facts())

    rendered = run_query(module)

    for line in rendered.splitlines():
        assert line == "[System]" or line.startswith("- "), line

    # Heuristic conclusions are banned from this module.
    for banned in ("会议", "离开", "勿扰", "工作中", "开会"):
        assert banned not in rendered


def test_query_omits_optional_rows():
    facts = base_facts(
        battery=None,
        charging=None,
        focus=None,
        events=[],
        idle=None,
    )

    rendered = run_query(make_module(facts))

    assert "battery" not in rendered

    assert "- focus:" not in rendered

    assert "- events:" not in rendered

    assert "idle: unknown" in rendered


def test_query_none_before_first_sample():
    module = SystemModule()

    assert run_query(module) is None


def test_probe_degrades_when_psutil_missing(monkeypatch):
    import src.nan_itself.utils.system as sys_utils

    monkeypatch.setattr(sys_utils, "HAS_PSUTIL", False)

    probe, _ = make_probe(metrics_fn=sys_utils._psutil_metrics)

    facts = probe.sample()

    assert facts["cpu"] is None

    assert facts["uptime"] is None


def test_focus_change_event():
    holder = Holder(("Code", "a.py"))

    probe, clock = make_probe(focus=holder)

    probe.sample()

    holder.value = ("Chrome", "github.com")

    clock.advance(wall=10.0, mono=10.0)

    probe.sample()

    assert any("focus -> Chrome" in t for _, t in probe.events)

    probe.sample()

    assert sum("focus ->" in t for _, t in probe.events) == 1
