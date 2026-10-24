"""
Probe diff-logic tests: SystemProbe.

main.py is a plain script (not an importable package), so it is
loaded by path. All probe inputs are injectable callables; no
test touches real hardware.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parent.parent / "main.py"

    spec = importlib.util.spec_from_file_location("nan_system_service_main", path)

    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


system_mod = _load()

# ============================================================================
# Shared fakes
# ============================================================================


class FakeClock:
    """Paired wall/mono clock with a steerable offset."""

    def __init__(self) -> None:
        self.wall = 1000.0

        self.mono = 0.0

    def now(self) -> float:
        return self.wall

    def tick(self, seconds: float, jump: float = 0.0) -> None:
        self.mono += seconds

        self.wall += seconds + jump


class FakeDiskIO:
    def __init__(self, read_bytes: int, write_bytes: int) -> None:
        self.read_bytes = read_bytes

        self.write_bytes = write_bytes


def event_texts(probe) -> list[str]:
    return [text for _, text in probe.events]



# ============================================================================
# SystemProbe
# ============================================================================


def make_system_probe(clock: FakeClock, **overrides):
    state: dict = {
        "idle": 0.0,
        "focus": None,
        "displays": 1,
        "metrics": {"cpu": 10.0, "mem": 40.0, "disk": 50.0, "uptime": 100.0},
        "procs": [],
        "locked": None,
        "dnd": None,
        "assertions": None,
        "disk_io": None,
    }

    probe = system_mod.SystemProbe(
        now=clock.now,
        mono=lambda: clock.mono,
        idle_fn=lambda: state["idle"],
        focus_fn=lambda: state["focus"],
        displays_fn=lambda: state["displays"],
        metrics_fn=lambda: (
            state["metrics"]
            | {
                "swap": None,
                "battery": None,
                "charging": None,
                "self_cpu": 1.0,
                "self_mem_mb": 10,
            }
        ),
        procs_fn=lambda cache: state["procs"],
        lock_fn=lambda: state["locked"],
        dnd_fn=lambda: state["dnd"],
        assertions_fn=lambda: state["assertions"],
        disk_io_fn=lambda: state["disk_io"],
        **overrides,
    )

    return probe, state


def test_system_wake_on_clock_jump():
    clock = FakeClock()

    probe, _ = make_system_probe(clock)

    probe.sample()

    clock.tick(5.0, jump=3600.0)  # wall jumped vs mono: woke from sleep

    probe.sample()

    assert any("woke from sleep" in text for text in event_texts(probe))


def test_system_focus_change_event():
    clock = FakeClock()

    probe, state = make_system_probe(clock)

    state["focus"] = ("Editor", "doc")

    probe.sample()

    state["focus"] = ("Browser", "")

    probe.sample()

    assert any("focus -> Browser" in text for text in event_texts(probe))


def test_system_display_change_event():
    clock = FakeClock()

    probe, state = make_system_probe(clock)

    probe.sample()

    state["displays"] = 2

    probe.sample()

    assert any("display +1" in text for text in event_texts(probe))


def test_system_disk_event_throttled():
    clock = FakeClock()

    probe, state = make_system_probe(
        clock,
        disk_threshold=90.0,
        disk_announce_every=600.0,
    )

    state["metrics"]["disk"] = 95.0

    probe.sample()

    assert any("disk 95%" in text for text in event_texts(probe))

    n = len(probe.events)

    clock.tick(5.0)

    probe.sample()  # still full disk but throttled

    assert len(probe.events) == n


def test_system_runaway_needs_consecutive_samples():
    clock = FakeClock()

    probe, state = make_system_probe(clock, runaway_cpu=90.0, runaway_samples=3)

    state["procs"] = [(42, "hog", 95.0)]

    probe.sample()

    probe.sample()

    assert not any("runaway" in text for text in event_texts(probe))

    probe.sample()  # third consecutive

    assert any("runaway: hog 95%" in text for text in event_texts(probe))


def test_system_work_resets_after_idle():
    clock = FakeClock()

    probe, state = make_system_probe(clock, idle_reset_after=300.0)

    state["idle"] = 1.0

    probe.sample()

    probe.work_seconds = 120.0

    state["idle"] = 400.0

    probe.sample()

    assert probe.work_seconds == 0.0


def test_system_top_procs_with_pid():
    clock = FakeClock()

    probe, state = make_system_probe(clock, top_cpu_threshold=20.0, top_procs_max=3)

    state["procs"] = [
        (101, "chrome", 45.0),
        (102, "Code", 18.0),  # below threshold: dropped
        (103, "Backup", 30.0),
        (104, "idle", 1.0),
    ]

    facts = probe.sample()

    assert facts["top_procs"] == [(101, "chrome", 45.0), (103, "Backup", 30.0)]

    text = system_mod.render_facts(facts, title_cap=80, events_limit=3)

    assert "top: chrome 45% (pid 101), Backup 30% (pid 103)" in text


def test_system_render_contains_facts():
    clock = FakeClock()

    probe, state = make_system_probe(clock)

    state["focus"] = ("Editor", "doc")

    facts = probe.sample()

    text = system_mod.render_facts(facts, title_cap=80, events_limit=3)

    assert "focus: Editor" in text

    assert "cpu 10%" in text


def test_system_lock_events():
    clock = FakeClock()

    probe, state = make_system_probe(clock)

    state["locked"] = False

    probe.sample()  # first observation, no event

    assert not any("screen" in text for text in event_texts(probe))

    state["locked"] = True

    probe.sample()

    assert any("screen locked" in text for text in event_texts(probe))

    state["locked"] = False

    probe.sample()

    assert any("screen unlocked" in text for text in event_texts(probe))


def test_system_dnd_change_event():
    clock = FakeClock()

    probe, state = make_system_probe(clock)

    state["dnd"] = "off"

    probe.sample()

    state["dnd"] = "sleep"

    probe.sample()

    assert any("focus mode sleep" in text for text in event_texts(probe))


def test_system_assertions_rendered():
    clock = FakeClock()

    probe, state = make_system_probe(clock)

    state["assertions"] = ["coreaudiod: PreventUserIdleSystemSleep"]

    facts = probe.sample()

    text = system_mod.render_facts(facts, title_cap=80, events_limit=3)

    assert "sleep held by: coreaudiod" in text


def test_system_disk_io_rate():
    clock = FakeClock()

    probe, state = make_system_probe(clock)

    state["disk_io"] = FakeDiskIO(0, 0)

    probe.sample()  # seeds counters, no rate yet

    state["disk_io"] = FakeDiskIO(5_000_000, 1_000_000)

    clock.tick(5.0)

    facts = probe.sample()

    assert facts["disk_read_bps"] == pytest.approx(1_000_000.0)

    assert facts["disk_write_bps"] == pytest.approx(200_000.0)

    text = system_mod.render_facts(facts, title_cap=80, events_limit=3)

    assert "disk io: r 977KB/s" in text  # 1e6 B/s / 1024


def test_system_parse_assertions():
    sample = (
        "2010-01-01 00:00:00 +0800\n"
        "Assertions:\n"
        "  pid 123(coreaudiod): [0x0000abc] 00:00:00 PreventUserIdleSystemSleep"
        ' named: "com.apple.audio.AppleHDAControllerOutput" \n'
        "  pid 200(backupd): [0x0000def] 00:00:05 PreventSystemSleep \n"
        "  pid 200(backupd): [0x0000def] 00:00:05 PreventSystemSleep \n"
        '  pid 300(app): [0x0000fff] 00:10:00 UserIsActive named: "com.apple.iohideventsystem" \n'
    )

    parsed = system_mod.parse_assertions(sample)

    assert parsed == [
        "coreaudiod: PreventUserIdleSystemSleep com.apple.audio.AppleHDAControllerOutput",
        "backupd: PreventSystemSleep",
        "app: UserIsActive com.apple.iohideventsystem",
    ]



# ============================================================================
# Turn endpoint contract (no hub needed)
# ============================================================================


def test_turn_contract_roundtrip():
    from fastapi.testclient import TestClient

    config = system_mod.Config({"hub_url": "http://127.0.0.1:1", "port": 0})

    service = system_mod.SystemService(config)

    service._rendered = "hello"

    app = system_mod.build_app(service)

    with TestClient(app) as client:  # lifespan starts/ends inside
        response = client.post("/turn", json={"turn": {}})

        assert response.status_code == 200

        assert response.json() == {"text": "hello"}

