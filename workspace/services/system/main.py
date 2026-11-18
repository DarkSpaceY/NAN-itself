"""
System service: the agent feeling its own body.

An injection service (see docs/for-service-authors): registers
with the hub, samples the host on a cadence, and answers every
turn request with a full snapshot for a new epoch or only the
changed fields for the rest of it.

Disciplines:

    facts only  every line is a directly verifiable measurement
                or a deterministic event; the module measures,
                the agent judges
    degrade     every platform shim returns None on failure; the
                service never dies from a missing binary
    serve fast  sampling runs in the background; /turn only
                quantizes and renders the ready snapshot

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import os
import platform
import re
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import uvicorn
from fastapi import FastAPI
from loguru import logger
from nan_service_kit import HubLink, ServiceConfig, read_config, setup_logging

try:
    import mss

    HAS_MSS = True

except Exception:
    mss = None  # type: ignore[assignment]

    HAS_MSS = False


HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"


# ============================================================================
# Platform shims (each returns None when unknown)
# ============================================================================


def _run(cmd: list[str] | str, timeout: float = 4.0) -> str:
    """One platform probe; any failure -> empty string."""
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result.stdout or ""

    except Exception:
        return ""


def idle_seconds() -> float | None:
    """Seconds since last user input."""
    system = platform.system()

    if system == "Darwin":
        out = _run(
            "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF/1000000000; exit}'",
        )

        try:
            return float(out.strip())

        except ValueError:
            return None

    if system == "Linux":
        out = _run("xprintidle")

        try:
            return float(out.strip()) / 1000.0

        except ValueError:
            return None

    if system == "Windows":

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("dwTime", ctypes.c_uint),
            ]

        info = LASTINPUTINFO()

        info.cbSize = ctypes.sizeof(LASTINPUTINFO)

        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            tick = ctypes.windll.kernel32.GetTickCount()

            return max(0.0, (tick - info.dwTime) / 1000.0)

        return None

    return None


def foreground_app() -> tuple[str, str] | None:
    """(app_name, window_title) of the frontmost app."""
    system = platform.system()

    if system == "Darwin":
        # lsappinfo needs no Automation permission, unlike
        # System Events UI scripting (-10004).
        asn = _run("lsappinfo front").strip()

        if not asn:
            return None

        out = _run(f'lsappinfo info -only name "{asn}"')

        match = re.search(r'"([^"]+)"\s*=\s*"([^"]+)"', out)

        if match:
            return match.group(2), ""

        out = _run(f'lsappinfo info "{asn}"')

        first = out.splitlines()[0] if out else ""

        quoted = re.search(r'"([^"]+)"', first)

        return (quoted.group(1), "") if quoted else None

    if system == "Linux":
        out = _run("xdotool getactivewindow getwindowname").strip()

        if out:
            return "unknown", out

        return None

    if system == "Windows":
        hwnd = ctypes.windll.user32.GetForegroundWindow()

        if not hwnd:
            return None

        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)

        buf = ctypes.create_unicode_buffer(length + 1)

        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)

        return "unknown", buf.value

    return None


def display_count() -> int | None:
    if not HAS_MSS:
        return None

    try:
        monitors = mss.MSS().monitors

        return max(0, len(monitors) - 1)  # [0] is the union

    except Exception:
        return None


def screen_locked() -> bool | None:
    """True when the user session's screen is locked."""
    system = platform.system()

    if system != "Darwin":
        return None

    out = _run("ioreg -n Root -d1 -a")

    if not out:
        return None

    return "CGSSessionScreenIsLocked" in out


DND_ASSERTIONS_PATH = Path.home() / "Library" / "DoNotDisturb" / "DB" / "Assertions.json"


def focus_mode() -> str | None:
    """
    The active Focus/Do-Not-Disturb mode name ("off" when none).
    None when unknowable (missing file, unparsable shape).
    """
    try:
        data = json.loads(DND_ASSERTIONS_PATH.read_text(encoding="utf-8"))

        records = data["data"][0]["storeAssertionRecords"]

    except Exception:
        return None

    for record in records:
        try:
            mode = record["assertionDetails"]["assertionDetailsModeIdentifier"]

            return mode.removeprefix("com.apple.donotdisturb.mode.")

        except Exception:
            continue

    return "off"


ASSERTION_TYPES = (
    "PreventUserIdleSystemSleep",
    "PreventSystemSleep",
    "PreventUserIdleDisplaySleep",
    "UserIsActive",
)


def parse_assertions(text: str) -> list[str] | None:
    """
    Who is holding the system awake: deduped "proc: Type name"
    strings, capped; None when pmset produced nothing (pmset is
    always present on macOS, so empty output means failure).
    """
    if not text:
        return None

    found: list[str] = []

    for line in text.splitlines():
        match = re.search(
            rf"pid \d+\(([^)]*)\):.*?({'|'.join(ASSERTION_TYPES)})(?: named: \"([^\"]+)\")?",
            line,
        )

        if not match:
            continue

        proc, kind, name = match.group(1), match.group(2), match.group(3)

        entry = f"{proc}: {kind}" + (f" {name}" if name else "")

        entry = entry[:80]  # assertion names can be very long

        if entry not in found:
            found.append(entry)

    return found


def power_assertions() -> list[str] | None:
    """Active sleep-preventing assertions, via `pmset -g`."""
    return parse_assertions(_run("pmset -g assertions"))


def static_identity() -> dict[str, Any]:
    """Hardware/software identity; read once at startup."""
    model = _run("sysctl -n hw.model").strip() or None

    version = _run("sw_vers -productVersion").strip() or None

    return {
        "hostname": platform.node() or None,
        "model": model,
        "os": f"macOS {version}" if version else platform.system(),
        "arch": platform.machine() or None,
    }


# ============================================================================
# Probe: one deterministic sample per call, events by diffing
# ============================================================================


def _resolve_core_pid() -> int | None:
    """
    The agent core's pid, exported by the core as NAN_CORE_PID
    when it launches autostart; None when unset or dead.
    """
    raw = os.environ.get("NAN_CORE_PID", "")

    if raw.isdigit() and psutil.pid_exists(int(raw)):
        return int(raw)

    return None


CORE_PID: int | None = _resolve_core_pid()


def _metrics() -> dict[str, Any]:
    """One psutil sweep; every key present, None when unknowable."""
    out: dict[str, Any] = {
        "cpu": None,
        "mem": None,
        "swap": None,
        "disk": None,
        "battery": None,
        "charging": None,
        "uptime": None,
        "self_cpu": None,
        "self_mem_mb": None,
    }

    out["cpu"] = psutil.cpu_percent(interval=None)

    mem = psutil.virtual_memory()

    out["mem"] = mem.percent

    out["swap"] = psutil.swap_memory().percent

    disk = psutil.disk_usage("/")

    out["disk"] = disk.percent

    battery = psutil.sensors_battery()

    if battery is not None:
        out["battery"] = round(battery.percent)

        out["charging"] = battery.power_plugged

    out["uptime"] = time.time() - psutil.boot_time()

    # "self" is the agent core, not this service; without
    # NAN_CORE_PID it degrades to this service's own process.
    try:
        me = psutil.Process(CORE_PID) if CORE_PID is not None else psutil.Process()

        out["self_pid"] = me.pid

        out["self_cpu"] = me.cpu_percent(interval=None)

        out["self_mem_mb"] = round(me.memory_info().rss / (1024 * 1024))

    except Exception:
        pass

    return out


def _process_label(short: str | None, argv: list[str] | None) -> str:
    """Prefer the argv program name when comm was truncated.

    macOS comm is 16 bytes, so names such as 'Helper (Renderer)'
    arrive cut off. The first argv entry is the full program path.
    """
    name = short or "?"

    if not argv:
        return name

    program = Path(argv[0]).name

    if len(program) > len(name):
        return program

    return name


def _top_processes(
    cache: dict[int, tuple[Any, str]],
) -> list[tuple[int, str, float]]:
    """[(pid, name, cpu%)] using cached Process objects."""
    out: list[tuple[int, str, float]] = []

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            entry = cache.get(proc.pid)

            if entry is None:
                short = proc.info["name"] or "?"

                try:
                    argv = proc.cmdline()

                except Exception:
                    argv = None

                entry = (proc, _process_label(short, argv))

                cache[proc.pid] = entry

            out.append((proc.pid, entry[1], entry[0].cpu_percent(None)))

        except Exception:
            continue

    # Prune dead entries occasionally enough.
    if len(cache) > 800:
        alive = {pid for pid, _, _ in out}

        for pid in list(cache):
            if pid not in alive:
                cache.pop(pid)

    return out


class SystemProbe:
    """
    One sample per call; deterministic; events detected by
    diffing against the previous sample. Every input is an
    injectable callable, so tests never touch real hardware.
    """

    def __init__(
        self,
        *,
        now: Any = time.time,
        mono: Any = time.monotonic,
        idle_fn: Any = idle_seconds,
        focus_fn: Any = foreground_app,
        displays_fn: Any = display_count,
        metrics_fn: Any = _metrics,
        procs_fn: Any = _top_processes,
        lock_fn: Any = screen_locked,
        dnd_fn: Any = focus_mode,
        assertions_fn: Any = power_assertions,
        disk_io_fn: Any = None,
        identity: dict[str, Any] | None = None,
        disk_threshold: float = 90.0,
        disk_announce_every: float = 600.0,
        runaway_cpu: float = 90.0,
        runaway_samples: int = 3,
        idle_reset_after: float = 300.0,
        assertions_every: float = 30.0,
        top_cpu_threshold: float = 20.0,
        top_procs_max: int = 3,
        events_max: int = 8,
    ) -> None:
        self.now = now

        self.mono = mono

        self.idle_fn = idle_fn

        self.focus_fn = focus_fn

        self.displays_fn = displays_fn

        self.metrics_fn = metrics_fn

        self.procs_fn = procs_fn

        self.lock_fn = lock_fn

        self.dnd_fn = dnd_fn

        self.assertions_fn = assertions_fn

        self.disk_io_fn = disk_io_fn

        self.identity = identity or {}

        self.disk_threshold = disk_threshold

        self.disk_announce_every = disk_announce_every

        self.runaway_cpu = runaway_cpu

        self.runaway_samples = runaway_samples

        self.idle_reset_after = idle_reset_after

        self.assertions_every = assertions_every

        self.top_cpu_threshold = top_cpu_threshold

        self.top_procs_max = top_procs_max

        self.events: deque[tuple[float, str]] = deque(maxlen=events_max)

        # Transient signals only: these ride the injection. `events`
        # keeps the full probe log for tests and debugging; changes
        # that are already fields (focus, lock, dnd, disk) stay out
        # of `notes` so they are not announced twice.
        self.notes: deque[tuple[float, str]] = deque(maxlen=events_max)

        self._clock_offset: float | None = None

        self._focus: tuple[str, str] | None = None

        self._displays: int | None = None

        self._disk_announced_at = 0.0

        self._runaway: dict[int, int] = {}

        self._proc_cache: dict[int, tuple[Any, str]] = {}

        self._last_sample_mono: float | None = None

        self.work_seconds = 0.0

        self._locked: bool | None = None

        self._dnd: str | None = None

        self._assertions: list[str] | None = None

        self._t_assertions = -1e9

        self._top: list[tuple[int, str, float]] = []

        self._prev_disk_io: tuple[int, int] | None = None

        self._disk_rate: tuple[float, float] | None = None

    # ------------------------------------------------------------------

    def _emit(self, text: str, *, note: bool = False) -> None:
        item = (self.now(), text)

        self.events.append(item)

        if note:
            self.notes.append(item)

    def _check_clock_jump(self) -> None:
        offset = self.now() - self.mono()

        if self._clock_offset is not None:
            jump = offset - self._clock_offset

            if jump > 60.0:
                self._emit("woke from sleep", note=True)

        self._clock_offset = offset

    def _check_work(self, idle: float | None, dt: float) -> None:
        if idle is None:
            return

        if idle >= self.idle_reset_after:
            self.work_seconds = 0.0

        else:
            if self._last_sample_mono is not None:
                self.work_seconds += min(dt, 30.0)

    def _check_focus(self, focus: tuple[str, str] | None) -> None:
        if focus and self._focus and focus[0] != self._focus[0]:
            self._emit(f"focus -> {focus[0]}")

        if focus:
            self._focus = focus

    def _check_displays(self) -> None:
        count = self.displays_fn()

        if count is not None and self._displays is not None and count != self._displays:
            sign = "+" if count > self._displays else "-"

            self._emit(f"display {sign}{abs(count - self._displays)}", note=True)

        if count is not None:
            self._displays = count

    def _check_disk(self, disk: float | None) -> None:
        if disk is None or disk < self.disk_threshold:
            return

        at = self.now()

        if at - self._disk_announced_at >= self.disk_announce_every:
            self._disk_announced_at = at

            self._emit(f"disk {disk:.0f}%")

    def _check_runaway(self) -> None:
        candidates: list[tuple[int, str, float]] = []

        for pid, name, cpu in self.procs_fn(self._proc_cache):
            if cpu >= self.runaway_cpu:
                count = self._runaway.get(pid, 0) + 1

                self._runaway[pid] = count

                if count == self.runaway_samples:
                    self._emit(f"runaway: {name} {cpu:.0f}%", note=True)

            else:
                self._runaway.pop(pid, None)

            if cpu >= self.top_cpu_threshold:
                candidates.append((pid, name, cpu))

        candidates.sort(key=lambda entry: entry[2], reverse=True)

        self._top = candidates[: self.top_procs_max]

    def _check_locked(self) -> None:
        locked = self.lock_fn()

        if locked is not None and self._locked is not None and locked != self._locked:
            self._emit("screen locked" if locked else "screen unlocked")

        if locked is not None:
            self._locked = locked

    def _check_dnd(self) -> None:
        mode = self.dnd_fn()

        if mode is not None and self._dnd is not None and mode != self._dnd:
            self._emit(f"focus mode {mode}")

        if mode is not None:
            self._dnd = mode

    def _check_assertions(self) -> None:
        if self.mono() - self._t_assertions < self.assertions_every:
            return

        self._t_assertions = self.mono()

        found = self.assertions_fn()

        if found is not None:
            self._assertions = found

    def _check_disk_io(self, dt: float) -> None:
        if self.disk_io_fn is None:
            return

        try:
            counters = self.disk_io_fn()

        except Exception:
            return

        if counters is None:
            return

        read_bytes = getattr(counters, "read_bytes", 0)

        write_bytes = getattr(counters, "write_bytes", 0)

        if self._prev_disk_io is not None and dt > 0:
            self._disk_rate = (
                max(0.0, (read_bytes - self._prev_disk_io[0]) / dt),
                max(0.0, (write_bytes - self._prev_disk_io[1]) / dt),
            )

        self._prev_disk_io = (read_bytes, write_bytes)

    # ------------------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        """Advance one cadence step; returns the full fact sheet."""
        now_wall = self.now()

        mono = self.mono()

        dt = mono - self._last_sample_mono if self._last_sample_mono is not None else 0.0

        self._last_sample_mono = mono

        self._check_clock_jump()

        idle = self.idle_fn()

        self._check_work(idle, dt)

        focus = self.focus_fn()

        self._check_focus(focus)

        self._check_displays()

        self._check_locked()

        self._check_dnd()

        self._check_assertions()

        self._check_disk_io(dt)

        metrics = self.metrics_fn()

        self._check_disk(metrics.get("disk"))

        self._check_runaway()

        read_rate, write_rate = self._disk_rate or (None, None)

        return {
            "now": now_wall,
            "identity": self.identity,
            "cpu": metrics.get("cpu"),
            "mem": metrics.get("mem"),
            "swap": metrics.get("swap"),
            "disk": metrics.get("disk"),
            "disk_read_bps": read_rate,
            "disk_write_bps": write_rate,
            "battery": metrics.get("battery"),
            "charging": metrics.get("charging"),
            "uptime": metrics.get("uptime"),
            "self_pid": metrics.get("self_pid"),
            "self_cpu": metrics.get("self_cpu"),
            "self_mem_mb": metrics.get("self_mem_mb"),
            "top_procs": self._top,
            "idle": idle,
            "work_seconds": self.work_seconds,
            "focus": focus,
            "locked": self._locked,
            "dnd": self._dnd,
            "assertions": self._assertions,
            "events": list(self.events),
            "notes": list(self.notes),
        }


# ============================================================================
# Rendering: the text that rides the prompt
# ============================================================================

WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _fmt_span(seconds: float | None) -> str | None:
    if seconds is None:
        return None

    if seconds < 90:
        return f"{int(seconds)}s"

    minutes = seconds / 60

    if minutes < 90:
        return f"{int(minutes)}m"

    return f"{minutes / 60:.1f}h"


def _fmt_bps(bps: float) -> str:
    if bps >= 1024 * 1024:
        return f"{bps / 1024 / 1024:.1f}MB/s"

    if bps >= 1024:
        return f"{bps / 1024:.0f}KB/s"

    return f"{bps:.0f}B/s"


def _severity(value: float | None, warn: float, critical: float) -> str | None:
    if value is None:
        return None

    if value >= critical:
        return "critical"

    if value >= warn:
        return "high"

    return "normal"


def _battery_severity(
    battery: float | None,
    charging: bool | None,
    low: float,
    critical: float,
) -> str | None:
    if battery is None:
        return None

    if charging:
        return "normal"

    if battery <= critical:
        return "critical"

    if battery <= low:
        return "low"

    return "normal"


def _idle_band(seconds: float | None) -> int | None:
    if seconds is None:
        return None

    if seconds < 120:
        return 0

    if seconds < 1800:
        return 1

    if seconds < 7200:
        return 2

    return 3


def _minute_key() -> str:
    return datetime.now().astimezone().strftime("%H:%M")


def system_values(
    facts: dict[str, Any],
    *,
    mem_warn: float = 85.0,
    mem_critical: float = 95.0,
    disk_warn: float = 90.0,
    disk_critical: float = 95.0,
    battery_low: float = 20.0,
    battery_critical: float = 10.0,
) -> dict[str, Any]:
    """Quantized state used for comparison; values, not text."""
    focus = facts.get("focus")

    idle = facts.get("idle")

    work = facts.get("work_seconds")

    awake = facts.get("uptime")

    cpu = facts.get("cpu")

    mem = facts.get("mem")

    disk = facts.get("disk")

    battery = facts.get("battery")

    charging = facts.get("charging")

    assertions = facts.get("assertions")

    return {
        "now": _minute_key(),
        "focus": focus[0] if focus else None,
        "idle": _idle_band(idle),
        "work": None if work is None else int(work // 900),
        "awake": None if awake is None else int(awake // 3600),
        "cpu": None if cpu is None else int(cpu // 10),
        "mem": None if mem is None else int(mem // 10),
        "disk": None if disk is None else int(disk // 5),
        "battery": (
            None
            if battery is None
            else (
                int(battery // 10),
                bool(charging),
                _battery_severity(battery, charging, battery_low, battery_critical),
            )
        ),
        "locked": facts.get("locked"),
        "dnd": facts.get("dnd"),
        "assertions": tuple(sorted(assertions)[:3]) if assertions else None,
        "mem_pressure": _severity(mem, mem_warn, mem_critical),
        "disk_pressure": _severity(disk, disk_warn, disk_critical),
    }


FIELD_ORDER = (
    "now",
    "focus",
    "idle",
    "work",
    "awake",
    "cpu",
    "mem",
    "disk",
    "battery",
    "locked",
    "dnd",
    "assertions",
    "mem_pressure",
    "disk_pressure",
)


def system_field_line(
    key: str,
    facts: dict[str, Any],
    *,
    mem_warn: float = 85.0,
    mem_critical: float = 95.0,
    disk_warn: float = 90.0,
    disk_critical: float = 95.0,
    battery_low: float = 20.0,
    battery_critical: float = 10.0,
) -> str | None:
    """One self-describing, absolute line for a changed field."""
    if key == "now":
        local = datetime.now().astimezone()

        offset = local.strftime("%z")

        return (
            f"- now: {local.strftime('%H:%M')} "
            f"{WEEKDAY_ZH[local.weekday()]} (UTC{offset[:3]})"
        )

    if key == "focus":
        focus = facts.get("focus")

        return f"- focus: {focus[0]}" if focus else "- focus: none"

    if key == "idle":
        return f"- idle: {_fmt_span(facts.get('idle')) or 'unknown'}"

    if key == "work":
        return f"- work: {_fmt_span(facts.get('work_seconds')) or '0s'}"

    if key == "awake":
        return f"- awake: {_fmt_span(facts.get('uptime')) or 'unknown'}"

    if key == "cpu":
        cpu = facts.get("cpu")

        return f"- cpu: {cpu:.0f}%" if cpu is not None else "- cpu: unknown"

    if key == "mem":
        mem = facts.get("mem")

        return f"- mem: {mem:.0f}%" if mem is not None else "- mem: unknown"

    if key == "disk":
        disk = facts.get("disk")

        return f"- disk: {disk:.0f}%" if disk is not None else "- disk: unknown"

    if key == "battery":
        battery = facts.get("battery")

        if battery is None:
            return "- battery: none"

        state = "charging" if facts.get("charging") else "on battery"

        severity = _battery_severity(
            battery,
            facts.get("charging"),
            battery_low,
            battery_critical,
        )

        if severity and severity != "normal":
            return f"- battery: {battery}% {severity} ({state})"

        return f"- battery: {battery}% {state}"

    if key == "locked":
        locked = facts.get("locked")

        if locked is None:
            return "- session: unknown"

        return "- session: locked" if locked else "- session: unlocked"

    if key == "dnd":
        mode = facts.get("dnd")

        return f"- focus mode: {mode}" if mode else "- focus mode: unknown"

    if key == "assertions":
        assertions = facts.get("assertions")

        return (
            f"- sleep held by: {'; '.join(sorted(assertions)[:3])}"
            if assertions
            else "- sleep held by: none"
        )

    if key == "mem_pressure":
        mem = facts.get("mem")

        if mem is None:
            return "- pressure: mem unknown"

        severity = _severity(mem, mem_warn, mem_critical)

        if severity == "normal":
            return f"- pressure: mem normal ({mem:.0f}%)"

        return f"- pressure: mem {mem:.0f}% ({severity})"

    if key == "disk_pressure":
        disk = facts.get("disk")

        if disk is None:
            return "- pressure: disk unknown"

        severity = _severity(disk, disk_warn, disk_critical)

        if severity == "normal":
            return f"- pressure: disk normal ({disk:.0f}%)"

        return f"- pressure: disk {disk:.0f}% ({severity})"

    return None


def render_delta(
    previous: dict[str, Any],
    current: dict[str, Any],
    facts: dict[str, Any],
    *,
    events: list[tuple[float, str]],
    mem_warn: float = 85.0,
    mem_critical: float = 95.0,
    disk_warn: float = 90.0,
    disk_critical: float = 95.0,
    battery_low: float = 20.0,
    battery_critical: float = 10.0,
) -> str:
    """Changed fields only; every line is absolute, never a ratio."""
    lines: list[str] = []

    for key in FIELD_ORDER:
        if previous.get(key) == current.get(key):
            continue

        # The pressure line already carries the resource value; do
        # not also emit the plain band line on the same turn.
        if key == "mem" and previous.get("mem_pressure") != current.get("mem_pressure"):
            continue

        if key == "disk" and previous.get("disk_pressure") != current.get("disk_pressure"):
            continue

        line = system_field_line(
            key,
            facts,
            mem_warn=mem_warn,
            mem_critical=mem_critical,
            disk_warn=disk_warn,
            disk_critical=disk_critical,
            battery_low=battery_low,
            battery_critical=battery_critical,
        )

        if line:
            lines.append(line)

    if events:
        lines.append("- events: " + " | ".join(text for _, text in events))

    return "\n".join(lines)


def _resource_line(
    facts: dict[str, Any],
    *,
    battery_low: float = 20.0,
    battery_critical: float = 10.0,
) -> str:
    cpu = facts.get("cpu")

    mem = facts.get("mem")

    disk = facts.get("disk")

    parts = [
        f"cpu {cpu:.0f}%" if cpu is not None else None,
        f"mem {mem:.0f}%" if mem is not None else None,
        f"disk {disk:.0f}%" if disk is not None else None,
    ]

    line = " | ".join(p for p in parts if p)

    battery = facts.get("battery")

    if battery is not None:
        state = "charging" if facts.get("charging") else "on battery"

        severity = _battery_severity(
            battery,
            facts.get("charging"),
            battery_low,
            battery_critical,
        )

        if severity and severity != "normal":
            line += f" | battery {battery}% {severity} ({state})"

        else:
            line += f" | battery {battery}% {state}"

    return line


def render_facts(
    facts: dict[str, Any],
    *,
    title_cap: int,
    events_limit: int,
    mem_warn: float = 85.0,
    mem_critical: float = 95.0,
    disk_warn: float = 90.0,
    disk_critical: float = 95.0,
    battery_low: float = 20.0,
    battery_critical: float = 10.0,
    note: bool = True,
) -> str:
    """The epoch's full fact block injected into the prompt.

    No header: the core labels the block by service_id when it
    packages injections into the observation.
    """
    lines = []

    identity = facts.get("identity") or {}

    host_parts = [
        identity.get("model"),
        identity.get("os"),
        identity.get("arch"),
    ]

    if any(host_parts):
        lines.append("- host: " + " ".join(str(part) for part in host_parts if part))

    local = datetime.now().astimezone()

    offset = local.strftime("%z")

    lines.append(
        f"- now: {local.strftime('%H:%M')} {WEEKDAY_ZH[local.weekday()]} (UTC{offset[:3]})",
    )

    focus = facts.get("focus")

    if focus:
        app, title = focus

        title_part = f" — {title[:title_cap]}" if title else ""

        lines.append(f"- focus: {app}{title_part}")

    idle = _fmt_span(facts.get("idle"))

    work = _fmt_span(facts.get("work_seconds"))

    awake = _fmt_span(facts.get("uptime"))

    lines.append(
        f"- idle: {idle or 'unknown'} | work: {work or '0s'} | awake: {awake or 'unknown'}",
    )

    lines.append(
        f"- {_resource_line(facts, battery_low=battery_low, battery_critical=battery_critical)}",
    )

    read_rate = facts.get("disk_read_bps")

    write_rate = facts.get("disk_write_bps")

    if read_rate is not None:
        lines.append(f"- disk io: r {_fmt_bps(read_rate)} w {_fmt_bps(write_rate or 0)}")

    locked = facts.get("locked")

    dnd = facts.get("dnd")

    session = [
        part
        for part in (
            "locked" if locked else None,
            f"focus mode {dnd}" if dnd and dnd != "off" else None,
        )
        if part
    ]

    if session:
        lines.append("- session: " + " | ".join(session))

    assertions = facts.get("assertions")

    if assertions:
        lines.append(f"- sleep held by: {'; '.join(sorted(assertions)[:3])}")

    mem = facts.get("mem")

    mem_severity = _severity(mem, mem_warn, mem_critical)

    if mem is not None and mem_severity != "normal":
        lines.append(f"- pressure: mem {mem:.0f}% ({mem_severity})")

    disk = facts.get("disk")

    disk_severity = _severity(disk, disk_warn, disk_critical)

    if disk is not None and disk_severity != "normal":
        lines.append(f"- pressure: disk {disk:.0f}% ({disk_severity})")

    events = facts.get("notes")

    if events is None:
        events = facts.get("events") or []

    if events:
        rendered = " | ".join(text for _, text in events[-events_limit:])

        lines.append(f"- events: {rendered}")

    self_cpu = facts.get("self_cpu")

    self_mem = facts.get("self_mem_mb")

    self_pid = facts.get("self_pid")

    if self_cpu is not None and self_mem is not None:
        pid_part = f" (pid {self_pid})" if self_pid else ""

        lines.append(f"- self: nan cpu {self_cpu:.1f}% mem {self_mem}MB{pid_part}")

    top = facts.get("top_procs") or []

    if top:
        rendered = ", ".join(f"{name} {cpu:.0f}% (pid {pid})" for pid, name, cpu in top)

        lines.append(f"- top: {rendered}")

    if note:
        lines.append("- note: snapshot for this epoch; only changed lines follow")

    return "\n".join(lines)


# ============================================================================
# Service shell: config, register, heartbeat, turn endpoint
# ============================================================================


class Config(ServiceConfig):
    """The system keys: thresholds and cadences for the probe."""

    poll_interval: float = 5.0

    disk_threshold: float = 90.0

    disk_announce_every: float = 600.0

    disk_critical: float = 95.0

    mem_warn: float = 85.0

    mem_critical: float = 95.0

    battery_low: float = 20.0

    battery_critical: float = 10.0

    state_ttl: float = 3600.0

    runaway_cpu: float = 90.0

    runaway_samples: int = 3

    idle_reset_after: float = 300.0

    assertions_every: float = 30.0

    top_cpu_threshold: float = 20.0

    top_procs_max: int = 3

    events_max: int = 8

    events_render_limit: int = 3

    focus_title_cap: int = 80


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    return read_config(Config, path)


@dataclass
class _AgentState:
    """What one agent has already been told; isolated per context."""

    epoch: int

    known: dict[str, Any]

    event_cursor: float

    turn_started_at: float

    last_text: str

    last_seen: float


class SystemService:
    """
    Owns the probe, the latest sample, the per-agent injection state,
    and the link that keeps this process registered with the hub.
    """

    def __init__(self, config: Config, *, link: HubLink | None = None) -> None:
        self.config = config

        self.link = link or HubLink(config)

        self.probe = SystemProbe(
            lock_fn=screen_locked,
            dnd_fn=focus_mode,
            assertions_fn=power_assertions,
            disk_io_fn=psutil.disk_io_counters,
            identity=static_identity(),
            disk_threshold=config.disk_threshold,
            disk_announce_every=config.disk_announce_every,
            runaway_cpu=config.runaway_cpu,
            runaway_samples=config.runaway_samples,
            idle_reset_after=config.idle_reset_after,
            assertions_every=config.assertions_every,
            top_cpu_threshold=config.top_cpu_threshold,
            top_procs_max=config.top_procs_max,
            events_max=config.events_max,
        )

        self._facts: dict[str, Any] | None = None

        self._states: dict[str, _AgentState] = {}

        self._executor = ThreadPoolExecutor(max_workers=1)

    # -- sampling ------------------------------------------------------

    def _sample_once(self) -> None:
        self._facts = self.probe.sample()

    async def _sample_loop(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            try:
                await loop.run_in_executor(self._executor, self._sample_once)

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception("system sample failed")

            await asyncio.sleep(self.config.poll_interval)

    # -- turn answering --------------------------------------------------

    def _prune(self, now: float) -> None:
        ttl = self.config.state_ttl

        for agent_id in [
            agent_id
            for agent_id, state in self._states.items()
            if now - state.last_seen > ttl
        ]:
            del self._states[agent_id]

    def contribution(self, turn: dict[str, Any] | None = None) -> str:
        facts = self._facts

        if facts is None:
            return ""

        turn = turn or {}

        agent_id = str(turn.get("agent_id") or "root")

        started = float(turn.get("started_at") or 0.0)

        epoch = int(turn.get("epoch") or 0)

        origin = str(turn.get("origin") or "normal")

        depth = int(turn.get("depth") or 0)

        now = time.time()

        self._prune(now)

        state = self._states.get(agent_id)

        # The hub re-injects a parked turn with the same started_at;
        # replay the exact text and do not advance the baseline again.
        if state is not None and state.turn_started_at == started:
            state.last_seen = now

            return state.last_text

        values = system_values(
            facts,
            mem_warn=self.config.mem_warn,
            mem_critical=self.config.mem_critical,
            disk_warn=self.config.disk_threshold,
            disk_critical=self.config.disk_critical,
            battery_low=self.config.battery_low,
            battery_critical=self.config.battery_critical,
        )

        notes = facts.get("notes") or []

        full = (
            state is None
            or state.epoch != epoch
            or origin in ("first", "context_cleared")
        )

        if full:
            rendered_facts = facts

            if origin == "first" and depth > 0 and started:
                rendered_facts = {
                    **facts,
                    "notes": [
                        (ts, text)
                        for ts, text in notes
                        if ts >= started
                    ],
                }

            text = render_facts(
                rendered_facts,
                title_cap=self.config.focus_title_cap,
                events_limit=self.config.events_render_limit,
                mem_warn=self.config.mem_warn,
                mem_critical=self.config.mem_critical,
                disk_warn=self.config.disk_threshold,
                disk_critical=self.config.disk_critical,
                battery_low=self.config.battery_low,
                battery_critical=self.config.battery_critical,
            )

            cursor = max((ts for ts, _ in notes), default=0.0)

        else:
            fresh = [
                (ts, text)
                for ts, text in notes
                if ts > state.event_cursor
            ]

            text = render_delta(
                state.known,
                values,
                facts,
                events=fresh,
                mem_warn=self.config.mem_warn,
                mem_critical=self.config.mem_critical,
                disk_warn=self.config.disk_threshold,
                disk_critical=self.config.disk_critical,
                battery_low=self.config.battery_low,
                battery_critical=self.config.battery_critical,
            )

            cursor = max((ts for ts, _ in fresh), default=state.event_cursor)

        self._states[agent_id] = _AgentState(
            epoch=epoch,
            known=values,
            event_cursor=cursor,
            turn_started_at=started,
            last_text=text,
            last_seen=now,
        )

        return text


def build_app(service: SystemService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with service.link:
            sampler = asyncio.create_task(service._sample_loop(), name="system-sample")

            logger.info(
                "system service up on {}:{} (hub {})",
                service.config.host,
                service.config.port,
                service.config.hub_url,
            )

            try:
                yield

            finally:
                sampler.cancel()

                with contextlib.suppress(asyncio.CancelledError):
                    await sampler

    app = FastAPI(lifespan=lifespan)

    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        text = service.contribution(body)

        return {"text": text} if text else {"text": ""}

    return app


def main() -> None:
    config = load_config()

    setup_logging(config)

    service = SystemService(config)

    uvicorn.run(
        build_app(service),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
