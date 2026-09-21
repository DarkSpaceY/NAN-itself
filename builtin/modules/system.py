# @module

"""
System: the agent feeling its own body.

Facts only -- every line in query() is a directly verifiable
measurement or a deterministic event; no heuristics, no
conclusions. The agent judges, this module measures.

All heavy lifting lives in the SYSTEM TOOLKIT inlined below; this
module is the async shell: park, sample on a cadence, publish,
render.
"""

from __future__ import annotations

import asyncio
import ctypes
import platform
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable

import mss
from loguru import logger

try:
    import psutil

    HAS_PSUTIL = True

except Exception:

    psutil = None

    HAS_PSUTIL = False


# ============================================================================
# SYSTEM TOOLKIT (inlined from backend/nan_itself/utils/system.py)
#
# System self-perception toolkit: deterministic facts only.
#
# Three disciplines (docs/system-features.md):
#
#     1. push-only   rhythm/events/state-trends here; pull-able
#                    state belongs to tools (touchpoint)
#     2. facts only  zero heuristics, zero conclusions -- the
#                    module measures, the agent judges
#     3. degrade     every platform shim returns None on failure;
#                    the loop never dies from a missing binary
#
# Platform shims (idle/foreground/wifi) are tiny per-OS blocks,
# injected and monkeypatchable; tests never touch real hardware.
# ============================================================================


MAX_EVENTS = 8


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


# ======================================================================
# Platform shims (each returns None when unknown)
# ======================================================================


def idle_seconds() -> float | None:
    """Seconds since last user input."""
    system = platform.system()

    if system == "Darwin":
        out = _run(
            "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ "
            "{print $NF/1000000000; exit}'",
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

        if ctypes.windll.user32.GetLastInputInfo(
            ctypes.byref(info)
        ):
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
        out = _run(
            "xdotool getactivewindow getwindowname"
        ).strip()

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
    try:
        monitors = mss.mss().monitors

        return max(0, len(monitors) - 1)  # [0] is the union

    except Exception:

        return None


# ======================================================================
# Probe: one deterministic sample + event detection
# ======================================================================


def _psutil_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {
        "cpu": None,
        "mem": None,
        "swap": None,
        "disk": None,
        "battery": None,
        "charging": None,
        "net_up": None,
        "uptime": None,
        "self_cpu": None,
        "self_mem_mb": None,
    }

    if not HAS_PSUTIL:
        return out

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

    me = psutil.Process()

    out["self_cpu"] = me.cpu_percent(interval=None)

    out["self_mem_mb"] = round(
        me.memory_info().rss / (1024 * 1024)
    )

    return out


def _top_processes(
    cache: dict[int, tuple[Any, str]],
) -> list[tuple[int, str, float]]:
    """[(pid, name, cpu%)] using cached Process objects."""
    out: list[tuple[int, str, float]] = []

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            entry = cache.get(proc.pid)

            if entry is None:
                entry = (proc, proc.info["name"] or "?")

                cache[proc.pid] = entry

            out.append(
                (proc.pid, entry[1], entry[0].cpu_percent(None))
            )

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
    diffing against the previous sample. Clock shims injectable
    for tests.
    """

    def __init__(
        self,
        *,
        now: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        idle_fn: Callable[[], float | None] = idle_seconds,
        focus_fn: Callable[
            [], tuple[str, str] | None
        ] = foreground_app,
        displays_fn: Callable[[], int | None] = display_count,
        metrics_fn: Callable[
            [], dict[str, Any]
        ] = _psutil_metrics,
        procs_fn: Callable[
            [dict], list
        ] = _top_processes,
        disk_threshold: float = 90.0,
        runaway_cpu: float = 90.0,
        runaway_samples: int = 3,
    ) -> None:
        self.now = now

        self.mono = mono

        self.idle_fn = idle_fn

        self.focus_fn = focus_fn

        self.displays_fn = displays_fn

        self.metrics_fn = metrics_fn

        self.procs_fn = procs_fn

        self.disk_threshold = disk_threshold

        self.runaway_cpu = runaway_cpu

        self.runaway_samples = runaway_samples

        self.events: deque[tuple[float, str]] = deque(
            maxlen=MAX_EVENTS,
        )

        self._clock_offset: float | None = None

        self._focus: tuple[str, str] | None = None

        self._displays: int | None = None

        self._disk_announced_at = 0.0

        self._runaway: dict[int, int] = {}

        self._proc_cache: dict[int, tuple[Any, str]] = {}

        self._last_sample_mono: float | None = None

        self.work_seconds = 0.0

    # ------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self.events.append((self.now(), text))

    def _check_clock_jump(self) -> None:
        offset = self.now() - self.mono()

        if self._clock_offset is not None:
            jump = offset - self._clock_offset

            if jump > 60.0:
                self._emit("woke from sleep")

        self._clock_offset = offset

    def _check_work(self, idle: float | None, dt: float) -> None:
        if idle is None:
            return

        if idle >= 300.0:
            self.work_seconds = 0.0

        else:
            if self._last_sample_mono is not None:
                self.work_seconds += min(dt, 30.0)

    def _check_focus(self, focus) -> None:
        if (
            focus
            and self._focus
            and focus[0] != self._focus[0]
        ):
            self._emit(f"focus -> {focus[0]}")

        if focus:
            self._focus = focus

    def _check_displays(self) -> None:
        count = self.displays_fn()

        if (
            count is not None
            and self._displays is not None
            and count != self._displays
        ):
            sign = "+" if count > self._displays else "-"

            self._emit(f"display {sign}{abs(count - self._displays)}")

        if count is not None:
            self._displays = count

    def _check_disk(self, disk: float | None) -> None:
        if disk is None or disk < self.disk_threshold:
            return

        now = self.now()

        if now - self._disk_announced_at >= 600.0:
            self._disk_announced_at = now

            self._emit(f"disk {disk:.0f}%")

    def _check_runaway(self) -> None:
        if not HAS_PSUTIL:
            return

        for pid, name, cpu in self.procs_fn(self._proc_cache):
            if cpu >= self.runaway_cpu:
                count = self._runaway.get(pid, 0) + 1

                self._runaway[pid] = count

                if count == self.runaway_samples:
                    self._emit(
                        f"runaway: {name} {cpu:.0f}%"
                    )

            else:
                self._runaway.pop(pid, None)

    # ------------------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        now_wall = self.now()

        mono = self.mono()

        dt = (
            mono - self._last_sample_mono
            if self._last_sample_mono is not None
            else 0.0
        )

        self._last_sample_mono = mono

        self._check_clock_jump()

        idle = self.idle_fn()

        self._check_work(idle, dt)

        focus = self.focus_fn()

        self._check_focus(focus)

        self._check_displays()

        metrics = self.metrics_fn()

        self._check_disk(metrics.get("disk"))

        self._check_runaway()

        focus = self.focus_fn()

        return {
            "now": now_wall,
            "cpu": metrics.get("cpu"),
            "mem": metrics.get("mem"),
            "swap": metrics.get("swap"),
            "disk": metrics.get("disk"),
            "battery": metrics.get("battery"),
            "charging": metrics.get("charging"),
            "net_up": metrics.get("net_up"),
            "uptime": metrics.get("uptime"),
            "self_cpu": metrics.get("self_cpu"),
            "self_mem_mb": metrics.get("self_mem_mb"),
            "idle": idle,
            "work_seconds": self.work_seconds,
            "focus": focus,
            "events": list(self.events),
        }


# ============================================================================
# MODULE
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

    return f"{hours:.1f}h" if (hours := minutes / 60) else None


class SystemModule(Module):
    id = "system"

    poll_interval: float = 5.0

    render_preview_cap: int = 80

    events_render_limit: int = 3

    def __init__(self) -> None:
        self.probe = SystemProbe()

        self._latest: dict | None = None

    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not HAS_PSUTIL:
            logger.warning("psutil missing; system module degraded")

        logger.info("system module sampling every {}s", self.poll_interval)

        while True:
            try:
                facts = await asyncio.to_thread(self.probe.sample)

                self._latest = facts

                self.data.publish(
                    {
                        key: value
                        for key, value in facts.items()
                        if key != "events"
                    }
                    | {"events": facts.get("events", [])[-3:]}
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception("system sample failed")

            await asyncio.sleep(self.poll_interval)

    async def on_turn(self, record) -> None:
        pass

    # ------------------------------------------------------------------

    async def query(self, turn: Turn) -> str | None:
        facts = self._latest

        if facts is None:
            return None

        lines = ["[System]"]

        local = datetime.now().astimezone()

        offset = local.strftime("%z")

        lines.append(
            f"- now: {local.strftime('%H:%M')} "
            f"{WEEKDAY_ZH[local.weekday()]} "
            f"(UTC{offset[:3]})"
        )

        focus = facts.get("focus")

        if focus:
            app, title = focus

            title_part = (
                f" — {title[: self.render_preview_cap]}"
                if title
                else ""
            )

            lines.append(f"- focus: {app}{title_part}")

        idle = _fmt_span(facts.get("idle"))

        work = _fmt_span(facts.get("work_seconds"))

        awake = _fmt_span(facts.get("uptime"))

        lines.append(
            f"- idle: {idle or 'unknown'}"
            f" | work: {work or '0s'}"
            f" | awake: {awake or 'unknown'}"
        )

        lines.append(f"- {self._resource_line(facts)}")

        events = facts.get("events") or []

        if events:
            recent = events[-self.events_render_limit:]

            rendered = " | ".join(
                f"{text}"
                for _, text in recent
            )

            lines.append(f"- events: {rendered}")

        self_cpu = facts.get("self_cpu")

        self_mem = facts.get("self_mem_mb")

        if self_cpu is not None and self_mem is not None:
            lines.append(
                f"- self: nan cpu {self_cpu:.1f}%"
                f" mem {self_mem}MB"
            )

        return "\n".join(lines)

    def _resource_line(self, facts: dict) -> str:
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
            state = (
                "charging"
                if facts.get("charging")
                else "on battery"
            )

            line += f" | battery {battery}% {state}"

        return line
