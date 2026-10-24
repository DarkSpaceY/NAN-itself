"""
System service: the agent feeling its own body.

An injection service (see docs/for-service-authors): registers
with the hub, samples the host on a cadence, and answers every
turn request with the latest rendered facts.

Disciplines:

    facts only  every line is a directly verifiable measurement
                or a deterministic event; the module measures,
                the agent judges
    degrade     every platform shim returns None on failure; the
                service never dies from a missing binary
    serve fast  sampling runs in the background; /turn replies
                with the ready text, never computing on request

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

import asyncio
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
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
import uvicorn
import yaml
from fastapi import FastAPI
from loguru import logger

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

            self._emit(f"display {sign}{abs(count - self._displays)}")

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
                    self._emit(f"runaway: {name} {cpu:.0f}%")

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


def _resource_line(facts: dict[str, Any]) -> str:
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

        line += f" | battery {battery}% {state}"

    return line


def render_facts(
    facts: dict[str, Any],
    *,
    title_cap: int,
    events_limit: int,
) -> str:
    """The compact fact block injected into the prompt.

    No header: the core labels the block by service_id when it
    packages injections into the observation.
    """
    lines = []

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

    lines.append(f"- {_resource_line(facts)}")

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
        lines.append(f"- sleep held by: {'; '.join(assertions[:3])}")

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

    return "\n".join(lines)


# ============================================================================
# Service shell: config, register, heartbeat, turn endpoint
# ============================================================================


class Config:
    """Typed view of config.yaml; unknown keys are ignored."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.hub_url: str = str(raw.get("hub_url", "http://127.0.0.1:8901")).rstrip("/")

        self.service_id: str = str(raw.get("service_id", "system"))

        self.host: str = str(raw.get("host", "127.0.0.1"))

        self.port: int = int(raw.get("port", 9001))

        self.poll_interval: float = float(raw.get("poll_interval", 5.0))

        self.disk_threshold: float = float(raw.get("disk_threshold", 90.0))

        self.disk_announce_every: float = float(raw.get("disk_announce_every", 600.0))

        self.runaway_cpu: float = float(raw.get("runaway_cpu", 90.0))

        self.runaway_samples: int = int(raw.get("runaway_samples", 3))

        self.idle_reset_after: float = float(raw.get("idle_reset_after", 300.0))

        self.assertions_every: float = float(raw.get("assertions_every", 30.0))

        self.top_cpu_threshold: float = float(raw.get("top_cpu_threshold", 20.0))

        self.top_procs_max: int = int(raw.get("top_procs_max", 3))

        self.events_max: int = int(raw.get("events_max", 8))

        self.events_render_limit: int = int(raw.get("events_render_limit", 3))

        self.focus_title_cap: int = int(raw.get("focus_title_cap", 80))


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return Config(raw)


class SystemService:
    """
    Owns the probe, the latest sample and its rendered text, and
    the hub lifecycle (register / heartbeat / unregister).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

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

        self._rendered: str | None = None

        self._executor = ThreadPoolExecutor(max_workers=1)

        self._client = httpx.Client(timeout=5.0)

    # -- sampling ------------------------------------------------------

    def _sample_once(self) -> None:
        facts = self.probe.sample()

        self._facts = facts

        self._rendered = render_facts(
            facts,
            title_cap=self.config.focus_title_cap,
            events_limit=self.config.events_render_limit,
        )

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

    # -- hub lifecycle ---------------------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> None:
        try:
            self._client.post(f"{self.config.hub_url}{path}", json=body)

        except Exception:
            logger.warning("hub {} unreachable", path)

    async def _heartbeat_loop(self) -> None:
        while True:
            self._post("/heartbeat", {"service_id": self.config.service_id})

            await asyncio.sleep(5.0)

    def register(self) -> None:
        self._post(
            "/register",
            {
                "service_id": self.config.service_id,
                "endpoint": f"http://{self.config.host}:{self.config.port}/turn",
            },
        )

    def unregister(self) -> None:
        self._post("/unregister", {"service_id": self.config.service_id})

    # -- turn answering --------------------------------------------------

    def contribution(self) -> str | None:
        return self._rendered


def build_app(service: SystemService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.register()

        sampler = asyncio.create_task(service._sample_loop(), name="system-sample")

        beater = asyncio.create_task(service._heartbeat_loop(), name="system-heartbeat")

        logger.info(
            "system service up on {}:{} (hub {})",
            service.config.host,
            service.config.port,
            service.config.hub_url,
        )

        yield

        sampler.cancel()

        beater.cancel()

        service.unregister()

    app = FastAPI(lifespan=lifespan)

    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        text = service.contribution()

        return {"text": text} if text else {"text": ""}

    return app


def main() -> None:
    config = load_config()

    logger.add(
        HERE / "system.log",
        rotation="5 MB",
        retention=2,
        level="INFO",
    )

    service = SystemService(config)

    uvicorn.run(
        build_app(service),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
