"""
Network self-perception toolkit: deterministic facts only.

Same disciplines as utils/system.py:

    push-only  rates, connections, reachability cadence --
               events are the reason this module exists
    facts only no heuristics, no conclusions; the agent judges
    degrade    every probe/shim returns None on failure; the
               module never dies from a missing network

All outbound queries (online probe, public IP) hit configurable
endpoints and are optional facts, never blocking the caller
beyond their own timeout inside the sampling thread.
"""

from __future__ import annotations

import ipaddress
import platform
import re
import subprocess
import time
from collections import deque
from typing import Any, Callable

import httpx
import psutil


try:
    import mss  # noqa: F401  (display sanity only)

except Exception:

    pass


MAX_EVENTS = 8

TUNNEL_PATTERN = re.compile(
    r"^(tun|tap|ppp|wg|ipsec|tailscale|gpd)",
    re.I,
)

PRIVATE_V4_CHECK = None


DEFAULT_PROBE_URLS = [
    "http://connect.rom.miui.com/generate_204",
    "https://www.baidu.com",
    "https://api.ipify.org",
]

DEFAULT_IP_URLS = [
    "https://api.ipify.org",
    "https://ip.3322.net",
    "https://ifconfig.me/ip",
]


def _run(cmd: str, timeout: float = 4.0) -> str:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result.stdout or ""

    except Exception:

        return ""


def is_public_ip(ip: str) -> bool:
    """
    is_global alone wrongly reports IPv4 multicast as public
    (CPython quirk); exclude multicast/reserved explicitly.
    """
    try:
        addr = ipaddress.ip_address(ip.strip())

        return (
            addr.is_global
            and not addr.is_multicast
            and not addr.is_reserved
        )

    except Exception:

        return False


# ======================================================================
# Shims
# ======================================================================


def default_gateway() -> tuple[str, str] | None:
    """(gateway_ip, interface); interface may be ''."""
    system = platform.system()

    if system == "Darwin":
        out = _run("route -n get default")

        gw = re.search(r"gateway:\s*(\d+\.\d+\.\d+\.\d+)", out)

        iface = re.search(r"interface:\s*(\S+)", out)

        if gw:
            return gw.group(1), iface.group(1) if iface else ""

        return None

    if system == "Linux":
        out = _run("ip route show default")

        match = re.search(
            r"default via (\S+)(?: dev (\S+))?", out
        )

        if match:
            return match.group(1), match.group(2) or ""

        return None

    if system == "Windows":
        out = _run("route print 0.0.0.0")

        match = re.search(
            r"0\.0\.0\.0\s+0\.0\.0\.0\s+(\d+\.\d+\.\d+\.\d+)", out
        )

        if match:
            return match.group(1), ""

        return None

    return None


def probe_online(
    urls: list[str] | None = None,
    timeout: float = 3.0,
) -> float | None:
    """Latency ms of the first responsive endpoint; None offline."""
    for url in urls or DEFAULT_PROBE_URLS:
        started = time.perf_counter()

        try:
            httpx.head(
                url,
                timeout=timeout,
                follow_redirects=False,
            )

        except Exception:

            continue

        return (time.perf_counter() - started) * 1000.0

    return None


def public_ip(
    urls: list[str] | None = None,
    timeout: float = 3.0,
) -> str | None:
    for url in urls or DEFAULT_IP_URLS:
        try:
            text = httpx.get(url, timeout=timeout).text.strip()

        except Exception:

            continue

        if is_public_ip(text):
            return text

    return None


def wifi_ssid() -> str | None:
    """Moved here from utils/system (network owns the link)."""
    system = platform.system()

    if system == "Darwin":
        for device in ("en0", "en1"):
            out = _run(f"ipconfig getsummary {device}")

            match = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)

            if match:
                return match.group(1).strip()

        return None

    if system == "Linux":
        return _run("iwgetid -r").strip() or None

    if system == "Windows":
        out = _run("netsh wlan show interfaces")

        match = re.search(r"^\s*SSID\s*:\s*(.+)$", out, re.M)

        return match.group(1).strip() if match else None

    return None


def _net_io(pernic: bool = True):
    return psutil.net_io_counters(pernic=pernic, nowrap=True)


def _net_connections():
    """
    Yields (conn, pid, owner_name). System-wide first (Linux/Win;
    psutil fills conn.pid), then per-process fallback for macOS
    AccessDenied: only the agent's own user's connections -- an
    honest fact boundary.
    """
    try:
        conns = psutil.net_connections(kind="inet")

        out = []

        for conn in conns:
            name = "unknown"

            if conn.pid:
                try:
                    name = psutil.Process(conn.pid).name()

                except Exception:

                    pass

            out.append((conn, conn.pid, name))

        return out

    except Exception:

        out = []

        for proc in psutil.process_iter(["name"]):
            name = proc.info.get("name") or "unknown"

            try:
                for conn in proc.net_connections(kind="inet"):
                    out.append((conn, proc.pid, name))

            except Exception:

                continue

        return out


# ======================================================================
# Probe
# ======================================================================


class NetworkProbe:
    """
    One sample per call; deterministic; events detected by
    diffing. Clock and all network shims injectable.
    """

    def __init__(
        self,
        *,
        now: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        gw_fn: Callable[
            [], tuple[str, str] | None
        ] = default_gateway,
        probe_fn: Callable[
            [], float | None
        ] = probe_online,
        ip_fn: Callable[[], str | None] = public_ip,
        ssid_fn: Callable[[], str | None] = wifi_ssid,
        io_fn: Callable = _net_io,
        conns_fn: Callable = _net_connections,
        online_every: float = 15.0,
        ip_every: float = 600.0,
        conns_every: float = 15.0,
        offline_after: int = 2,
        spike_ratio: float = 8.0,
    ) -> None:
        self.now = now

        self.mono = mono

        self.gw_fn = gw_fn

        self.probe_fn = probe_fn

        self.ip_fn = ip_fn

        self.ssid_fn = ssid_fn

        self.io_fn = io_fn

        self.conns_fn = conns_fn

        self.online_every = online_every

        self.ip_every = ip_every

        self.conns_every = conns_every

        self.offline_after = offline_after

        self.spike_ratio = spike_ratio

        self.events: deque[tuple[float, str]] = deque(
            maxlen=MAX_EVENTS,
        )

        self._t_online = -1e9

        self._t_ip = -1e9

        self._t_conns = -1e9

        self._fail_count = 0

        self._offline_since: float | None = None

        self._latency: float | None = None

        self._public: str | None = None

        self._public_at: float | None = None

        self._prev_io: dict[str, tuple[int, int]] | None = None

        self._rates: dict[str, tuple[float, float]] = {}

        self._seen_outbound: dict[tuple, float] = {}

        self._seen_listen: set[tuple] = set()

        self._proc_names: dict[int, str] = {}

        self._baseline: tuple[float, float] | None = None

    # ------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self.events.append((self.now(), text))

    def _proc_name(
        self,
        pid: int | None,
        name: str | None = None,
    ) -> str:
        if pid is not None and pid in self._proc_names:
            return self._proc_names[pid]

        if not name:
            try:
                name = psutil.Process(pid).name()

            except Exception:

                name = "unknown"

        if pid is not None:
            self._proc_names[pid] = name

        return name

    def _check_online(self) -> None:
        if self.mono() - self._t_online < self.online_every:
            return

        self._t_online = self.mono()

        latency = self.probe_fn()

        if latency is None:
            self._fail_count += 1

            if (
                self._fail_count >= self.offline_after
                and self._offline_since is None
            ):
                self._offline_since = self.now()

                self._emit("offline")

            self._latency = None

            return

        self._latency = latency

        if self._offline_since is not None:
            span = self.now() - self._offline_since

            self._offline_since = None

            self._fail_count = 0

            self._emit(
                f"offline {_fmt_span(span)} -> online"
            )

        elif self._fail_count:
            self._fail_count = 0

    def _check_public_ip(self) -> None:
        if self.mono() - self._t_ip < self.ip_every:
            return

        if self._offline_since is not None:
            return

        self._t_ip = self.mono()

        ip = self.ip_fn()

        if ip and ip != self._public:
            if self._public is not None:
                self._emit(f"public ip -> {ip}")

            self._public = ip

            self._public_at = self.now()

    def _check_traffic(self, dt: float) -> None:
        try:
            counters = self.io_fn()

        except Exception:

            return

        if not counters or dt <= 0:
            self._prev_io = counters

            return

        rates: dict[str, tuple[float, float]] = {}

        for name, io in counters.items():
            prev = self._prev_io.get(name) if self._prev_io else None

            if prev is None:
                continue

            dr = (io.bytes_recv - prev[0]) / dt

            ds = (io.bytes_sent - prev[1]) / dt

            if dr < 0 or ds < 0:
                continue

            rates[name] = (dr, ds)

        self._prev_io = {
            name: (io.bytes_recv, io.bytes_sent)
            for name, io in counters.items()
        }

        self._rates = rates

        # Spike detection on the summed rate (deterministic).
        total_down = sum(r for r, _ in rates.values())

        total_up = sum(s for _, s in rates.values())

        if self._baseline is None:
            if total_down > 0:
                self._baseline = (total_down, total_up)

            return

        if total_down > self._baseline[0] * self.spike_ratio and total_down > 1e6:
            self._emit(
                f"traffic spike ↓{_fmt_bps(total_down)}"
            )

            self._baseline = (total_down, total_up)

        elif total_down < self._baseline[0] / 2:
            self._baseline = (total_down, total_up)

    def _check_connections(self) -> None:
        if self.mono() - self._t_conns < self.conns_every:
            return

        self._t_conns = self.mono()

        try:
            conns = self.conns_fn()

        except Exception:

            return

        est = 0

        public = 0

        listen_keys: set[tuple] = set()

        now_mono = self.mono()

        for conn, pid, name in conns:
            if pid is not None:
                self._proc_names[pid] = name

            status = getattr(conn, "status", "")

            if status == psutil.CONN_LISTEN:
                lport = conn.laddr.port if conn.laddr else 0

                key = (lport, pid)

                listen_keys.add(key)

                continue

            if status != psutil.CONN_ESTABLISHED:
                continue

            est += 1

            raddr = conn.raddr

            if raddr is None or not is_public_ip(raddr.ip):
                continue

            public += 1

            key = (
                self._proc_name(pid, name),
                raddr.ip,
                raddr.port,
            )

            if key not in self._seen_outbound:
                self._seen_outbound[key] = now_mono

                if len(self._seen_outbound) > 512:
                    cutoff = now_mono - 3600

                    self._seen_outbound = {
                        k: v
                        for k, v in self._seen_outbound.items()
                        if v > cutoff
                    }

                self._emit(
                    f"new outbound: {key[0]} -> "
                    f"{key[1]}:{key[2]}"
                )

        listen_count = len(listen_keys)

        new_listen = listen_keys - self._seen_listen

        if self._seen_listen and new_listen:
            for lport, pid in sorted(new_listen)[:2]:
                self._emit(
                    f"listen added: {lport} "
                    f"({self._proc_name(pid)})"
                )

        self._seen_listen = listen_keys

        self._conn_stats = (
            est,
            public,
            listen_count,
        )

    # ------------------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        mono = self.mono()

        now_wall = self.now()

        dt = 5.0

        self._check_online()

        self._check_public_ip()

        self._check_traffic(dt)

        self._check_connections()

        ssid = self.ssid_fn()

        gw = self.gw_fn()

        addrs = None

        primary_if = None

        local_ip = None

        try:
            stats = psutil.net_if_stats()

            addrs = psutil.net_if_addrs()

            candidates = []

            for name, addr_list in addrs.items():
                if name.startswith(("lo", "utun", "tun", "tap")):
                    continue

                if not stats.get(
                    name
                ) or not stats[name].isup:
                    continue

                for addr in addr_list:
                    if (
                        addr.family.name
                        == "AF_INET"
                        and not addr.address.startswith("169.254")
                    ):
                        candidates.append((name, addr.address))

                        break

            if candidates:
                candidates.sort(
                    key=lambda pair: (
                        not pair[0].startswith(("en", "eth", "wl")),
                        pair[0],
                    )
                )

                primary_if, local_ip = candidates[0]

        except Exception:

            pass

        tunnel = None

        if gw and gw[1] and re.search(
            r"(tun|tap|ppp|wg|ipsec|tailscale)",
            gw[1],
            re.I,
        ):
            # Default route riding a tunnel interface IS the
            # VPN fact; macOS system utuns never carry it.
            tunnel = gw[1]

        est, public, listen = getattr(
            self, "_conn_stats", (None, None, None),
        )

        total_down = sum(r for r, _ in self._rates.values())

        total_up = sum(s for _, s in self._rates.values())

        return {
            "now": now_wall,
            "primary_if": primary_if,
            "local_ip": local_ip,
            "ssid": ssid,
            "gw": gw[0] if gw else None,
            "gw_iface": gw[1] if gw else None,
            "tunnel": tunnel,
            "online": self._offline_since is None,
            "latency": self._latency,
            "offline_since": self._offline_since,
            "public_ip": self._public,
            "dl": total_down if self._rates else None,
            "ul": total_up if self._rates else None,
            "est": est,
            "public_conns": public,
            "listen": listen,
            "events": list(self.events),
        }


def _fmt_span(seconds: float) -> str:
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
