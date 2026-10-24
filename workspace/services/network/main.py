"""
Network service: the agent feeling its long-range nerves.

An injection service (see docs/for-service-authors): registers
with the hub, samples the link on a cadence, and answers every
turn request with the latest rendered facts.

Disciplines:

    facts only  rates, connections, reachability, public
                identity; no conclusions ("untrusted network"
                belongs to the agent)
    degrade     every probe/shim returns None on failure; the
                service never dies from a missing network
    serve fast  sampling runs in the background; /turn replies
                with the ready text, never computing on request

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import platform
import re
import socket
import subprocess
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import psutil
import uvicorn
import yaml
from fastapi import FastAPI
from loguru import logger

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"


def _run(cmd: str, timeout: float = 4.0) -> str:
    """One platform probe; any failure -> empty string."""
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

        return addr.is_global and not addr.is_multicast and not addr.is_reserved

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

        match = re.search(r"default via (\S+)(?: dev (\S+))?", out)

        if match:
            return match.group(1), match.group(2) or ""

        return None

    if system == "Windows":
        out = _run("route print 0.0.0.0")

        match = re.search(r"0\.0\.0\.0\s+0\.0\.0\.0\s+(\d+\.\d+\.\d+\.\d+)", out)

        if match:
            return match.group(1), ""

        return None

    return None


def probe_online(urls: list[str], timeout: float) -> tuple[float, int | None] | None:
    """
    (latency ms, probe code) of the first responsive endpoint;
    None when offline. probe_code is None when the first URL (a
    204 endpoint, captive.apple.com) answered 204 as expected,
    else the HTTP code actually received (captive portal, DNS
    hijack, ...). Judging the anomaly is the agent's job.
    """
    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        for index, url in enumerate(urls):
            started = time.perf_counter()

            try:
                response = client.head(url)

            except Exception:
                continue

            latency = (time.perf_counter() - started) * 1000.0

            code = response.status_code if index == 0 else None

            return latency, None if code == 204 else code

    return None


def dns_resolves(host: str) -> bool | None:
    """True/False on a clean answer; None only on resolver errors
    that are not name failures."""
    try:
        socket.getaddrinfo(host, None)

        return True

    except socket.gaierror:
        return False

    except Exception:
        return None


def ping_gateway(ip: str) -> bool:
    """One ICMP echo; True only on a reply carrying a time."""
    system = platform.system()

    if system == "Darwin":
        out = _run(f"ping -c 1 -t 2 {ip}")

    elif system == "Linux":
        out = _run(f"ping -c 1 -W 2 {ip}")

    else:
        out = _run(f"ping -n 1 -w 2000 {ip}")

    return "time=" in out.lower()


def arp_table() -> dict[str, str] | None:
    """ip -> mac for the LAN neighbor cache; None on failure."""
    out = _run("arp -a")

    if not out:
        return None

    entries: dict[str, str] = {}

    for line in out.splitlines():
        match = re.match(r"\S+ \((\d+\.\d+\.\d+\.\d+)\) at (\S+)", line)

        if match:
            entries[match.group(1)] = match.group(2)

    return entries


def public_ip(urls: list[str], timeout: float) -> str | None:
    with httpx.Client(timeout=timeout) as client:
        for url in urls:
            try:
                text = client.get(url).text.strip()

            except Exception:
                continue

            if is_public_ip(text):
                return text

    return None


def wifi_ssid() -> str | None:
    """The network owns the link, so SSID lives here."""
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


def _net_connections() -> list[tuple[Any, int | None, str]]:
    """
    Returns [(conn, pid, owner_name)]. System-wide first; on
    macOS AccessDenied, per-process fallback for only the
    service's own user's connections -- an honest fact boundary.
    """
    try:
        conns = psutil.net_connections(kind="inet")

        out = []

        for conn in conns:
            name = "unknown"

            if conn.pid:
                with contextlib.suppress(Exception):
                    name = psutil.Process(conn.pid).name()

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
        probe_urls: list[str] | None = None,
        ip_urls: list[str] | None = None,
        probe_timeout: float = 3.0,
        now: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        gw_fn: Callable[[], tuple[str, str] | None] = default_gateway,
        probe_fn: Callable[[list[str], float], tuple[float, int | None] | None] = probe_online,
        ip_fn: Callable[[list[str], float], str | None] = public_ip,
        ssid_fn: Callable[[], str | None] = wifi_ssid,
        dns_fn: Callable[[str], bool | None] = dns_resolves,
        gw_ping_fn: Callable[[str], bool] = ping_gateway,
        arp_fn: Callable[[], dict[str, str] | None] = arp_table,
        io_fn: Callable[..., Any] = _net_io,
        conns_fn: Callable[[], list] = _net_connections,
        online_every: float = 15.0,
        ip_every: float = 600.0,
        conns_every: float = 15.0,
        arp_every: float = 60.0,
        offline_after: int = 2,
        arp_watch: bool = False,
        dns_host: str = "www.apple.com",
        spike_ratio: float = 8.0,
        spike_floor_bps: float = 1_000_000.0,
        outbound_max: int = 512,
        outbound_ttl: float = 3600.0,
        events_max: int = 8,
    ) -> None:
        self.probe_urls = probe_urls or [
            "http://captive.apple.com",  # 204 endpoint; non-204 = captive
            "https://www.baidu.com",
            "https://api.ipify.org",
        ]

        self.ip_urls = ip_urls or [
            "https://api.ipify.org",
            "https://ip.3322.net",
            "https://ifconfig.me/ip",
        ]

        self.probe_timeout = probe_timeout

        self.now = now

        self.mono = mono

        self.gw_fn = gw_fn

        self.probe_fn = probe_fn

        self.ip_fn = ip_fn

        self.ssid_fn = ssid_fn

        self.dns_fn = dns_fn

        self.gw_ping_fn = gw_ping_fn

        self.arp_fn = arp_fn

        self.io_fn = io_fn

        self.conns_fn = conns_fn

        self.online_every = online_every

        self.ip_every = ip_every

        self.conns_every = conns_every

        self.arp_every = arp_every

        self.offline_after = offline_after

        self.arp_watch = arp_watch

        self.dns_host = dns_host

        self.spike_ratio = spike_ratio

        self.spike_floor_bps = spike_floor_bps

        self.outbound_max = outbound_max

        self.outbound_ttl = outbound_ttl

        self.events: deque[tuple[float, str]] = deque(maxlen=events_max)

        self._t_online = -1e9

        self._t_ip = -1e9

        self._t_conns = -1e9

        self._fail_count = 0

        self._offline_since: float | None = None

        self._latency: float | None = None

        self._probe_code: int | None = None

        self._dns_ok: bool | None = None

        self._lan_key: tuple | None = None

        self._seen_arp: set[str] | None = None

        self._t_arp = -1e9

        self._public: str | None = None

        self._prev_io: dict[str, tuple[int, int]] | None = None

        self._rates: dict[str, tuple[float, float]] = {}

        self._seen_outbound: dict[tuple, float] = {}

        self._seen_listen: set[tuple] = set()

        self._proc_names: dict[int, str] = {}

        self._baseline: tuple[float, float] | None = None

        self._conn_stats: tuple[int | None, int | None, int | None] = (None, None, None)

        self._last_sample_mono: float | None = None

    # ------------------------------------------------------------------

    def _emit(self, text: str) -> None:
        self.events.append((self.now(), text))

    def _proc_name(self, pid: int | None, name: str | None = None) -> str:
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

    def _gw_note(self) -> str:
        """
        Offline detail: can we still reach the gateway? Distinguishes
        a local break from an upstream outage.
        """
        gw = self.gw_fn()

        if not gw:
            return ""

        return (
            " (gw ok, upstream down)" if self.gw_ping_fn(gw[0]) else " (gw unreachable)"
        )

    def _check_online(self) -> None:
        if self.mono() - self._t_online < self.online_every:
            return

        self._t_online = self.mono()

        result = self.probe_fn(self.probe_urls, self.probe_timeout)

        if result is None:
            self._fail_count += 1

            if self._fail_count >= self.offline_after and self._offline_since is None:
                self._offline_since = self.now()

                self._emit(f"offline{self._gw_note()}")

            self._latency = None

            return

        latency, probe_code = result

        self._latency = latency

        if self._probe_code is not None and probe_code != self._probe_code:
            self._emit(
                f"probe anomaly (http {probe_code})"
                if probe_code
                else "probe clean (204)"
            )

        self._probe_code = probe_code

        if self._offline_since is not None:
            span = self.now() - self._offline_since

            self._offline_since = None

            self._fail_count = 0

            self._emit(f"offline {_fmt_span(span)} -> online")

        elif self._fail_count:
            self._fail_count = 0

        self._check_dns()

    def _check_dns(self) -> None:
        ok = self.dns_fn(self.dns_host)

        if ok is not None and self._dns_ok is not None and ok != self._dns_ok:
            self._emit("dns ok" if ok else "dns fail")

        if ok is not None:
            self._dns_ok = ok

    def _check_arp(self) -> None:
        if not self.arp_watch or self.mono() - self._t_arp < self.arp_every:
            return

        self._t_arp = self.mono()

        devices = self.arp_fn()

        if devices is None:
            return

        if self._seen_arp:
            fresh = sorted(set(devices) - self._seen_arp)[:2]

            for ip in fresh:
                self._emit(f"new lan device: {ip} ({devices[ip]})")

        self._seen_arp = set(devices)

    def _check_public_ip(self) -> None:
        if self.mono() - self._t_ip < self.ip_every:
            return

        if self._offline_since is not None:
            return

        self._t_ip = self.mono()

        ip = self.ip_fn(self.ip_urls, self.probe_timeout)

        if ip and ip != self._public:
            if self._public is not None:
                self._emit(f"public ip -> {ip}")

            self._public = ip

    def _check_traffic(self, dt: float) -> None:
        try:
            counters = self.io_fn()

        except Exception:
            return

        if not counters or dt <= 0:
            # Keep the stored shape identical to the normal path
            # (name -> (recv, sent)); raw counters would leak
            # non-subscriptable values into the next diff.
            self._prev_io = {
                name: (io.bytes_recv, io.bytes_sent) for name, io in counters.items()
            }

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

        self._prev_io = {name: (io.bytes_recv, io.bytes_sent) for name, io in counters.items()}

        self._rates = rates

        # Spike detection on the summed rate (deterministic).
        total_down = sum(r for r, _ in rates.values())

        if self._baseline is None:
            if total_down > 0:
                self._baseline = (total_down, sum(s for _, s in rates.values()))

            return

        total_up = sum(s for _, s in rates.values())

        if total_down > self._baseline[0] * self.spike_ratio and total_down > self.spike_floor_bps:
            self._emit(f"traffic spike ↓{_fmt_bps(total_down)}")

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

                listen_keys.add((lport, pid))

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

                if len(self._seen_outbound) > self.outbound_max:
                    cutoff = now_mono - self.outbound_ttl

                    self._seen_outbound = {
                        k: v for k, v in self._seen_outbound.items() if v > cutoff
                    }

                self._emit(f"new outbound: {key[0]} -> {key[1]}:{key[2]}")

        new_listen = listen_keys - self._seen_listen

        if self._seen_listen and new_listen:
            for lport, pid in sorted(new_listen)[:2]:
                self._emit(f"listen added: {lport} ({self._proc_name(pid)})")

        self._seen_listen = listen_keys

        self._conn_stats = (est, public, len(listen_keys))

    # ------------------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        """Advance one cadence step; returns the full fact sheet."""
        now_wall = self.now()

        mono = self.mono()

        dt = mono - self._last_sample_mono if self._last_sample_mono is not None else 0.0

        self._last_sample_mono = mono

        self._check_online()

        self._check_public_ip()

        self._check_traffic(dt)

        self._check_connections()

        self._check_arp()

        ssid = self.ssid_fn()

        gw = self.gw_fn()

        primary_if, local_ip = self._primary_interface()

        lan_key = (primary_if, local_ip, ssid)

        if (
            self._lan_key is not None
            and lan_key != self._lan_key
            and primary_if
            and local_ip
        ):
            label = ssid or primary_if

            self._emit(f"network switched: {label} ({local_ip})")

        if primary_if and local_ip:
            self._lan_key = lan_key

        tunnel = None

        if (
            gw
            and gw[1]
            and re.search(r"(tun|tap|ppp|wg|ipsec|tailscale)", gw[1], re.I)
        ):
            # Default route riding a tunnel interface IS the
            # VPN fact; macOS system utuns never carry it.
            tunnel = gw[1]

        est, public, listen = self._conn_stats

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
            "probe_code": self._probe_code,
            "dns_ok": self._dns_ok,
            "offline_since": self._offline_since,
            "public_ip": self._public,
            "dl": total_down if self._rates else None,
            "ul": total_up if self._rates else None,
            "est": est,
            "public_conns": public,
            "listen": listen,
            "events": list(self.events),
        }

    def _primary_interface(self) -> tuple[str | None, str | None]:
        """(interface, ipv4) of the default-looking up interface."""
        try:
            stats = psutil.net_if_stats()

            addrs = psutil.net_if_addrs()

            candidates = []

            for name, addr_list in addrs.items():
                if name.startswith(("lo", "utun", "tun", "tap")):
                    continue

                if not stats.get(name) or not stats[name].isup:
                    continue

                for addr in addr_list:
                    if addr.family.name == "AF_INET" and not addr.address.startswith("169.254"):
                        candidates.append((name, addr.address))

                        break

            if candidates:
                candidates.sort(
                    key=lambda pair: (
                        not pair[0].startswith(("en", "eth", "wl")),
                        pair[0],
                    )
                )

                return candidates[0]

        except Exception:
            pass

        return None, None


# ============================================================================
# Rendering: the text that rides the prompt
# ============================================================================


def _fmt_span(seconds: float | None) -> str:
    if seconds is None:
        return ""

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


def render_facts(facts: dict[str, Any], *, events_limit: int) -> str:
    """The compact fact block injected into the prompt.

    No header: the core labels the block by service_id when it
    packages injections into the observation.
    """
    lines = []

    lines.append(_link_line(facts))

    public = facts.get("public_ip")

    tunnel = facts.get("tunnel")

    if public or tunnel:
        parts = []

        if public:
            parts.append(f"public: {public}")

        if tunnel:
            parts.append(f"tunnel: {tunnel}")

        lines.append("- " + " | ".join(parts))

    dl = facts.get("dl")

    ul = facts.get("ul")

    if dl is None and ul is None:
        lines.append("- traffic: sampling")

    else:
        lines.append(f"- traffic: ↓{_fmt_bps(dl or 0)} ↑{_fmt_bps(ul or 0)}")

    est = facts.get("est")

    if est is None:
        lines.append("- conn: sampling")

    else:
        lines.append(
            f"- conn: {est} est ({facts.get('public_conns') or 0} public)"
            f" | listen {facts.get('listen') or 0}",
        )

    events = facts.get("events") or []

    if events:
        rendered = " | ".join(text for _, text in events[-events_limit:])

        lines.append(f"- events: {rendered}")

    return "\n".join(lines)


def _link_line(facts: dict[str, Any]) -> str:
    if not facts.get("online"):
        since = _fmt_span(
            (lambda ts: time.time() - ts if ts else None)(facts.get("offline_since")),
        )

        return f"- link: offline{(' ' + since) if since else ''}"

    parts = []

    if facts.get("ssid"):
        parts.append(facts["ssid"])

    if facts.get("local_ip"):
        parts.append(facts["local_ip"])

    head = facts.get("primary_if") or "link"

    line = f"- link: {head} " + " ".join(parts)

    if facts.get("gw"):
        line += f" -> gw {facts['gw']}"

    latency = facts.get("latency")

    if latency is not None:
        line += f" | online {latency:.0f}ms"

    else:
        line += " | online"

    probe_code = facts.get("probe_code")

    if probe_code:
        line += f" | probe: http {probe_code} (expected 204)"

    return line


# ============================================================================
# Service shell: config, register, heartbeat, turn endpoint
# ============================================================================


class Config:
    """Typed view of config.yaml; unknown keys are ignored."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.hub_url: str = str(raw.get("hub_url", "http://127.0.0.1:8901")).rstrip("/")

        self.service_id: str = str(raw.get("service_id", "network"))

        self.host: str = str(raw.get("host", "127.0.0.1"))

        self.port: int = int(raw.get("port", 9002))

        self.probe_urls: list[str] = list(
            raw.get(
                "probe_urls",
                ["http://captive.apple.com", "https://www.baidu.com"],
            ),
        )

        self.dns_host: str = str(raw.get("dns_host", "www.apple.com"))

        self.arp_watch: bool = bool(raw.get("arp_watch", False))

        self.ip_urls: list[str] = list(
            raw.get("ip_urls", ["https://api.ipify.org", "https://ip.3322.net"]),
        )

        self.probe_timeout: float = float(raw.get("probe_timeout", 3.0))

        self.online_every: float = float(raw.get("online_every", 15.0))

        self.ip_every: float = float(raw.get("ip_every", 600.0))

        self.conns_every: float = float(raw.get("conns_every", 15.0))

        self.arp_every: float = float(raw.get("arp_every", 60.0))

        self.offline_after: int = int(raw.get("offline_after", 2))

        self.spike_ratio: float = float(raw.get("spike_ratio", 8.0))

        self.spike_floor_bps: float = float(raw.get("spike_floor_bps", 1_000_000.0))

        self.outbound_max: int = int(raw.get("outbound_max", 512))

        self.outbound_ttl: float = float(raw.get("outbound_ttl", 3600.0))

        self.events_max: int = int(raw.get("events_max", 8))

        self.events_render_limit: int = int(raw.get("events_render_limit", 3))


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return Config(raw)


class NetworkService:
    """
    Owns the probe, the latest sample and its rendered text, and
    the hub lifecycle (register / heartbeat / unregister).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        self.probe = NetworkProbe(
            probe_urls=config.probe_urls,
            ip_urls=config.ip_urls,
            probe_timeout=config.probe_timeout,
            online_every=config.online_every,
            ip_every=config.ip_every,
            conns_every=config.conns_every,
            arp_every=config.arp_every,
            offline_after=config.offline_after,
            arp_watch=config.arp_watch,
            dns_host=config.dns_host,
            spike_ratio=config.spike_ratio,
            spike_floor_bps=config.spike_floor_bps,
            outbound_max=config.outbound_max,
            outbound_ttl=config.outbound_ttl,
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
                logger.exception("network sample failed")

            await asyncio.sleep(5.0)

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


def build_app(service: NetworkService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.register()

        sampler = asyncio.create_task(service._sample_loop(), name="network-sample")

        beater = asyncio.create_task(service._heartbeat_loop(), name="network-heartbeat")

        logger.info(
            "network service up on {}:{} (hub {})",
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
        HERE / "network.log",
        rotation="5 MB",
        retention=2,
        level="INFO",
    )

    service = NetworkService(config)

    uvicorn.run(
        build_app(service),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
