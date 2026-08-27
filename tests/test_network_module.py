"""
Network module contracts: deterministic facts only.

httpx, psutil, platform shims and the clock are all stubbed.
"""

from types import SimpleNamespace

import pytest

from src.nan_itself.modules.builtin.network import (
    NetworkModule,
)
from src.nan_itself.utils.network import (
    NetworkProbe,
    is_public_ip,
)


import psutil


class Clock:
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


class Holder:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        return self.value


def conn(status, rip=None, rport=None, lport=None, pid=10):
    laddr = SimpleNamespace(ip="127.0.0.1", port=lport or 0)

    raddr = (
        SimpleNamespace(ip=rip, port=rport)
        if rip
        else None
    )

    return SimpleNamespace(
        status=status,
        laddr=laddr,
        raddr=raddr,
        pid=pid,
    )


def make_probe(
    clock=None,
    latency=35.0,
    ip="203.0.113.7",
    counters=None,
    conns=None,
    conns_fn=None,
    gw=("192.168.1.1", "en0"),
    ssid='"HOME"',
):
    clock = clock or Clock()

    counters = counters or {}

    seq = {"n": 0}

    def probe_fn():
        seq["n"] += 1

        value = latency

        if isinstance(value, list):
            if not value:
                return None

            return value.pop(0)

        return value

    probe = NetworkProbe(
        now=clock.now,
        mono=clock.mono,
        gw_fn=Holder(gw) if not callable(gw) else gw,
        probe_fn=probe_fn,
        ip_fn=ip if callable(ip) else Holder(ip),
        ssid_fn=ssid if callable(ssid) else Holder(ssid),
        io_fn=lambda: counters,
        conns_fn=conns_fn or (
            conns if callable(conns) else (lambda: conns or [])
        ),
    )

    return probe, clock


# ======================================================================
# pure helpers
# ======================================================================


def test_is_public_ip_matrix():
    assert is_public_ip("8.8.8.8")

    assert not is_public_ip("192.168.1.1")

    assert not is_public_ip("10.0.0.5")

    assert not is_public_ip("127.0.0.1")

    assert not is_public_ip("169.254.3.4")

    assert not is_public_ip("224.0.0.1")

    assert not is_public_ip("::1")

    assert not is_public_ip("garbage")


# ======================================================================
# probe: rates / events
# ======================================================================


def test_traffic_rate_from_counter_diff():
    clock = Clock()

    counters = {"en0": SimpleNamespace(bytes_recv=1000, bytes_sent=200)}

    probe, _ = make_probe(clock=clock, counters=counters)

    probe.sample()  # baseline

    clock.advance(wall=5.0, mono=5.0)

    counters["en0"].bytes_recv = 1000 + 10000

    counters["en0"].bytes_sent = 200 + 2000

    facts = probe.sample()

    assert facts["dl"] == pytest.approx(2000.0)

    assert facts["ul"] == pytest.approx(400.0)


def test_offline_then_recovery_events():
    # first two probes fail -> offline; then success -> recovered
    latency = [None, None, 40.0]

    probe, clock = make_probe(latency=latency)

    probe.sample()  # fail 1

    clock.advance(wall=15.0, mono=15.0)

    probe.sample()  # fail 2 -> offline event

    assert any("offline" in t for _, t in probe.events)

    clock.advance(wall=15.0, mono=15.0)

    probe.sample()  # success -> recovered

    texts = [t for _, t in probe.events]

    assert any("-> online" in t for t in texts)

    offline_event = [t for t in texts if t == "offline"]

    recovered = [t for t in texts if "-> online" in t]

    assert len(offline_event) == 1

    assert "offline" in recovered[0]


def test_public_ip_change_event_not_fired_on_first():
    holder = Holder("203.0.113.7")

    probe, clock = make_probe(ip=holder)

    probe.sample()

    assert not any("public ip" in t for _, t in probe.events)

    holder.value = "198.51.100.9"

    clock.advance(wall=700.0, mono=700.0)

    probe.sample()

    assert any("public ip -> 198.51.100.9" in t
               for _, t in probe.events)


def test_new_outbound_event_and_dedup():
    conns = [
        conn("ESTABLISHED", "140.82.112.3", 443, pid=10),
    ]

    holder = Holder([(c, 10, "Chrome") for c in conns])

    probe, clock = make_probe(conns_fn=holder)

    probe.sample()

    assert any(
        "new outbound: Chrome -> 140.82.112.3:443"
        in t
        for _, t in probe.events
    )

    clock.advance(wall=20.0, mono=20.0)

    probe.sample()

    assert sum("new outbound" in t
               for _, t in probe.events) == 1


def test_new_outbound_ignores_private_peers():
    conns = [
        conn("ESTABLISHED", "192.168.1.1", 445, pid=10),
    ]

    holder = Holder([(c, 10, "Explorer") for c in conns])

    probe, _ = make_probe(conns_fn=holder)

    probe.sample()

    assert not any("new outbound" in t for _, t in probe.events)


def test_listen_added_event():
    conns1 = [conn("LISTEN", lport=8080, pid=10)]

    holder = Holder([(c, 10, "Helper") for c in conns1])

    probe, clock = make_probe(conns_fn=holder)

    probe.sample()  # baseline

    holder.value = [
        (c, 10, "Helper") for c in conns1
    ] + [
        (conn("LISTEN", lport=9090, pid=11), 11, "Helper"),
    ]

    clock.advance(wall=20.0, mono=20.0)

    probe.sample()

    assert any("listen added: 9090 (Helper)" in t
               for _, t in probe.events)


def test_tunnel_detected_via_gateway_interface():
    probe, _ = make_probe(gw=("10.8.0.1", "utun5"))

    facts = probe.sample()

    assert facts["tunnel"] == "utun5"


def test_connection_cadence():
    calls = []

    def conns_fn():
        calls.append(1)

        return []

    probe, clock = make_probe(conns_fn=conns_fn)

    probe.sample()

    clock.advance(wall=2.0, mono=2.0)

    probe.sample()  # inside 15s cadence -> skipped

    assert len(calls) == 1

    clock.advance(wall=20.0, mono=20.0)

    probe.sample()

    assert len(calls) == 2


# ======================================================================
# module rendering
# ======================================================================


def base_facts(**overrides):
    facts = {
        "now": 0.0,
        "primary_if": "en0",
        "local_ip": "192.168.1.23",
        "ssid": '"HOME"',
        "gw": "192.168.1.1",
        "gw_iface": "en0",
        "tunnel": None,
        "online": True,
        "latency": 34.0,
        "offline_since": None,
        "public_ip": "203.0.113.7",
        "dl": 2.3 * 1024 * 1024,
        "ul": 210.0 * 1024,
        "est": 68,
        "public_conns": 12,
        "listen": 9,
        "events": [(0.0, "new outbound Chrome -> 140.82.x.x:443")],
    }

    facts.update(overrides)

    return facts


def run_query(module):
    import asyncio

    return asyncio.run(module.query(turn=None))


def make_module(facts):
    module = NetworkModule()

    module._latest = facts

    return module


def test_query_renders_single_territory():
    module = make_module(base_facts())

    rendered = run_query(module)

    assert rendered.startswith("[Network]\n")

    assert rendered.count("[Network]") == 1

    assert '- link: en0 "HOME" 192.168.1.23 -> gw 192.168.1.1' in rendered

    assert "online 34ms" in rendered

    assert "- public: 203.0.113.7" in rendered

    assert "- traffic: ↓2.3MB/s ↑210KB/s" in rendered

    assert "- conn: 68 est (12 public) | listen 9" in rendered

    assert "- events: new outbound Chrome -> 140.82.x.x:443" in rendered


def test_query_lines_are_all_facts():
    rendered = run_query(make_module(base_facts()))

    for line in rendered.splitlines():
        assert line == "[Network]" or line.startswith("- "), line

    for banned in ("不可信", "安全", "可疑"):
        assert banned not in rendered


def test_query_offline_facts():
    module = make_module(
        base_facts(
            online=False,
            latency=None,
            offline_since=1000.0,
        )
    )

    rendered = run_query(module)

    assert "- link: offline" in rendered

    assert "34ms" not in rendered


def test_query_none_before_first_sample():
    assert run_query(NetworkModule()) is None
