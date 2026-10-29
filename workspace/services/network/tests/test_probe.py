"""
Probe diff-logic tests: NetworkProbe.

main.py is a plain script (not an importable package), so it is
loaded by path. All probe inputs are injectable callables; no
test touches real hardware or the real network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parent.parent / "main.py"

    spec = importlib.util.spec_from_file_location("nan_network_service_main", path)

    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


network_mod = _load()



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


class FakeIO:
    def __init__(self, recv: int, sent: int) -> None:
        self.bytes_recv = recv

        self.bytes_sent = sent


class FakeAddr:
    def __init__(self, ip: str, port: int) -> None:
        self.ip = ip

        self.port = port


class FakeConn:
    def __init__(self, status: str, laddr=None, raddr=None, pid: int | None = None) -> None:
        self.status = status

        self.laddr = laddr

        self.raddr = raddr

        self.pid = pid


class FakeDiskIO:
    def __init__(self, read_bytes: int, write_bytes: int) -> None:
        self.read_bytes = read_bytes

        self.write_bytes = write_bytes


def event_texts(probe) -> list[str]:
    return [text for _, text in probe.events]


# ============================================================================
# NetworkProbe
# ============================================================================


def make_network_probe(clock: FakeClock, **overrides):
    state: dict = {
        "latency": 20.0,
        "captive": None,
        "dns": True,
        "gw_ping": True,
        "arp": None,
        "ip": "1.2.3.4",
        "io": {},
        "conns": [],
        "ssid": "home",
        "gw": ("192.168.1.1", "en0"),
    }

    probe = network_mod.NetworkProbe(
        now=clock.now,
        mono=lambda: clock.mono,
        probe_fn=lambda urls, timeout: (
            (state["latency"], state["captive"]) if state["latency"] is not None else None
        ),
        ip_fn=lambda urls, timeout: state["ip"],
        ssid_fn=lambda: state["ssid"],
        gw_fn=lambda: state["gw"],
        dns_fn=lambda host: state["dns"],
        gw_ping_fn=lambda ip: state["gw_ping"],
        arp_fn=lambda: state["arp"],
        io_fn=lambda pernic=True: state["io"],
        conns_fn=lambda: state["conns"],
        online_every=5.0,
        ip_every=600.0,
        conns_every=5.0,
        arp_every=5.0,
        offline_after=2,
        **overrides,
    )

    return probe, state


def test_network_offline_then_recovery():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    probe.sample()  # online

    state["latency"] = None

    clock.tick(5.0)

    probe.sample()  # fail 1: not offline yet

    assert probe._offline_since is None

    clock.tick(5.0)

    probe.sample()  # fail 2: offline

    assert any(text.startswith("offline") for text in event_texts(probe))

    state["latency"] = 25.0

    clock.tick(5.0)

    probe.sample()  # recovered

    assert any("-> online" in text for text in event_texts(probe))


def test_network_offline_gw_detail():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    probe.sample()

    state["latency"] = None

    state["gw_ping"] = False

    clock.tick(5.0)

    probe.sample()  # fail 1

    clock.tick(5.0)

    probe.sample()  # offline with gateway unreachable

    assert any("offline (gw unreachable)" in text for text in event_texts(probe))


def test_network_switch_event():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    probe.sample()

    assert not any("network switched" in text for text in event_texts(probe))

    state["ssid"] = "office"

    clock.tick(5.0)

    probe.sample()

    assert any("network switched: office" in text for text in event_texts(probe))


def test_network_probe_anomaly_events():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    state["captive"] = 200

    probe.sample()  # first observation: fact only, no event

    assert not any("probe" in text for text in event_texts(probe))

    assert probe._probe_code == 200

    state["captive"] = None

    clock.tick(5.0)

    probe.sample()

    assert any("probe clean" in text for text in event_texts(probe))


def test_network_dns_fail_event():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    probe.sample()  # dns ok

    state["dns"] = False

    clock.tick(5.0)

    probe.sample()

    assert any("dns fail" in text for text in event_texts(probe))

    state["dns"] = True

    clock.tick(5.0)

    probe.sample()

    assert any("dns ok" in text for text in event_texts(probe))


def test_network_arp_new_device():
    clock = FakeClock()

    probe, state = make_network_probe(clock, arp_watch=True)

    state["arp"] = {"192.168.1.1": "aa:bb:cc:dd:ee:ff"}

    probe.sample()  # baseline

    assert not any("new lan device" in text for text in event_texts(probe))

    state["arp"] = {
        "192.168.1.1": "aa:bb:cc:dd:ee:ff",
        "192.168.1.42": "11:22:33:44:55:66",
    }

    clock.tick(5.0)

    probe.sample()

    assert any("new lan device: 192.168.1.42" in text for text in event_texts(probe))

    clock.tick(5.0)

    probe.sample()  # no further events

    assert sum("new lan device" in text for text in event_texts(probe)) == 1


def test_network_public_ip_change_event():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    probe.sample()  # learns 1.2.3.4

    assert not any("public ip" in text for text in event_texts(probe))

    state["ip"] = "5.6.7.8"

    clock.tick(601.0)

    probe.sample()

    assert any("public ip -> 5.6.7.8" in text for text in event_texts(probe))


def test_network_traffic_spike():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    state["io"] = {"en0": FakeIO(0, 0)}

    probe.sample()  # seeds prev counters

    state["io"] = {"en0": FakeIO(100_000, 0)}  # 20KB/s over dt=5: baseline

    clock.tick(5.0)

    probe.sample()

    state["io"] = {"en0": FakeIO(100_000 + 10_000_000, 0)}  # 2MB/s: 100x baseline

    clock.tick(5.0)

    probe.sample()

    assert any("traffic spike" in text for text in event_texts(probe))


def test_network_new_outbound_connection():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    probe.sample()  # empty

    state["conns"] = [
        (FakeConn("ESTABLISHED", raddr=FakeAddr("8.8.8.8", 443)), 42, "app"),
    ]

    clock.tick(5.0)

    probe.sample()

    assert any("new outbound: " in text for text in event_texts(probe))

    clock.tick(5.0)

    probe.sample()  # same connection: no repeat

    assert sum("new outbound: " in text for text in event_texts(probe)) == 1


def test_network_render_offline():
    clock = FakeClock()

    probe, state = make_network_probe(clock)

    probe.sample()

    state["latency"] = None

    clock.tick(5.0)

    probe.sample()

    clock.tick(5.0)

    probe.sample()

    facts = probe.sample()

    text = network_mod.render_facts(facts, events_limit=3)

    assert "offline" in text

# ============================================================================
# Turn endpoint contract (no hub needed)
# ============================================================================


def test_turn_contract_roundtrip():
    from fastapi.testclient import TestClient

    config = network_mod.Config({"hub_url": "http://127.0.0.1:1", "port": 0})

    service = network_mod.NetworkService(config)

    service._rendered = "hello"

    app = network_mod.build_app(service)

    with TestClient(app) as client:  # lifespan starts/ends inside
        response = client.post("/turn", json={})

        assert response.status_code == 200

        assert response.json() == {"text": "hello"}


