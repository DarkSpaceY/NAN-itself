"""
Delta/state tests: one full snapshot per epoch, changed fields after.

The probe's samplers are not started here; facts are placed on the
service directly so each turn is deterministic.
"""

from __future__ import annotations

import main as network_mod
from nan_service_kit import HubLink
from nan_service_kit.testing import StubHub


def make_service() -> network_mod.NetworkService:
    config = network_mod.load_config()

    stub = StubHub()

    return network_mod.NetworkService(
        config,
        link=HubLink(config, transport=stub.transport()),
    )


def facts(**overrides) -> dict:
    base = {
        "now": 1000.0,
        "primary_if": "en0",
        "local_ip": "192.168.1.42",
        "ssid": "home",
        "gw": "192.168.1.1",
        "gw_iface": "en0",
        "tunnel": None,
        "online": True,
        "latency": 18.0,
        "probe_code": None,
        "dns_ok": True,
        "offline_since": None,
        "offline_detail": None,
        "public_ip": "1.2.3.4",
        "dl": 0.0,
        "ul": 0.0,
        "est": 1,
        "public_conns": 0,
        "listen": 1,
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

    assert "- link: en0 home 192.168.1.42 -> gw 192.168.1.1" in first

    assert "- reach: online 18ms" in first

    assert "- note: snapshot for this epoch" in first

    assert service.contribution(turn(started=2.0, origin="normal")) == ""


def test_latency_band_change_is_one_line():
    service = make_service()

    service._facts = facts(latency=18.0)

    service.contribution(turn())

    service._facts = facts(latency=25.0)

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- reach: online 25ms"
    )


def test_offline_recovery_uses_the_reach_field():
    service = make_service()

    service._facts = facts()

    service.contribution(turn())

    service._facts = facts(
        online=False,
        latency=None,
        offline_since=1000.0,
        offline_detail="gw ok, upstream down",
    )

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- reach: offline 0s (gw ok, upstream down)"
    )

    service._facts = facts(
        online=False,
        latency=None,
        offline_since=1000.0,
        offline_detail="gw ok, upstream down",
        now=1120.0,
    )

    assert service.contribution(turn(started=3.0, origin="normal")) == (
        "- reach: offline 2m (gw ok, upstream down)"
    )

    service._facts = facts(now=1130.0)

    assert service.contribution(turn(started=4.0, origin="normal")) == (
        "- reach: online 18ms"
    )


def test_link_identity_change():
    service = make_service()

    service._facts = facts()

    service.contribution(turn())

    service._facts = facts(ssid="office", local_ip="192.168.1.43")

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- link: en0 office 192.168.1.43 -> gw 192.168.1.1"
    )


def test_probe_dns_public_and_tunnel_fields():
    service = make_service()

    service._facts = facts()

    service.contribution(turn())

    service._facts = facts(probe_code=302)

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- probe: http 302 (captive check failed)"
    )

    service._facts = facts(probe_code=302, dns_ok=False)

    assert service.contribution(turn(started=3.0, origin="normal")) == "- dns: fail"

    service._facts = facts(probe_code=None, dns_ok=True, public_ip="5.6.7.8")

    assert service.contribution(turn(started=4.0, origin="normal")) == (
        "- probe: clean\n- dns: ok\n- public: 5.6.7.8"
    )

    service._facts = facts(probe_code=None, dns_ok=True, public_ip=None, tunnel="utun3")

    assert service.contribution(turn(started=5.0, origin="normal")) == (
        "- public: none\n- tunnel: utun3"
    )


def test_new_notes_are_sent_once():
    service = make_service()

    service._facts = facts()

    service.contribution(turn())

    service._facts = facts(notes=[(1.0, "new outbound: app -> 8.8.8.8:443")])

    assert service.contribution(turn(started=2.0, origin="normal")) == (
        "- events: new outbound: app -> 8.8.8.8:443"
    )

    assert service.contribution(turn(started=3.0, origin="normal")) == ""


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
