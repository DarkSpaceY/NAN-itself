"""
Contract test: an inject service registers its endpoint, answers a
turn with prompt text alone, and unregisters on exit.
"""

from __future__ import annotations

import main as network_mod
from fastapi.testclient import TestClient
from nan_service_kit import HubLink
from nan_service_kit.testing import StubHub, assert_registered, assert_turn_text


def test_inject_contract():
    config = network_mod.load_config()

    stub = StubHub()

    service = network_mod.NetworkService(
        config, link=HubLink(config, transport=stub.transport())
    )

    service._rendered = "hello"

    with TestClient(network_mod.build_app(service)) as client:
        assert_turn_text(client, "hello")

        assert_registered(stub, config)

    assert stub.bodies("/unregister") == [{"service_id": config.service_id}]
