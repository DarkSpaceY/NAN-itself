"""
Contract test: the gateway registers its endpoint, answers a turn with
prompt text alone, and unregisters on exit.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from nan_service_kit import HubLink
from nan_service_kit.testing import FRESH_TURN, StubHub, assert_registered

from main import MCPService, build_app, load_config


def test_inject_contract():
    config = load_config()

    stub = StubHub()

    service = MCPService(config, link=HubLink(config, transport=stub.transport()))

    with TestClient(build_app(service)) as client:
        reply = client.post("/turn", json=FRESH_TURN)

        assert reply.status_code == 200

        assert set(reply.json()) == {"text"}

        assert "mcp.sh" in reply.json()["text"]

        assert_registered(stub, config)

    assert stub.bodies("/unregister") == [{"service_id": config.service_id}]
