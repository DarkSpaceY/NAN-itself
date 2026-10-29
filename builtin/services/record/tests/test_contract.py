"""
Contract test: the archive registers in archive mode, acknowledges a
completed turn, keeps its panel up while it runs and drops it on exit.
"""

from __future__ import annotations

import main as record
from fastapi.testclient import TestClient
from nan_service_kit import HubLink
from nan_service_kit.testing import StubHub, assert_registered, assert_turn_ack


def test_archive_contract(tmp_path):
    config = record.load_config()

    config.batch_interval = 0.01

    stub = StubHub()

    service = record.RecordService(
        config,
        store_dir=tmp_path / "turns",
        link=HubLink(config, transport=stub.transport()),
    )

    with TestClient(record.build_app(service)) as client:
        assert_turn_ack(client)

        assert_registered(stub, config)

    assert [body["url"] for body in stub.bodies("/surfaces")] == [
        f"http://{config.host}:{config.port}/"
    ]

    assert stub.deletes == [f"/surfaces/{config.service_id}"]
