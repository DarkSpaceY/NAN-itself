from __future__ import annotations

import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(SERVICE))

import pytest  # noqa: E402
from segv.config import Config  # noqa: E402
from segv.fact.config import FactConfig  # noqa: E402


@pytest.fixture
def config(tmp_path: Path) -> Config:
    config = Config(
        service_id="segv",
        port=9004,
        hub_url="http://127.0.0.1:1",
        data_dir="data",
        max_bytes=0,
        fact=FactConfig(enabled=False),
    )

    config._home = tmp_path

    return config
