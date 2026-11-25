from __future__ import annotations

import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(SERVICE))

import pytest  # noqa: E402
from segv.config import CoreConfig  # noqa: E402
from segv.fact.config import FactConfig  # noqa: E402


@pytest.fixture
def config(tmp_path: Path) -> CoreConfig:
    config = CoreConfig(
        port=9004,
        data_dir="data",
        max_bytes=0,
        fact=FactConfig(enabled=False),
    )

    config._home = tmp_path

    return config
