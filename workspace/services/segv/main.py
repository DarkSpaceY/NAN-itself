"""
SEGV memory service: the core plus NAN's protocol adapter.

One process, two hub registrations: `segv-archive` receives every
committed turn at `POST /turn` and writes it durably to L0; `segv`
answers `POST /inject` with the small block an epoch opens with.

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

from pathlib import Path

import uvicorn
from nan_service_kit import HubLink, setup_logging
from segv.app import SegvService, build_app
from segv_nan.app import install
from segv_nan.config import load_config

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    setup_logging(config)

    archive = HubLink(
        config.model_copy(
            update={
                "service_id": config.archive_id,
                "mode": "archive",
                "path": "turn",
            }
        )
    )
    inject = HubLink(
        config.model_copy(
            update={
                "service_id": config.inject_id,
                "mode": "inject",
                "path": "inject",
                "surface": None,
            }
        )
    )

    service = SegvService(config.core(), links=[archive, inject])

    uvicorn.run(
        build_app(service, routes=lambda app: install(app, service)),
        host=config.host,
        port=config.port,
        log_level="warning",
        timeout_graceful_shutdown=1,
    )


if __name__ == "__main__":
    main()
