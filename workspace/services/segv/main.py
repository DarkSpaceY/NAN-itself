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
from fastapi import FastAPI
from nan_service_kit import HubLink, health_route, setup_logging
from segv.app import SegvService, build_app
from segv_nan.app import install
from segv_nan.config import load_config

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)

    setup_logging(config)

    archive_config = config.model_copy(
        update={
            "service_id": config.archive_id,
            "mode": "archive",
            "path": "turn",
        }
    )

    # The identity this process answers to. `/health` reports it rather
    # than the shared `service_id`, which only defaults to the directory
    # name and can drift from the id the hub actually registered.
    inject_config = config.model_copy(
        update={
            "service_id": config.inject_id,
            "mode": "inject",
            "path": "inject",
            "surface": None,
        }
    )

    service = SegvService(config.core(), links=[HubLink(archive_config), HubLink(inject_config)])

    def routes(app: FastAPI) -> None:
        """NAN's endpoints, then `/health`: the panel mounts at `/` last."""
        install(app, service)

        health_route(app, inject_config)

    uvicorn.run(
        build_app(service, routes=routes),
        host=config.host,
        port=config.port,
        log_level="warning",
        timeout_graceful_shutdown=1,
    )


if __name__ == "__main__":
    main()
