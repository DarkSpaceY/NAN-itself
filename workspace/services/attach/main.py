"""
Attach service: a stateless reader the agent calls with a file and a request.

Not an injection service -- nothing is pushed into a turn, and it never
registers with the hub. The agent passes a path and a question; the
answer comes back as one string, so a file costs the conversation
nothing beyond that answer.

Text and images are separate capabilities, each with its own endpoint
(config.yaml), so reading does not compete with the agent's own turns.
A file is routed by what its bytes are, and a request for an
unconfigured capability is refused instead of falling back to the other
endpoint.

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from attach import Attacher, AttachError, CapabilityUnavailable, TooLarge
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from nan_service_kit import ServiceConfig, health_route, read_config, setup_logging
from pydantic import BaseModel, ConfigDict, Field


class EndpointConfig(BaseModel):
    """One OpenAI-compatible chat-completions endpoint."""

    model_config = ConfigDict(extra="forbid")

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    timeout: float = 900.0

    @property
    def enabled(self) -> bool:
        return bool(self.model.strip() and self.base_url.strip())


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: EndpointConfig = Field(default_factory=EndpointConfig)
    image: EndpointConfig = Field(default_factory=EndpointConfig)


class AttachConfig(ServiceConfig):
    # Pull-only: there is no hub registration, so the shared default
    # (which reads NAN_INJECTION_PORT) would be a lie.
    hub_url: str = ""

    max_chars: int = 128_000
    max_image_bytes: int = 8_000_000
    workspace: str = ""
    capabilities: Capabilities = Field(default_factory=Capabilities)


class AttachService:
    def __init__(self, config: AttachConfig) -> None:
        self.config = config
        self.attacher = Attacher(config, self._workspace())

    def _workspace(self) -> Path:
        override = self.config.workspace or os.environ.get("NAN_WORKSPACE")

        if override:
            return Path(override)

        # <workspace>/services/attach/config.yaml
        return self.config.home.parent.parent

    def close(self) -> None:
        self.attacher.close()


def build_app(service: AttachService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        capabilities = service.config.capabilities

        logger.info(
            "attach service up on {}:{} (workspace {}, text {}, image {})",
            service.config.host,
            service.config.port,
            service.attacher.workspace,
            capabilities.text.model or "off",
            capabilities.image.model or "off",
        )

        yield

        service.close()

    app = FastAPI(lifespan=lifespan)

    health_route(app, service.config)

    @app.post("/attach")
    async def attach(body: dict[str, Any]) -> JSONResponse:
        path = str(body.get("path") or "").strip()

        ask = str(body.get("ask") or "").strip()

        if not path or not ask:
            return JSONResponse(
                {
                    "status": "bad_request",
                    "code": "bad_request",
                    "message": "path and ask are required",
                },
                status_code=400,
            )

        # One long read must not freeze the event loop for every caller.
        try:
            result = await asyncio.to_thread(service.attacher.attach, path, ask)

        except AttachError as exc:
            payload: dict[str, Any] = {
                "status": exc.code,
                "code": exc.code,
                "message": str(exc),
            }

            if isinstance(exc, TooLarge):
                payload.update({"size": exc.size, "limit": exc.limit, "unit": exc.unit})

            if isinstance(exc, CapabilityUnavailable):
                payload.update({"capability": exc.capability})

            return JSONResponse(payload, status_code=exc.status)

        return JSONResponse(result)

    return app


CONFIG = Path(__file__).resolve().parent / "config.yaml"

EXAMPLE = CONFIG.parent / f"{CONFIG.name}.example"


def load_config(path: Path = CONFIG) -> AttachConfig:
    """First run: the committed example becomes the live, ignored copy."""
    if not path.exists():
        example = path.parent / f"{path.name}.example"

        if not example.exists():
            raise FileNotFoundError(f"{path} and {example} are both missing")

        shutil.copyfile(example, path)

        logger.warning("attach: created {} from {}; edit it", path, example.name)

    return read_config(AttachConfig, path)


def main() -> None:
    config = load_config()

    setup_logging(config)

    uvicorn.run(
        build_app(AttachService(config)),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
