"""
Read service: a stateless reader the agent calls with a file and a request.

Not an injection service -- nothing is pushed into a turn, and it never
registers with the hub. The agent passes a path and a question; the
answer comes back as one string, so a long text costs the conversation
nothing beyond that answer.

The service keeps its own LLM endpoint (config.yaml), separate from the
core's, so reading does not compete with the agent's own turns.

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
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from nan_service_kit import ServiceConfig, read_config, setup_logging
from pydantic import BaseModel, ConfigDict
from reader import ReadError, Reader, TooLarge


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.0
    timeout: float = 900.0


class ReadConfig(ServiceConfig):
    # Pull-only: there is no hub registration, so the shared default
    # (which reads NAN_INJECTION_PORT) would be a lie.
    hub_url: str = ""

    max_chars: int = 128_000
    workspace: str = ""
    llm: LlmConfig


class ReadService:
    def __init__(self, config: ReadConfig) -> None:
        self.config = config
        self.reader = Reader(config, self._workspace())

    def _workspace(self) -> Path:
        override = self.config.workspace or os.environ.get("NAN_WORKSPACE")

        if override:
            return Path(override)

        # <workspace>/services/read/config.yaml
        return self.config.home.parent.parent

    def close(self) -> None:
        self.reader.close()


def build_app(service: ReadService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info(
            "read service up on {}:{} (workspace {}, limit {} chars)",
            service.config.host,
            service.config.port,
            service.reader.workspace,
            service.config.max_chars,
        )

        yield

        service.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/read")
    async def read(body: dict[str, Any]) -> JSONResponse:
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
            result = await asyncio.to_thread(service.reader.read, path, ask)

        except ReadError as exc:
            payload: dict[str, Any] = {
                "status": exc.code,
                "code": exc.code,
                "message": str(exc),
            }

            if isinstance(exc, TooLarge):
                payload.update({"chars": exc.chars, "limit": exc.limit})

            return JSONResponse(payload, status_code=exc.status)

        return JSONResponse(result)

    return app


CONFIG = Path(__file__).resolve().parent / "config.yaml"

EXAMPLE = CONFIG.parent / f"{CONFIG.name}.example"


def load_config(path: Path = CONFIG) -> ReadConfig:
    """First run: the committed example becomes the live, ignored copy."""
    if not path.exists():
        example = path.parent / f"{path.name}.example"

        if not example.exists():
            raise FileNotFoundError(f"{path} and {example} are both missing")

        shutil.copyfile(example, path)

        logger.warning("read: created {} from {}; edit it", path, example.name)

    return read_config(ReadConfig, path)


def main() -> None:
    config = load_config()

    setup_logging(config)

    uvicorn.run(
        build_app(ReadService(config)),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
