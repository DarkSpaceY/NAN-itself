"""
MCP gateway: one process holding every configured MCP server.

Each `servers` entry in config.yaml is spawned as a persistent
stdio subprocess through a fastmcp Client. The service registers
with the injection hub and answers every turn request with a
catalog of the available tools and the shell invocation for them;
the tools themselves are reached over this service's HTTP API by
the agent through the sibling `mcp.sh` CLI.

config.yaml is reloaded on change: servers whose definition changed
are restarted, added ones are spawned, removed ones are stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from loguru import logger
from nan_service_kit import HubLink, ServiceConfig, read_config, setup_logging

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"

CACHE_DIR = "cache"  # package-manager caches; regenerable, safe to delete


class ServerSpec:
    """
    One configured MCP server: identity is (name, definition).
    """

    def __init__(self, raw: dict[str, Any], placeholders: dict[str, str]) -> None:
        self.name: str = str(raw["name"])

        self.command: str = str(raw["command"])

        args = raw.get("args", [])

        self.args: list[str] = [self._expand(a, placeholders) for a in args]

        self.cwd: str | None = (
            self._expand(raw["cwd"], placeholders) if raw.get("cwd") else None
        )

        env = raw.get("env", {})

        self.env: dict[str, str] = {
            str(k): self._expand(str(v), placeholders) for k, v in env.items()
        }

    @staticmethod
    def _expand(value: Any, placeholders: dict[str, str]) -> str:
        text = str(value)

        for key, replacement in placeholders.items():
            text = text.replace("{" + key + "}", replacement)

        return text

    def fingerprint(self) -> str:
        return json.dumps([self.command, self.args, self.cwd, self.env], sort_keys=True)


class Config(ServiceConfig):
    """
    The gateway keys: which servers to hold, and how long a call may run.
    A server entry stays raw here; `MCPService` expands its placeholders
    once the service's own directory is known.
    """

    call_timeout: float = 120.0

    servers: list[dict[str, Any]] = []


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    return read_config(Config, path)


class MCPServer:
    """
    One running MCP server: a persistent fastmcp Client over
    stdio plus its tool snapshot.
    """

    def __init__(self, spec: ServerSpec, config: Config) -> None:
        self.spec = spec

        self.config = config

        self.tools: list[dict[str, Any]] = []

        self._client: Client | None = None

        self._lock = asyncio.Lock()

    def _build_client(self) -> Client:
        # Package-manager caches land in this service's own cache dir:
        # the launch sandbox only grants writes under workspace.
        cache = self.config.home / CACHE_DIR

        cache_env = {
            "npm_config_cache": str(cache / "npm"),
            "UV_CACHE_DIR": str(cache / "uv"),
            "PLAYWRIGHT_BROWSERS_PATH": str(cache / "playwright"),
        }

        env = {**os.environ, **cache_env, **self.spec.env}

        transport = StdioTransport(self.spec.command, self.spec.args, env=env, cwd=self.spec.cwd)

        return Client(transport)

    async def start(self) -> None:
        async with self._lock:
            client = self._build_client()

            await client.__aenter__()

            try:
                tools = await client.list_tools()

            except Exception:
                with contextlib.suppress(Exception):
                    await client.__aexit__(None, None, None)

                raise

            self._client = client

            self.tools = [
                {
                    "name": tool.name,
                    "description": (tool.description or "").strip(),
                    "input_schema": tool.inputSchema,
                }
                for tool in tools
            ]

            logger.info(
                "MCP server {}: up ({} tools)", self.spec.name, len(self.tools)
            )

    async def stop(self) -> None:
        async with self._lock:
            if self._client is None:
                return

            client, self._client = self._client, None

            self.tools = []

            with contextlib.suppress(Exception):
                await client.__aexit__(None, None, None)

            logger.info("MCP server {}: stopped", self.spec.name)

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if self._client is None:
                return {"ok": False, "error": f"server {self.spec.name} is not running"}

            try:
                result = await asyncio.wait_for(
                    self._client.call_tool(tool, arguments), timeout=self.config.call_timeout
                )

            except asyncio.TimeoutError:
                return {
                    "ok": False,
                    "error": f"{self.spec.name}.{tool} timed out after {self.config.call_timeout}s",
                }

            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        texts = [
            block.text for block in result.content if type(block).__name__ == "TextContent"
        ]

        if texts:
            return {"ok": not result.is_error, "result": "\n".join(texts)}

        if getattr(result, "data", None) is not None:
            return {"ok": not result.is_error, "result": json.dumps(result.data, default=str)}

        return {"ok": not result.is_error, "result": ""}


class MCPService:
    """
    Owns the running servers, the config watch, the hub link and the
    prompt contribution.
    """

    def __init__(
        self,
        config: Config,
        config_path: Path = DEFAULT_CONFIG_PATH,
        *,
        link: HubLink | None = None,
    ) -> None:
        self.config = config

        self.config_path = config_path

        self._config_mtime: float = config_path.stat().st_mtime

        self._running: dict[str, MCPServer] = {}

        self._lifecycle = asyncio.Lock()

        self.link = link or HubLink(config)

        self.specs = self._specs_for(config)

    @staticmethod
    def _specs_for(config: Config) -> list[ServerSpec]:
        """
        Expand each raw entry against this service's own paths.
        """
        placeholders = {
            "workspace": os.environ.get("NAN_WORKSPACE", str(config.home.parents[1])),
            "service_cache": str(config.home / CACHE_DIR),
            "service_dir": str(config.home),
        }

        return [ServerSpec(entry, placeholders) for entry in config.servers]

    # -- server lifecycle -------------------------------------------------

    async def start_all(self) -> None:
        for spec in self.specs:
            await self._ensure(spec)

    async def _ensure(self, spec: ServerSpec) -> None:
        async with self._lifecycle:
            current = self._running.get(spec.name)

            if current is not None and current.spec.fingerprint() == spec.fingerprint():
                return

            if current is not None:
                await current.stop()

            server = MCPServer(spec, self.config)

        try:
            await server.start()

        except Exception as exc:
            logger.warning("MCP server {}: failed to start: {}", spec.name, exc)

            return

        async with self._lifecycle:
            self._running[spec.name] = server

    async def reload_if_changed(self) -> None:
        try:
            mtime = self.config_path.stat().st_mtime

        except OSError:
            return

        if mtime == self._config_mtime:
            return

        self._config_mtime = mtime

        logger.info("config.yaml changed: reloading")

        try:
            self.config = load_config(self.config_path)

        except Exception as exc:
            logger.warning("config.yaml reload failed, keeping previous: {}", exc)

            return

        self.specs = self._specs_for(self.config)

        wanted = {spec.name: spec for spec in self.specs}

        async with self._lifecycle:
            for name, server in list(self._running.items()):
                if name not in wanted:
                    await server.stop()

                    del self._running[name]

        for spec in self.specs:
            await self._ensure(spec)

    async def _watch_config(self) -> None:
        while True:
            await asyncio.sleep(5.0)

            await self.reload_if_changed()

    # -- prompt contribution ------------------------------------------------

    def contribution(self, turn: dict[str, Any]) -> str:
        """
        The tool table is not re-sent every turn: the CLI is the
        interface and its --help carries the same information. The
        pointer is injected when the context that held it is gone.
        """
        if turn.get("origin") not in ("first", "context_cleared"):
            return ""

        return (
            f"External MCP tool servers are reachable through `{HERE}/mcp.sh`; "
            "run it with `--help` for list, schema and call."
        )


def build_app(service: MCPService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with service.link:
            # Servers start in the background: first-time npx/uvx
            # downloads take minutes, the API must not wait for them.
            starter = asyncio.create_task(service.start_all(), name="mcp-start")

            watcher = asyncio.create_task(service._watch_config(), name="mcp-config")

            logger.info("mcp gateway up on {}:{}", service.config.host, service.config.port)

            try:
                yield

            finally:
                for task in (watcher, starter):
                    task.cancel()

                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                for server in service._running.values():
                    await server.stop()

    app = FastAPI(lifespan=lifespan)

    # ---------- inject endpoint ----------

    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        return {"text": service.contribution(body)}

    # ---------- tool API ----------

    @app.get("/tools")
    async def tools() -> dict[str, Any]:
        servers = [
            {"name": name, "tools": server.tools} for name, server in sorted(service._running.items())
        ]

        return {"servers": servers}

    @app.get("/schema/{server}/{tool}")
    async def schema(server: str, tool: str) -> dict[str, Any]:
        entry = service._running.get(server)

        if entry is None:
            return {"ok": False, "error": f"unknown server: {server}"}

        for candidate in entry.tools:
            if candidate["name"] == tool:
                return {"ok": True, "server": server, "tool": tool, "schema": candidate["input_schema"]}

        return {"ok": False, "error": f"unknown tool: {server}.{tool}"}

    @app.post("/call")
    async def call(body: dict[str, Any]) -> dict[str, Any]:
        server = str(body.get("server", ""))

        tool = str(body.get("tool", ""))

        arguments = body.get("arguments") or {}

        entry = service._running.get(server)

        if entry is None:
            return {"ok": False, "error": f"unknown server: {server}"}

        if not isinstance(arguments, dict):
            return {"ok": False, "error": "arguments must be a JSON object"}

        return await entry.call(tool, arguments)

    return app


def main() -> None:
    config = load_config()

    setup_logging(config)

    service = MCPService(config)

    app = build_app(service)

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
