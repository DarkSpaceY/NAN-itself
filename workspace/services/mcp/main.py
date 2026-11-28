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
import base64
import binascii
import contextlib
import json
import os
import re
import secrets
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
from nan_service_kit import HubLink, ServiceConfig, health_route, read_config, setup_logging

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"

CACHE_DIR = "cache"  # package-manager caches; regenerable, safe to delete

# Non-text tool results (images, audio, PDFs, 3D assets) are written here,
# under the service's own directory, and reported as workspace paths.
ARTIFACT_DIR = "data"

# Budgets for regenerable data. Trimming aims below the budget so a full
# cache does not evict again on the very next write.
CACHE_BUDGET = 1 << 30  # 1 GiB: npm, uv and playwright caches
ARTIFACT_BUDGET = 500 << 20  # 500 MiB: saved tool results
TRIM_TARGET = 0.9

MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "video/mp4": ".mp4",
    "application/pdf": ".pdf",
    "model/gltf+json": ".gltf",
    "model/gltf-binary": ".glb",
}

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _extension_for(mime: str) -> str:
    base = mime.split(";", 1)[0].strip().lower()

    if base in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[base]

    if base.startswith("image/"):
        return "." + base.split("/", 1)[1].replace("+xml", "")

    return ".bin"


def _safe_name(value: str) -> str:
    cleaned = _SAFE.sub("-", value).strip("-._")

    return cleaned[:60] or "tool"


def _trim_lru(root: Path, budget: int, target: float = TRIM_TARGET) -> int:
    """
    Delete the least recently modified files under `root` until it fits
    `budget * target`; only when it is over `budget` at all. Returns the
    bytes freed. Regenerable data only: a deleted file comes back on the
    next run that needs it.
    """
    if not root.is_dir():
        return 0

    entries: list[tuple[float, Path]] = []

    total = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        with contextlib.suppress(OSError):
            stat = path.stat()

            entries.append((stat.st_mtime, path))

            total += stat.st_size

    if total <= budget:
        return 0

    goal = int(budget * target)

    freed = 0

    for _, path in sorted(entries):
        if total <= goal:
            break

        with contextlib.suppress(OSError):
            size = path.stat().st_size

            path.unlink()

            total -= size

            freed += size

    return freed


# The spec's five content kinds for a tool result. Anything else is a
# warning, not a guess: the union is closed and has no extension clause.
_KIND_BY_CLASS = {
    "TextContent": "text",
    "ImageContent": "image",
    "AudioContent": "audio",
    "EmbeddedResource": "resource",
    "ResourceLink": "resource_link",
}


def _content_kind(block: Any) -> str:
    """The spec's `type` field; the class name is a fallback for stubs."""
    kind = getattr(block, "type", None)

    if isinstance(kind, str) and kind:
        return kind

    return _KIND_BY_CLASS.get(type(block).__name__, "")


def _decode(raw: Any) -> bytes | None:
    """A base64 payload, as the spec requires for image/audio/blob; None
    when it is missing or not decodable."""
    if not isinstance(raw, str) or not raw:
        return None

    try:
        return base64.b64decode(raw)

    except (binascii.Error, ValueError):
        return None


def _text_of(block: Any) -> str | None:
    """Text the block carries: a text block, or a resource's text body."""
    kind = _content_kind(block)

    if kind == "text":
        text = getattr(block, "text", None)

        return text if isinstance(text, str) else None

    if kind == "resource":
        text = getattr(getattr(block, "resource", None), "text", None)

        return text if isinstance(text, str) else None

    return None


def _link_of(block: Any) -> dict[str, Any]:
    """A resource_link: a reference only, so there is nothing to save."""
    link = {
        key: value
        for key in ("uri", "name", "title", "description", "mimeType", "size")
        if (value := getattr(block, key, None)) is not None
    }

    # The model may hand back a URL object rather than a plain string.
    if "uri" in link:
        link["uri"] = str(link["uri"])

    return link


def _error(code: str, message: str) -> dict[str, Any]:
    """A failed call result with a machine-readable code."""
    return {"ok": False, "error": {"code": code, "message": message}}


def _maybe_json(text: str) -> Any:
    """Tool text is usually JSON; keep plain text as a string."""
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        return text


class ServerSpec:
    """
    One configured MCP server: identity is (name, definition).
    """

    def __init__(self, raw: dict[str, Any], placeholders: dict[str, str]) -> None:
        self.name: str = str(raw["name"])

        self.command: str = str(raw["command"])

        args = raw.get("args", [])

        self.args: list[str] = [self._expand(a, placeholders) for a in args]

        self.cwd: str | None = self._expand(raw["cwd"], placeholders) if raw.get("cwd") else None

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

    def __init__(
        self,
        spec: ServerSpec,
        config: Config,
        *,
        artifact_root: Path | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.spec = spec

        self.config = config

        self.artifact_root = artifact_root or config.home / ARTIFACT_DIR

        self.workspace = (
            workspace or Path(os.environ.get("NAN_WORKSPACE", str(config.home.parents[1])))
        ).resolve()

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
            "UV_TOOL_DIR": str(cache / "uv-tools"),
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

            logger.info("MCP server {}: up ({} tools)", self.spec.name, len(self.tools))

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
                return _error("server_unavailable", f"server {self.spec.name} is not running")

            try:
                result = await asyncio.wait_for(
                    self._client.call_tool(tool, arguments), timeout=self.config.call_timeout
                )

            except TimeoutError:
                return _error(
                    "timeout",
                    f"{self.spec.name}.{tool} timed out after {self.config.call_timeout}s",
                )

            except Exception as exc:
                return _error("tool_error", str(exc))

        texts = [text for block in result.content if (text := _text_of(block)) is not None]

        text = "\n".join(texts).strip()

        if result.is_error:
            return _error("tool_error", text or f"{self.spec.name}.{tool} returned an error")

        files: list[dict[str, Any]] = []

        links: list[dict[str, Any]] = []

        warnings: list[str] = []

        for block in result.content:
            kind = _content_kind(block)

            # Text blocks, and resources whose body is text, were read above.
            if _text_of(block) is not None:
                continue

            if kind in ("image", "audio"):
                mime = str(getattr(block, "mimeType", None) or "application/octet-stream")

                data = _decode(getattr(block, "data", None))

                if data is None:
                    warnings.append(f"{kind} content without valid base64 data")

                else:
                    files.append(self._save_artifact(tool, data, mime, len(files) + 1))

                continue

            if kind == "resource":
                resource = getattr(block, "resource", None)

                mime = str(getattr(resource, "mimeType", None) or "application/octet-stream")

                data = _decode(getattr(resource, "blob", None))

                if data is None:
                    warnings.append("resource content without a text or blob body")

                else:
                    files.append(self._save_artifact(tool, data, mime, len(files) + 1))

                continue

            if kind == "resource_link":
                links.append(_link_of(block))

                continue

            warnings.append(f"unsupported content type: {kind or type(block).__name__}")

        if not (texts or files or links or warnings):
            if getattr(result, "data", None) is not None:
                return {"ok": True, "result": result.data}

            return {"ok": True, "result": ""}

        reply: dict[str, Any] = {"ok": True}

        if texts:
            reply["result"] = _maybe_json(text)

        if files:
            reply["files"] = files

        if links:
            reply["links"] = links

        if warnings:
            reply["warnings"] = warnings

        return reply

    def _save_artifact(self, tool: str, data: bytes, mime: str, index: int) -> dict[str, Any]:
        """
        Write one non-text block under the service's data dir and report
        it as a workspace-relative path, so `attach` can read it next.
        """
        directory = self.artifact_root / _safe_name(self.spec.name)

        directory.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())

        name = f"{stamp}-{_safe_name(tool)}-{index}-{secrets.token_hex(3)}{_extension_for(mime)}"

        path = directory / name

        path.write_bytes(data)

        # One walk per artifact keeps the budget exact even when several
        # servers share the root; the files are few and the walk is cheap.
        _trim_lru(self.artifact_root, ARTIFACT_BUDGET)

        try:
            shown = str(path.relative_to(self.workspace))

        except ValueError:
            shown = str(path)

        return {"path": shown, "mime": mime, "bytes": len(data)}


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

        self._failed: dict[str, str] = {}

        # Names inside a start attempt right now. Kept apart from the
        # queued specs, which have not been touched yet.
        self._starting: set[str] = set()

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

                self._running.pop(spec.name, None)

            server = MCPServer(spec, self.config)

        self._starting.add(spec.name)

        try:
            await server.start()

        except Exception as exc:
            logger.warning("MCP server {}: failed to start: {}", spec.name, exc)

            self._failed[spec.name] = str(exc)

            return

        finally:
            self._starting.discard(spec.name)

        async with self._lifecycle:
            self._running[spec.name] = server

            self._failed.pop(spec.name, None)

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

            for name in list(self._failed):
                if name not in wanted:
                    del self._failed[name]

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
        pointer and the category list are injected when the context
        that held them is gone.
        """
        if turn.get("origin") not in ("first", "context_cleared"):
            return ""

        text = (
            f"External MCP tool servers: `mcp` (on PATH) or `{HERE}/mcp.sh`; "
            "`mcp list` shows connected tools, queued and starting servers, and unavailable ones, "
            "`mcp schema <server> <tool>` shows the arguments, "
            "and `mcp call <server> <tool> '<json>'` prints the tool result. "
            "A non-text result (image, audio, PDF, 3D asset) is saved under "
            "`services/mcp/data/` and printed as a workspace path, so `attach` can read it; "
            "a resource link is printed as a reference, and an unknown content type is "
            "reported as a warning and skipped. "
            "On failure the call exits non-zero and writes a JSON error to stderr."
        )

        names = sorted({spec.name for spec in self.specs})

        if names:
            text += f" Categories: {', '.join(names)}."

        return text


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

    health_route(app, service.config)

    # ---------- inject endpoint ----------

    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        return {"text": service.contribution(body)}

    # ---------- tool API ----------

    @app.get("/tools")
    async def tools() -> dict[str, Any]:
        servers = [
            {"name": name, "tools": server.tools}
            for name, server in sorted(service._running.items())
        ]

        # Configured, but not up, failed, or being started right now:
        # the background starter has not reached them yet, so the
        # catalog is incomplete rather than empty.
        pending = sorted(
            spec.name
            for spec in service.specs
            if spec.name not in service._running
            and spec.name not in service._failed
            and spec.name not in service._starting
        )

        return {
            "servers": servers,
            "starting": sorted(service._starting),
            "pending": pending,
            "unavailable": dict(sorted(service._failed.items())),
        }

    @app.get("/schema/{server}/{tool}")
    async def schema(server: str, tool: str) -> dict[str, Any]:
        entry = service._running.get(server)

        if entry is None:
            return _error("unknown_server", f"unknown server: {server}")

        for candidate in entry.tools:
            if candidate["name"] == tool:
                return {
                    "ok": True,
                    "server": server,
                    "tool": tool,
                    "description": candidate.get("description", ""),
                    "schema": candidate["input_schema"],
                }

        return _error("unknown_tool", f"unknown tool: {server}.{tool}")

    @app.post("/call")
    async def call(body: dict[str, Any]) -> dict[str, Any]:
        server = str(body.get("server", ""))

        tool = str(body.get("tool", ""))

        arguments = body.get("arguments", {})

        if arguments is None:
            arguments = {}

        entry = service._running.get(server)

        if entry is None:
            return _error("unknown_server", f"unknown server: {server}")

        if not isinstance(arguments, dict):
            return _error("invalid_arguments", "arguments must be a JSON object")

        if not any(candidate["name"] == tool for candidate in entry.tools):
            return _error("unknown_tool", f"unknown tool: {server}.{tool}")

        return await entry.call(tool, arguments)

    return app


def main() -> None:
    config = load_config()

    setup_logging(config)

    # Shrink the regenerable caches before any server can be using them;
    # a later `npx`/`uvx` simply refetches what was evicted. Kept out of
    # the app lifespan so tests never touch a real cache.
    freed = _trim_lru(config.home / CACHE_DIR, CACHE_BUDGET)

    if freed:
        logger.info("cache trimmed: {} bytes freed", freed)

    service = MCPService(config)

    app = build_app(service)

    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
