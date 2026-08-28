"""
MCP backend.

Owns everything specific to stdio MCP servers:

    - establishing connections (stdio client + session)
    - parsing MCP provider configuration from YAML

Both builtin and workspace MCP providers flow through here;
origin only tags the resulting specs.
"""

from __future__ import annotations

import os
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .provider import (
    Provider,
)
from .spec import (
    PROVIDER_KIND_MCP,
    ProviderOrigin,
    ProviderSpec,
)


async def connect(
    spec: ProviderSpec,
) -> Provider:
    """
    Start one stdio MCP server and cache its tool table.
    """
    command = spec.command

    if not command:
        raise ValueError(
            f"MCP '{spec.name}' is missing command"
        )

    resolved_command = (
        shutil.which(command) or command
    )

    # Local-first tools run on the user's machine: inherit the
    # parent environment by default so npm/uv caches, proxies and
    # toolchains behave exactly as in the user's own shell (the
    # MCP SDK's minimal default environment breaks npx-based
    # servers). An explicit spec.env still overlays on top.
    env = dict(os.environ)

    if spec.env is not None:
        env.update(
            dict(spec.env)
        )

    server_params = StdioServerParameters(
        command=resolved_command,
        args=list(spec.args),
        env=env,
        cwd=spec.cwd,
    )

    logger.info(
        "Starting MCP provider '{}': {} {}",
        spec.name,
        resolved_command,
        list(spec.args),
    )

    stack = AsyncExitStack()

    try:
        read, write = await stack.enter_async_context(
            stdio_client(server_params)
        )

        session = await stack.enter_async_context(
            ClientSession(read, write)
        )

        await session.initialize()

        result = await session.list_tools()

        provider = Provider(
            spec=spec,
            stack=stack,
            session=session,
            tools={
                tool.name: tool
                for tool in result.tools
            },
        )

        logger.info(
            "MCP '{}' connected, {} tools",
            spec.name,
            len(provider.tools),
        )

        return provider

    except BaseException:
        await stack.aclose()
        raise


async def refresh_tools(
    provider: Provider,
) -> None:
    """
    Re-list tools from the live session.
    """
    result = await provider.session.list_tools()

    provider.tools = {
        tool.name: tool
        for tool in result.tools
    }


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML object: {path}"
        )

    return config


def parse_builtin_config(
    config: dict[str, Any],
    source: Path,
) -> list[ProviderSpec]:
    """
    Builtin config only accepts the explicit mapping form:

        mcp_servers:
          name:
            command: ...
    """
    servers = config.get(
        "mcp_servers",
        {},
    )

    if not isinstance(servers, dict):
        raise ValueError(
            f"'mcp_servers' must be a mapping: {source}"
        )

    specs: list[ProviderSpec] = []

    for name, raw in servers.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"Invalid MCP config for '{name}'"
            )

        specs.append(
            spec_from_mapping(
                name=str(name),
                raw=raw,
                source=str(source),
                origin="builtin",
            )
        )

    return specs


def parse_workspace_config(
    config: dict[str, Any],
    source: Path,
) -> list[ProviderSpec]:
    """
    Workspace supports both:

    1. Single-MCP file:

        name: github
        command: npx
        args: [...]

    2. Multi-MCP file:

        mcp_servers:
          github:
            command: npx
            args: [...]
    """

    if "mcp_servers" in config:
        servers = config["mcp_servers"]

        if not isinstance(servers, dict):
            raise ValueError(
                f"'mcp_servers' must be a mapping: {source}"
            )

        result: list[ProviderSpec] = []

        for name, raw in servers.items():
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Invalid MCP config for '{name}'"
                )

            result.append(
                spec_from_mapping(
                    name=str(name),
                    raw=raw,
                    source=str(source),
                    origin="workspace",
                )
            )

        return result

    name = config.get(
        "name",
        source.stem,
    )

    if not isinstance(name, str) or not name:
        raise ValueError(
            f"Invalid workspace MCP name: {source}"
        )

    return [
        spec_from_mapping(
            name=name,
            raw=config,
            source=str(source),
            origin="workspace",
        )
    ]


def spec_from_mapping(
    *,
    name: str,
    raw: dict[str, Any],
    source: str,
    origin: ProviderOrigin,
) -> ProviderSpec:
    command = raw.get("command")

    if not isinstance(command, str) or not command:
        raise ValueError(
            f"MCP '{name}' is missing command"
        )

    args_raw = raw.get(
        "args",
        [],
    )

    if not isinstance(args_raw, list):
        raise ValueError(
            f"MCP '{name}'.args must be a list"
        )

    env_raw = raw.get("env")

    if env_raw is not None:
        if not isinstance(env_raw, dict):
            raise ValueError(
                f"MCP '{name}'.env must be a mapping"
            )

        env = MappingProxyType({
            str(key): str(value)
            for key, value in env_raw.items()
        })

    else:
        env = None

    cwd = raw.get("cwd")

    if cwd is not None:
        cwd = str(cwd)

    return ProviderSpec(
        name=name,
        kind=PROVIDER_KIND_MCP,
        command=command,
        args=tuple(
            str(value)
            for value in args_raw
        ),
        env=env,
        cwd=cwd,
        source=source,
        origin=origin,
    )
