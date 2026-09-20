"""
MCP backend.

Owns everything specific to stdio MCP servers:

    - establishing connections (stdio client + session)
    - parsing MCP provider configuration from YAML

One YAML file maps to exactly one provider (name defaults to the
file stem). This keeps every configuration file a hot-reloadable
entity: builtin and workspace configs follow the same rule.
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
from mcp import (
    ClientSession,
    StdioServerParameters,
)
from mcp.client.stdio import stdio_client

from nan_itself.utils import paths as _paths

from .provider import Provider
from .spec import (
    PROVIDER_KIND_MCP,
    ProviderSpec,
)


# ============================================================================
# Connection
# ============================================================================


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
    # toolchains behave exactly as in the user's own shell.
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
        read, write = (
            await stack.enter_async_context(
                stdio_client(server_params)
            )
        )

        session = (
            await stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                )
            )
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


# ============================================================================
# YAML
# ============================================================================


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        ) or {}

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            f"Invalid YAML object: {path}"
        )

    return config


# ============================================================================
# Provider configuration
# ============================================================================


def parse_config(
    config: dict[str, Any],
    source: Path,
) -> list[ProviderSpec]:
    """
    Parse exactly ONE MCP provider.

    Supported form:

        name: github
        command: npx
        args:
          - ...

    `name` is optional. When omitted, source.stem is used.

    Every configuration file obeys the strict:

        one file <-> one provider

    rule so it can be hot-reloaded as a single entity.

    The multi-provider form:

        mcp_servers:
          github:
            ...
          playwright:
            ...

    is rejected: split it into one file per provider.
    """
    if "mcp_servers" in config:
        raise ValueError(
            f"MCP source '{source}' must define "
            "exactly one provider; the 'mcp_servers' "
            "mapping is not supported. Split it into "
            "one file per provider."
        )

    name = config.get(
        "name",
        source.stem,
    )

    if (
        not isinstance(name, str)
        or not name
    ):
        raise ValueError(
            f"Invalid MCP name: {source}"
        )

    return [
        spec_from_mapping(
            name=name,
            raw=config,
            source=str(source),
        )
    ]


# ============================================================================
# Provider specification
# ============================================================================


def spec_from_mapping(
    *,
    name: str,
    raw: dict[str, Any],
    source: str,
) -> ProviderSpec:
    command = raw.get(
        "command"
    )

    if (
        not isinstance(command, str)
        or not command
    ):
        raise ValueError(
            f"MCP '{name}' is missing command"
        )

    args_raw = raw.get(
        "args",
        [],
    )

    if not isinstance(
        args_raw,
        list,
    ):
        raise ValueError(
            f"MCP '{name}'.args must be a list"
        )

    env_raw = raw.get(
        "env"
    )

    if env_raw is not None:
        if not isinstance(
            env_raw,
            dict,
        ):
            raise ValueError(
                f"MCP '{name}'.env must be a mapping"
            )

        env = MappingProxyType(
            {
                str(key): str(value)
                for key, value in env_raw.items()
            }
        )

    else:
        env = None

    cwd = raw.get(
        "cwd"
    )

    if cwd is not None:
        cwd = str(
            cwd
        )

        if not Path(cwd).is_absolute():
            # Relative cwd in an MCP config is repo-anchored,
            # never process-cwd-anchored: launching nan from
            # any directory must not move the server's
            # working directory (same rule as utils/paths.py).
            cwd = str(
                _paths.repo_root() / cwd
            )

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
    )