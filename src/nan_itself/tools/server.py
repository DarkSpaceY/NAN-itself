"""
Standalone MCP server shell.

Wraps the whole provider runtime as a single stdio MCP server,
so external MCP clients can consume every backend through one
connection. The agent process itself does not use this file;
it talks to ProviderRuntime directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from loguru import logger
from mcp.server.lowlevel import (
    NotificationOptions,
    Server,
)
from mcp.server.models import InitializationOptions

from src.nan_itself.tools.builtin import (
    BUILTIN_MCP_CONFIG,
)
from src.nan_itself.tools.results import (
    error_result,
)
from src.nan_itself.tools.runtime import (
    ProviderRuntime,
)
from src.nan_itself.tools.view import (
    AgentToolView,
)


class MCPFacade:
    """
    Compatibility wrapper around ProviderRuntime.

    This preserves the existing MCP stdio-server use case while the
    underlying runtime now supports MCP and local Python providers
    through independent Agent views.
    """

    def __init__(
        self,
        config_path: Path,
        workspace_mcp_dir: str | Path | None = None,
        workspace_local_dir: str | Path | None = None,
        *,
        builtin_tools: Any | None = None,
        scan_interval: float = 1.0,
    ) -> None:
        self.runtime = ProviderRuntime(
            builtin_config_path=config_path,
            workspace_mcp_dir=workspace_mcp_dir,
            workspace_local_dir=workspace_local_dir,
            builtin_tools=builtin_tools,
            scan_interval=scan_interval,
        )

        self.server = Server("mcp-facade")
        self.view = None

        self._register_handlers()

    @property
    def servers(
        self,
    ) -> dict[str, Any]:
        """
        Compatibility view for the previous MCPFacade API.
        """
        return {
            name: provider.session
            for name, provider
            in self.runtime.providers.items()
            if provider.session is not None
        }

    @property
    def tools(
        self,
    ) -> dict[str, dict[str, types.Tool]]:
        """
        Compatibility view for the previous MCPFacade API.
        """
        return {
            name: provider.tools
            for name, provider
            in self.runtime.providers.items()
        }

    async def start(self) -> None:
        await self.runtime.start()

        self.view = AgentToolView(self.runtime)

        logger.info(
            "MCPFacade started: {}",
            list(self.runtime.provider_names()),
        )

    async def close(self) -> None:
        await self.runtime.stop()
        self.view = None

        logger.info(
            "MCPFacade stopped"
        )

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            if self.view is None:
                return []

            return await self.view.list_tools()

        @self.server.call_tool()
        async def call_tool(
            name: str,
            arguments: dict[str, Any],
        ) -> types.CallToolResult:
            if self.view is None:
                return error_result(
                    "MCPFacade is not started."
                )

            return await self.view.call_tool(
                name,
                arguments,
            )

    async def run_stdio(self) -> None:
        await self.start()

        try:
            async with mcp.server.stdio.stdio_server() as (
                read_stream,
                write_stream,
            ):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="mcp-facade",
                        server_version="2.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(
                                tools_changed=True,
                            ),
                            experimental_capabilities={},
                        ),
                    ),
                )
        finally:
            await self.close()


async def main() -> None:
    facade = MCPFacade(
        config_path=BUILTIN_MCP_CONFIG,
    )

    await facade.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
