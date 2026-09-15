"""
Standalone MCP server shell.

Wraps the whole provider runtime as a single stdio MCP server,
so external MCP clients can consume every backend through one
connection. The agent process itself does not use this file;
it talks to ProviderRuntime directly.

Tools are exposed under their 'provider/tool' composite names —
the same addressing the agent verbs use.
"""

from __future__ import annotations

import asyncio
from typing import Any

import mcp.server.stdio
import mcp.types as types
from loguru import logger
from mcp.server.lowlevel import (
    NotificationOptions,
    Server,
)
from mcp.server.models import InitializationOptions

from .results import (
    error_result,
)
from .runtime import (
    ProviderRuntime,
)


class MCPFacade:
    """
    MCP stdio-server shell around ProviderRuntime.
    """

    def __init__(
        self,
        *,
        builtin_tools_dir=None,
        workspace_mcp_dir=None,
        workspace_local_dir=None,
        scan_interval: float = 1.0,
    ) -> None:
        self.runtime = ProviderRuntime(
            builtin_tools_dir=builtin_tools_dir,
            workspace_mcp_dir=workspace_mcp_dir,
            workspace_local_dir=workspace_local_dir,
            scan_interval=scan_interval,
        )

        self.server = Server("mcp-facade")

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

        logger.info(
            "MCPFacade started: {}",
            list(self.runtime.provider_names()),
        )

    async def close(self) -> None:
        await self.runtime.stop()

        logger.info(
            "MCPFacade stopped"
        )

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                tool
                for _, tool in (
                    self.runtime.list_all_tools()
                )
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> types.CallToolResult:
            provider_name, _, tool_name = (
                name.partition("/")
            )

            if not tool_name:
                return error_result(
                    f"Tool name must be 'provider/tool': "
                    f"{name}"
                )

            resolved = (
                await self.runtime.resolve_tool(
                    name
                )
            )

            if resolved is None:
                return error_result(
                    f"Unknown tool: {name}"
                )

            provider, tool = resolved

            return await self.runtime.call_tool(
                provider.spec.name,
                tool.name,
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
    facade = MCPFacade()

    await facade.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
