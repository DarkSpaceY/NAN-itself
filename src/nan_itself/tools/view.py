"""
Per-Agent tool exposure.

The runtime is shared by everyone; what an individual Agent can
see and call lives here. Progressive disclosure is the point:

    - only `route` is visible until a provider is activated
    - activating a provider reveals its whole tool block
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mcp.types as types

from .results import (
    error_result,
    text_result,
)

if TYPE_CHECKING:
    # Import cycle guard: the runtime instantiates views;
    # views only reference it for typing.
    from .runtime import (
        ProviderRuntime,
    )


ROUTE_TOOL_NAME = "route"
ROUTE_TOOL_ARGUMENT = "provider_name"


class AgentToolView:
    """
    Tool exposure for one Agent.

    Composes a shared ProviderRuntime; the runtime never knows
    about views. All routing state (`active_provider`) is local
    to this object.
    """

    def __init__(
        self,
        runtime: ProviderRuntime,
        *,
        active_provider: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.active_provider = active_provider

    def available_providers(self) -> tuple[str, ...]:
        return self.runtime.provider_names()

    async def list_tools(self) -> list[types.Tool]:
        tools = [
            self._route_tool(),
        ]

        if self.active_provider is None:
            return tools

        provider = self.runtime.get_provider(
            self.active_provider
        )

        if provider is None:
            # The provider may have disappeared from the
            # workspace since this view was created.
            self.active_provider = None
            return tools

        await self.runtime.refresh_provider_tools(
            self.active_provider
        )

        provider = self.runtime.get_provider(
            self.active_provider
        )

        if provider is None:
            self.active_provider = None
            return tools

        tools.extend(
            provider.tools.values()
        )

        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        arguments = arguments or {}

        if name == ROUTE_TOOL_NAME:
            return await self._route_provider(
                arguments.get(ROUTE_TOOL_ARGUMENT)
            )

        if self.active_provider is None:
            return error_result(
                "No tool provider is active. "
                "Call route first."
            )

        provider = self.runtime.get_provider(
            self.active_provider
        )

        if provider is None:
            self.active_provider = None

            return error_result(
                "The active tool provider is "
                "no longer available."
            )

        if name not in provider.tools:
            await self.runtime.refresh_provider_tools(
                self.active_provider
            )

            provider = self.runtime.get_provider(
                self.active_provider
            )

            if (
                provider is None
                or name not in provider.tools
            ):
                return error_result(
                    f"Tool '{name}' is not available "
                    f"from provider "
                    f"'{self.active_provider}'."
                )

        return await self.runtime.call_tool(
            self.active_provider,
            name,
            arguments,
        )

    async def _route_provider(
        self,
        provider_name: str | None,
    ) -> types.CallToolResult:
        if not provider_name:
            return error_result(
                f"{ROUTE_TOOL_ARGUMENT} is required"
            )

        if provider_name not in self.runtime.providers:
            return error_result(
                f"Unknown tool provider: {provider_name}. "
                f"Available: "
                f"{', '.join(self.runtime.provider_names())}"
            )

        if self.active_provider == provider_name:
            return text_result(
                f"Provider '{provider_name}' is "
                "already active."
            )

        self.active_provider = provider_name

        await self.runtime.refresh_provider_tools(
            provider_name
        )

        provider = self.runtime.get_provider(
            provider_name
        )

        tool_count = (
            len(provider.tools)
            if provider is not None
            else 0
        )

        return text_result(
            f"Provider '{provider_name}' activated. "
            f"{tool_count} tools are now available."
        )

    def _route_tool(self) -> types.Tool:
        names = list(
            self.runtime.provider_names()
        )

        return types.Tool(
            name=ROUTE_TOOL_NAME,
            description=(
                "选择要激活的工具组；激活后该组的工具才会"
                "出现在可见工具列表中。可选工具组："
                + ", ".join(
                    self.runtime.provider_names()
                )
                + "。\n"
                "若需要的能力不在当前可见工具中，"
                "先调用本工具激活对应工具组。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    ROUTE_TOOL_ARGUMENT: {
                        "type": "string",
                        "enum": names,
                        "description": (
                            "The tool provider to activate."
                        ),
                    }
                },
                "required": [ROUTE_TOOL_ARGUMENT],
                "additionalProperties": False,
            },
        )
