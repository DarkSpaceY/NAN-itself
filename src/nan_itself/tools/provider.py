"""
Live tool provider.

One Provider wraps exactly one active backend:

    kind="mcp":
        stack + session (stdio child process)

    kind="local":
        instance (in-process LocalToolProvider object)

`tools` is the cached tool-definition table shared by views.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

import mcp.types as types
from loguru import logger
from pydantic import ValidationError

from .results import (
    error_result,
    serialize_value,
    text_result,
)
from .spec import (
    PROVIDER_KIND_LOCAL,
    ProviderSpec,
)


class Provider:
    """
    One live tool provider.

    Exactly one backend is active:

        - kind="mcp": stack + session (stdio child process)
        - kind="local": instance (in-process object)

    `tools` is the cached tool-definition table shared by views.
    """

    spec: ProviderSpec
    tools: dict[str, types.Tool]

    stack: AsyncExitStack | None = None
    session: Any | None = None

    # In-process LocalToolProvider instance. Deliberately duck-typed
    # so the provider layer does not depend on the local backend.
    instance: Any | None = None

    def __init__(
        self,
        *,
        spec: ProviderSpec,
        tools: dict[str, types.Tool],
        stack: AsyncExitStack | None = None,
        session: Any | None = None,
        instance: Any | None = None,
    ) -> None:
        self.spec = spec
        self.tools = tools
        self.stack = stack
        self.session = session
        self.instance = instance

    @property
    def kind(self) -> str:
        return self.spec.kind

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
    ) -> types.CallToolResult:
        if (
            self.spec.kind
            == PROVIDER_KIND_LOCAL
        ):
            return await self._call_local(
                name,
                arguments,
            )

        return await self.session.call_tool(
            name,
            arguments or {},
        )

    async def _call_local(
        self,
        name: str,
        arguments: dict[str, Any] | None,
    ) -> types.CallToolResult:
        provider = self.instance

        method = (
            provider.get_tool_method(name)
            if provider is not None
            else None
        )

        if method is None:
            return error_result(
                f"Tool '{name}' is not exposed "
                f"by local provider "
                f"'{self.spec.name}'."
            )

        try:
            result = await method.invoke(
                arguments
            )

        except ValidationError as exc:
            return error_result(
                f"Invalid arguments for tool "
                f"'{name}':\n{exc}"
            )

        except Exception as exc:
            logger.exception(
                "Local tool call failed: %s.%s",
                self.spec.name,
                name,
            )

            return error_result(
                f"{type(exc).__name__}: {exc}"
            )

        return text_result(
            serialize_value(result)
        )
