"""
Local Python backend.

One concrete ToolSet subclass plays the same role as
one MCP server: it exposes a group of sub-tools that route
switches as a block.

This module owns:

    - the @tool decorator and method collection
    - JSON schema derivation from type hints
    - class validation for builtin registration
    - dynamic import of local tool provider files
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    get_type_hints,
)

import mcp.types as types
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    create_model,
)

from .provider import (
    Provider,
)
from .results import (
    error_result,
    text_result,
)
from .spec import (
    PROVIDER_KIND_LOCAL,
    ProviderSpec,
)

_LOCAL_TOOL_MARKER = "_nan_local_tool"

# Shared with the modules loader: main.py is the only entry.
MAIN_FILENAME = "main.py"


@dataclass(frozen=True)
class LocalToolMethod:
    """
    One method of a ToolSet exposed as a tool.

    `input_model` is a pydantic model derived from the method
    signature. It provides both the JSON schema handed to models
    and argument validation at call time.
    """

    name: str
    description: str
    handler: Callable[..., Any]
    input_model: type[BaseModel]

    def input_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()

        schema.pop("title", None)

        for property_schema in schema.get(
            "properties",
            {},
        ).values():
            property_schema.pop("title", None)

        return schema

    async def invoke(
        self,
        arguments: dict[str, Any] | None,
    ) -> Any:
        validated = self.input_model.model_validate(arguments or {})

        kwargs = validated.model_dump()

        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**kwargs)

        # A synchronous handler must never run on the event
        # loop: one blocking call (time.sleep, requests, ...)
        # would freeze every tool, module and agent, and the
        # runtime's tool_timeout could not fire. Offload it
        # to the default thread pool. If a sync handler
        # returns an awaitable it is awaited back on the loop.
        result = await asyncio.to_thread(
            self.handler,
            **kwargs,
        )

        if inspect.isawaitable(result):
            result = await result

        return result


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """
    Mark a method as an exposed tool.

    Usable as:

        @tool
        @tool()
        @tool(name="other_name", description="...")

    Without an explicit description the method docstring is used.
    Parameters must be type-annotated; the JSON schema derives
    from the annotations.
    """

    def decorate(
        target: Callable[..., Any],
    ) -> Callable[..., Any]:
        if not inspect.isfunction(target):
            raise TypeError("@tool can only decorate functions or methods")

        setattr(
            target,
            _LOCAL_TOOL_MARKER,
            {
                "name": name or target.__name__,
                "description": description,
            },
        )

        return target

    if func is not None:
        return decorate(func)

    return decorate


class ToolSet:
    """
    Base class for in-process Python class tool providers.

    The provider's identity is the folder it was loaded from;
    there is no separate id.
    """

    def __init__(self) -> None:
        self._tool_methods: dict[
            str,
            LocalToolMethod,
        ] = self._collect_tool_methods()

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _collect_tool_methods(
        self,
    ) -> dict[str, LocalToolMethod]:
        collected: dict[str, LocalToolMethod] = {}

        # Track which class registered each tool name so a
        # subclass override can be distinguished from a
        # genuine duplicate.
        owners: dict[str, type] = {}

        # Walk the MRO base-first: inherited tools register
        # first and a subclass that re-declares @tool on an
        # override replaces the base version. Two @tool
        # methods with the same name in ONE class remain an
        # error.
        for klass in reversed(type(self).__mro__):
            for _attr_name, attr_value in vars(klass).items():
                if not inspect.isfunction(attr_value):
                    continue

                marker = getattr(
                    attr_value,
                    _LOCAL_TOOL_MARKER,
                    None,
                )

                if marker is None:
                    continue

                method = self._build_tool_method(
                    attr_value,
                    marker,
                )

                if owners.get(method.name) is klass:
                    raise ValueError(
                        f"Local tool provider "
                        f"{type(self).__name__} "
                        f"duplicates tool name "
                        f"'{method.name}'"
                    )

                collected[method.name] = method

                owners[method.name] = klass

        return collected

    def _build_tool_method(
        self,
        func: Callable[..., Any],
        marker: dict[str, Any],
    ) -> LocalToolMethod:
        tool_name = marker["name"]

        if not isinstance(tool_name, str) or not tool_name:
            raise TypeError(
                f"@tool name on {type(self).__name__}.{func.__name__} must be a non-empty string"
            )

        description = marker.get("description")

        if description is None:
            description = (inspect.getdoc(func) or "").strip()

        input_model = self._build_input_model(
            func,
        )

        return LocalToolMethod(
            name=tool_name,
            description=description,
            handler=getattr(self, func.__name__),
            input_model=input_model,
        )

    def _build_input_model(
        self,
        func: Callable[..., Any],
    ) -> type[BaseModel]:
        signature = inspect.signature(func)

        parameters = list(signature.parameters.values())

        if parameters and parameters[0].name == "self":
            parameters = parameters[1:]

        try:
            hints = get_type_hints(func)
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve type hints for {type(self).__name__}.{func.__name__}: {exc}"
            ) from exc

        fields: dict[str, Any] = {}

        for parameter in parameters:
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"Tool method {type(self).__name__}.{func.__name__} does not support var args"
                )

            annotation = hints.get(
                parameter.name,
                parameter.annotation,
            )

            if annotation is inspect.Parameter.empty:
                raise ValueError(
                    f"Tool method "
                    f"{type(self).__name__}."
                    f"{func.__name__} parameter "
                    f"'{parameter.name}' must be "
                    "type-annotated"
                )

            if parameter.default is inspect.Parameter.empty:
                fields[parameter.name] = (
                    annotation,
                    ...,
                )

            else:
                fields[parameter.name] = (
                    annotation,
                    parameter.default,
                )

        model_name = "".join(part.title() for part in func.__name__.split("_"))

        return create_model(
            f"{type(self).__name__}{model_name}Input",
            # Unknown argument names are a model mistake and
            # must surface as a validation error, not be
            # silently ignored while defaults apply.
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._tool_methods)

    def tool_methods(
        self,
    ) -> tuple[LocalToolMethod, ...]:
        return tuple(self._tool_methods.values())

    def get_tool_method(
        self,
        name: str,
    ) -> LocalToolMethod | None:
        return self._tool_methods.get(name)

    def build_tools(
        self,
    ) -> dict[str, types.Tool]:
        return {
            method.name: types.Tool(
                name=method.name,
                description=method.description,
                inputSchema=(method.input_schema()),
            )
            for method in self._tool_methods.values()
        }


# ============================================================================
# Registration & loading helpers
# ============================================================================


def validate_class(
    cls: Any,
) -> None:
    if not (
        isinstance(cls, type)
        and issubclass(
            cls,
            ToolSet,
        )
        and cls is not ToolSet
        and not inspect.isabstract(cls)
    ):
        raise TypeError("Local tool provider must be a concrete ToolSet subclass")


def load_class_from_folder(
    folder: Path,
) -> tuple[
    type[ToolSet],
    str,
]:
    """
    Import one local tool folder and extract its class.

    The folder is imported as a synthetic package seeded with the
    local-tool vocabulary (ToolSet, tool, text_result,
    error_result). Returns the single concrete ToolSet subclass
    defined in main.py plus the package root name, so callers can
    evict the whole generation from sys.modules on reload or
    removal.
    """
    from nan_itself.utils import hotload

    pkg_name, main = hotload.load_folder_package(
        folder,
        kind="tool",
        seeds={
            "ToolSet": ToolSet,
            "tool": tool,
            "text_result": text_result,
            "error_result": error_result,
        },
    )

    try:
        candidates = [
            cls
            for _, cls in inspect.getmembers(main, inspect.isclass)
            if (
                cls.__module__ == main.__name__
                and issubclass(
                    cls,
                    ToolSet,
                )
                and cls is not ToolSet
                and not inspect.isabstract(cls)
            )
        ]

    except BaseException:
        hotload.evict_package(pkg_name)

        raise

    if len(candidates) != 1:
        hotload.evict_package(pkg_name)

        raise RuntimeError(
            f"{folder / MAIN_FILENAME} must contain exactly one concrete ToolSet; "
            f"found {len(candidates)}"
        )

    return (
        candidates[0],
        pkg_name,
    )


def build_provider(
    spec: ProviderSpec,
    cls: type[ToolSet],
) -> Provider:
    instance = cls()

    tools = instance.build_tools()

    if not tools:
        raise ValueError(f"Local tool provider '{spec.name}' exposes no @tool methods")

    logger.info(
        "Local tool provider '{}' ready, {} tools",
        spec.name,
        len(tools),
    )

    return Provider(
        spec=spec,
        tools=tools,
        instance=instance,
    )


def local_provider_spec(
    *,
    name: str,
    file: Path | None = None,
    source: str,
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        kind=PROVIDER_KIND_LOCAL,
        file=file,
        source=source,
    )
