"""
Local Python backend.

One concrete LocalToolProvider subclass plays the same role as
one MCP server: it exposes a group of sub-tools that route
switches as a block.

This module owns:

    - the @tool decorator and method collection
    - JSON schema derivation from type hints
    - class validation for builtin registration
    - dynamic import of workspace provider files
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    Callable,
    ClassVar,
    get_type_hints,
)

from loguru import logger
from pydantic import BaseModel, create_model

import mcp.types as types

from .provider import (
    Provider,
)
from .spec import (
    LOCAL_TOOL_HEADER,
    LOCAL_TOOL_HEADER_SCAN_LINES,
    PROVIDER_KIND_LOCAL,
    ProviderSpec,
)


_LOCAL_TOOL_MARKER = "_nan_local_tool"


@dataclass(frozen=True)
class LocalToolMethod:
    """
    One method of a LocalToolProvider exposed as a tool.

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
        validated = self.input_model.model_validate(
            arguments or {}
        )

        result = self.handler(
            **validated.model_dump()
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
            raise TypeError(
                "@tool can only decorate "
                "functions or methods"
            )

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


class LocalToolProvider:
    """
    Base class for in-process Python class tool providers.
    """

    id: ClassVar[str]

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

        # Walk the MRO base-first so inherited tools register
        # before overridden ones and source order is preserved.
        for klass in reversed(
            type(self).__mro__
        ):
            for attr_name, attr_value in vars(
                klass
            ).items():
                if not inspect.isfunction(
                    attr_value
                ):
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

                if method.name in collected:
                    raise ValueError(
                        f"Local tool provider "
                        f"{type(self).__name__} "
                        f"duplicates tool name "
                        f"'{method.name}'"
                    )

                collected[method.name] = method

        return collected

    def _build_tool_method(
        self,
        func: Callable[..., Any],
        marker: dict[str, Any],
    ) -> LocalToolMethod:
        tool_name = marker["name"]

        if (
            not isinstance(tool_name, str)
            or not tool_name
        ):
            raise TypeError(
                f"@tool name on "
                f"{type(self).__name__}."
                f"{func.__name__} must be a "
                "non-empty string"
            )

        description = marker.get("description")

        if description is None:
            description = (
                inspect.getdoc(func) or ""
            ).strip()

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

        parameters = list(
            signature.parameters.values()
        )

        if (
            parameters
            and parameters[0].name == "self"
        ):
            parameters = parameters[1:]

        try:
            hints = get_type_hints(func)
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve type hints for "
                f"{type(self).__name__}."
                f"{func.__name__}: {exc}"
            ) from exc

        fields: dict[str, Any] = {}

        for parameter in parameters:
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"Tool method "
                    f"{type(self).__name__}."
                    f"{func.__name__} does not "
                    "support var args"
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

            if (
                parameter.default
                is inspect.Parameter.empty
            ):
                fields[parameter.name] = (
                    annotation,
                    ...,
                )

            else:
                fields[parameter.name] = (
                    annotation,
                    parameter.default,
                )

        model_name = "".join(
            part.title()
            for part in func.__name__.split("_")
        )

        return create_model(
            f"{type(self).__name__}{model_name}"
            "Input",
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
        return tuple(
            self._tool_methods.values()
        )

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
                inputSchema=(
                    method.input_schema()
                ),
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
            LocalToolProvider,
        )
        and cls is not LocalToolProvider
        and not inspect.isabstract(cls)
    ):
        raise TypeError(
            "Local tool provider must be a "
            "concrete LocalToolProvider subclass"
        )

    provider_id = getattr(
        cls,
        "id",
        None,
    )

    if (
        not isinstance(provider_id, str)
        or not provider_id
    ):
        raise ValueError(
            f"Local tool provider "
            f"{getattr(cls, '__name__', cls)} must "
            "define a non-empty string id"
        )


def has_tool_header(
    path: Path,
) -> bool:
    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except OSError:
        return False

    head = "\n".join(
        text.splitlines()[
            :LOCAL_TOOL_HEADER_SCAN_LINES
        ]
    )

    return LOCAL_TOOL_HEADER in head


def load_class_from_file(
    path: Path,
) -> tuple[
    type[LocalToolProvider],
    str,
]:
    """
    Import a workspace provider file and extract its class.

    Returns the class plus the synthetic module name it was
    imported under, so callers can evict it from sys.modules
    on reload/removal.
    """
    path = path.resolve()

    digest = hashlib.sha1(
        str(path).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]

    module_name = (
        f"_workspace_tool_"
        f"{path.stem}_"
        f"{digest}"
    )

    sys.modules.pop(
        module_name,
        None,
    )

    # Compile from freshly read source instead of going through
    # the import system: the bytecode cache (__pycache__) keys on
    # coarse mtime + size, so an edit that keeps both (a one-
    # character change within the same timestamp tick) would
    # silently execute STALE code. Hot reload must never do that.
    try:
        source = path.read_text(
            encoding="utf-8"
        )

        code = compile(
            source,
            str(path),
            "exec",
        )

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    module: ModuleType = ModuleType(
        module_name
    )

    module.__file__ = str(path)

    module.__dict__["LocalToolProvider"] = LocalToolProvider
    module.__dict__["tool"] = tool

    sys.modules[
        module_name
    ] = module

    try:
        exec(
            code,
            module.__dict__,
        )

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    candidates = [
        cls
        for _, cls in inspect.getmembers(
            module,
            inspect.isclass,
        )
        if (
            cls.__module__
            == module.__name__
            and issubclass(
                cls,
                LocalToolProvider,
            )
            and cls is not LocalToolProvider
            and not inspect.isabstract(cls)
        )
    ]

    if len(candidates) != 1:
        sys.modules.pop(
            module_name,
            None,
        )

        raise RuntimeError(
            f"{path} must contain exactly "
            f"one concrete LocalToolProvider; "
            f"found {len(candidates)}"
        )

    return (
        candidates[0],
        module_name,
    )


def build_provider(
    spec: ProviderSpec,
    cls: type[LocalToolProvider],
) -> Provider:
    instance = cls()

    tools = instance.build_tools()

    if not tools:
        raise ValueError(
            f"Local tool provider "
            f"'{spec.name}' exposes no "
            "@tool methods"
        )

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
    origin: str,
) -> ProviderSpec:
    return ProviderSpec(
        name=name,
        kind=PROVIDER_KIND_LOCAL,
        file=file,
        source=source,
        origin=origin,  # type: ignore[arg-type]
    )
