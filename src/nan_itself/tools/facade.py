from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    get_type_hints,
)

import mcp.server.stdio
import mcp.types as types
import yaml
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import (
    NotificationOptions,
    Server,
)
from mcp.server.models import InitializationOptions
from pydantic import BaseModel, ValidationError, create_model


MCP_SOURCE_BUILTIN = "builtin"
MCP_SOURCE_WORKSPACE = "workspace"

PROVIDER_KIND_MCP = "mcp"
PROVIDER_KIND_LOCAL = "local"

ProviderKind = Literal[
    "mcp",
    "local",
]

ProviderOrigin = Literal[
    "builtin",
    "workspace",
]

LOCAL_TOOL_HEADER = "# @tool"
LOCAL_TOOL_HEADER_SCAN_LINES = 20

# A hung tool call must never freeze the agent loop. Every call is
# bounded; timeouts surface to the model as error results.
DEFAULT_TOOL_TIMEOUT = 300.0

_LOCAL_TOOL_MARKER = "_nan_local_tool"

ROUTE_TOOL_NAME = "route"
ROUTE_TOOL_ARGUMENT = "provider_name"


# ============================================================================
# Result helpers
# ============================================================================


def _text_result(
    text: str,
) -> types.CallToolResult:
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=text,
            )
        ]
    )


def _error_result(
    text: str,
) -> types.CallToolResult:
    return types.CallToolResult(
        isError=True,
        content=[
            types.TextContent(
                type="text",
                text=text,
            )
        ],
    )


def _serialize_value(
    value: Any,
) -> str:
    if value is None:
        return "null"

    if isinstance(value, str):
        return value

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    except Exception:
        return str(value)


# ============================================================================
# Local Python class tools
# ============================================================================


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

    One concrete subclass plays the same role as one MCP server:
    it exposes a group of sub-tools that route switches as a block.

    Subclasses declare tools by decorating methods with @tool.
    Instantiation collects the decorated methods, so the provider
    may keep regular instance state between calls.
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


# Add builtin local tool classes here.
#
# Example:
#
# class ExampleTools(LocalToolProvider):
#     id = "example"
#
#     @tool(description="Return a greeting.")
#     def greet(self, who: str) -> str:
#         return f"hello {who}"
#
#
# BUILTIN_TOOLS = (ExampleTools,)

BUILTIN_TOOLS: tuple[
    type[LocalToolProvider],
    ...
] = ()


# ============================================================================
# Provider configuration
# ============================================================================


@dataclass(frozen=True)
class ProviderSpec:
    """
    Declarative configuration for one tool provider.

    kind="mcp":
        A stdio MCP server described by command/args/env/cwd.

    kind="local":
        An in-process Python class tool provider. `file` points at
        the workspace source it was loaded from, or stays None for
        builtin classes registered directly.
    """

    name: str

    kind: ProviderKind = PROVIDER_KIND_MCP

    command: str | None = None
    args: tuple[str, ...] = ()
    env: MappingProxyType[str, str] | None = None
    cwd: str | None = None

    file: Path | None = None

    source: str = "<unknown>"
    origin: ProviderOrigin = "builtin"


@dataclass
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
    session: ClientSession | None = None

    instance: LocalToolProvider | None = None

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
            return _error_result(
                f"Tool '{name}' is not exposed "
                f"by local provider "
                f"'{self.spec.name}'."
            )

        try:
            result = await method.invoke(
                arguments
            )

        except ValidationError as exc:
            return _error_result(
                f"Invalid arguments for tool "
                f"'{name}':\n{exc}"
            )

        except Exception as exc:
            logger.exception(
                "Local tool call failed: %s.%s",
                self.spec.name,
                name,
            )

            return _error_result(
                f"{type(exc).__name__}: {exc}"
            )

        return _text_result(
            _serialize_value(result)
        )


# ============================================================================
# Runtime
# ============================================================================


class ProviderRuntime:
    """
    Shared tool provider runtime.

    Responsibilities:
        - load builtin MCP configuration
        - register builtin local Python tool providers
        - discover workspace MCP configuration
        - discover workspace local Python tool sources
        - maintain provider sessions/instances
        - cache tool definitions
        - hot reload workspace providers
        - expose per-Agent tool views

    This object does NOT have an active provider field.

    Agent-specific tool exposure lives in AgentToolView.
    """

    def __init__(
        self,
        builtin_config_path: str | Path | None = None,
        workspace_mcp_dir: str | Path | None = None,
        workspace_local_dir: str | Path | None = None,
        *,
        builtin_tools: (
            tuple[type[LocalToolProvider], ...]
            | None
        ) = None,
        scan_interval: float = 1.0,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
    ) -> None:
        project_root = Path(__file__).resolve().parents[3]

        self.tool_timeout = tool_timeout

        self.builtin_config_path = (
            Path(builtin_config_path).resolve()
            if builtin_config_path is not None
            else (
                project_root
                / "data"
                / "yaml_config"
                / "tools.yaml"
            ).resolve()
        )

        self.workspace_mcp_dir = (
            Path(workspace_mcp_dir).resolve()
            if workspace_mcp_dir is not None
            else (
                project_root
                / "workspace"
                / "tools"
                / "mcps"
            ).resolve()
        )

        self.workspace_local_dir = (
            Path(workspace_local_dir).resolve()
            if workspace_local_dir is not None
            else (
                project_root
                / "workspace"
                / "tools"
                / "local"
            ).resolve()
        )

        self.scan_interval = scan_interval

        self.builtin_tools = (
            BUILTIN_TOOLS
            if builtin_tools is None
            else tuple(builtin_tools)
        )

        # Provider name -> live provider.
        self.providers: dict[str, Provider] = {}

        # ----------------------------------------------------------
        # Workspace MCP bookkeeping.
        # ----------------------------------------------------------

        # Source file -> fingerprint.
        self._workspace_fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        # Source file -> provider names loaded from it.
        self._workspace_sources: dict[
            Path,
            set[str],
        ] = {}

        # Source file -> load error for current fingerprint.
        self._workspace_errors: dict[
            Path,
            BaseException,
        ] = {}

        # ----------------------------------------------------------
        # Workspace local tool bookkeeping.
        # ----------------------------------------------------------

        # Source file -> fingerprint.
        self._local_fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        # Source file -> provider names loaded from it.
        self._local_sources: dict[
            Path,
            set[str],
        ] = {}

        # Source file -> load error for current fingerprint.
        self._local_errors: dict[
            Path,
            BaseException,
        ] = {}

        # Source file -> imported module name.
        self._local_imported_names: dict[
            Path,
            str,
        ] = {}

        self._supervisor_task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._stopping = False

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        if self._supervisor_task is not None:
            return

        self._stopping = False

        await self._load_builtin_config()
        self._load_builtin_tools()

        await self._scan_workspace()

        self._supervisor_task = asyncio.create_task(
            self._supervisor(),
            name="provider-runtime-supervisor",
        )

    async def stop(self) -> None:
        if self._stopping:
            return

        self._stopping = True

        supervisor = self._supervisor_task
        self._supervisor_task = None

        if supervisor is not None:
            supervisor.cancel()

            try:
                await supervisor
            except asyncio.CancelledError:
                pass

        for name in list(self.providers):
            await self._remove_provider(name)

        self._workspace_fingerprints.clear()
        self._workspace_sources.clear()
        self._workspace_errors.clear()

        self._local_fingerprints.clear()
        self._local_sources.clear()
        self._local_errors.clear()
        self._local_imported_names.clear()

    # ==================================================================
    # Agent views
    # ==================================================================

    def create_agent_view(
        self,
        *,
        active_provider: str | None = None,
    ) -> AgentToolView:
        """
        Create an independent per-Agent tool exposure view.

        The underlying providers remain shared.
        """
        return AgentToolView(
            runtime=self,
            active_provider=active_provider,
        )

    # ==================================================================
    # Provider access
    # ==================================================================

    def provider_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.providers))

    def get_provider(
        self,
        name: str,
    ) -> Provider | None:
        return self.providers.get(name)

    # ==================================================================
    # Provider lifecycle
    # ==================================================================

    async def _connect_provider(
        self,
        spec: ProviderSpec,
    ) -> Provider:
        if spec.name in self.providers:
            raise ValueError(
                f"Duplicate tool provider: {spec.name}"
            )

        command = spec.command

        if not command:
            raise ValueError(
                f"MCP '{spec.name}' is missing command"
            )

        resolved_command = (
            shutil.which(command) or command
        )

        env = None

        if spec.env is not None:
            env = {
                **os.environ,
                **dict(spec.env),
            }

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

    def _connect_local_provider(
        self,
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

    async def _remove_provider(
        self,
        name: str,
    ) -> None:
        provider = self.providers.pop(name, None)

        if provider is None:
            return

        await self._close_provider(provider)

        logger.info(
            "Tool provider '{}' removed",
            name,
        )

    @staticmethod
    async def _close_provider(
        provider: Provider,
    ) -> None:
        if provider.stack is None:
            # Local providers hold no external resources.
            return

        try:
            await provider.stack.aclose()

        except Exception:
            logger.exception(
                "Failed to close MCP provider '{}'",
                provider.spec.name,
            )

    async def refresh_provider_tools(
        self,
        name: str,
    ) -> None:
        provider = self.providers[name]

        if (
            provider.spec.kind
            == PROVIDER_KIND_LOCAL
        ):
            # Local tools are fixed at instantiation time.
            return

        result = await provider.session.list_tools()

        provider.tools = {
            tool.name: tool
            for tool in result.tools
        }

    async def call_tool(
        self,
        provider_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        provider = self.providers.get(provider_name)

        if provider is None:
            return _error_result(
                f"Unknown tool provider: {provider_name}"
            )

        arguments = arguments or {}

        try:
            return await asyncio.wait_for(
                provider.call_tool(
                    tool_name,
                    arguments,
                ),
                timeout=self.tool_timeout,
            )

        except asyncio.TimeoutError:
            logger.warning(
                "Tool call timed out: %s.%s "
                "(%gs)",
                provider_name,
                tool_name,
                self.tool_timeout,
            )

            return _error_result(
                f"Tool '{tool_name}' timed out "
                f"after {self.tool_timeout:g} seconds."
            )

        except Exception as exc:
            logger.exception(
                "Tool call failed: %s.%s",
                provider_name,
                tool_name,
            )

            return _error_result(str(exc))

    # ==================================================================
    # Builtin MCP configuration
    # ==================================================================

    async def _load_builtin_config(self) -> None:
        if not self.builtin_config_path.exists():
            logger.warning(
                "Builtin MCP config not found: {}",
                self.builtin_config_path,
            )
            return

        config = self._load_yaml(
            self.builtin_config_path
        )

        specs = self._parse_builtin_config(
            config,
            self.builtin_config_path,
        )

        for spec in specs:
            if spec.name in self.providers:
                raise ValueError(
                    f"Duplicate builtin tool provider: {spec.name}"
                )

            try:
                provider = await self._connect_provider(
                    spec
                )

            except Exception:
                # A single broken builtin must never prevent
                # the persistent process from booting.
                logger.exception(
                    "Failed to start builtin MCP "
                    "provider '{}'; skipping",
                    spec.name,
                )

                continue

            self.providers[spec.name] = provider

    @staticmethod
    def _parse_builtin_config(
        config: dict[str, Any],
        source: Path,
    ) -> list[ProviderSpec]:
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
                ProviderRuntime._spec_from_mapping(
                    name=str(name),
                    raw=raw,
                    source=str(source),
                    origin="builtin",
                )
            )

        return specs

    # ==================================================================
    # Builtin local tools
    # ==================================================================

    def _load_builtin_tools(self) -> None:
        for cls in self.builtin_tools:
            self._validate_local_tool_class(cls)

            if cls.id in self.providers:
                raise ValueError(
                    f"Duplicate builtin tool provider: {cls.id}"
                )

            spec = ProviderSpec(
                name=cls.id,
                kind=PROVIDER_KIND_LOCAL,
                source="<builtin>",
                origin="builtin",
            )

            self.providers[
                spec.name
            ] = self._connect_local_provider(
                spec,
                cls,
            )

    # ==================================================================
    # Workspace discovery
    # ==================================================================

    async def _scan_workspace(self) -> None:
        await self._scan_workspace_mcps()
        await self._scan_workspace_locals()

    async def _scan_workspace_mcps(self) -> None:
        self.workspace_mcp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_files = {
            path.resolve()
            for path in self.workspace_mcp_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".yaml", ".yml"}
            and not path.name.startswith("_")
        }

        known_files = set(
            self._workspace_sources
        )

        # --------------------------------------------------------------
        # Removed provider files
        # --------------------------------------------------------------

        for path in known_files - current_files:
            await self._remove_workspace_source(path)

        # --------------------------------------------------------------
        # New / changed provider files
        # --------------------------------------------------------------

        for path in sorted(current_files):
            fingerprint = self._fingerprint(path)

            previous = self._workspace_fingerprints.get(
                path
            )

            previous_error = self._workspace_errors.get(
                path
            )

            if (
                previous == fingerprint
                and previous_error is not None
            ):
                continue

            if (
                previous == fingerprint
                and path in self._workspace_sources
            ):
                continue

            self._workspace_fingerprints[path] = fingerprint
            self._workspace_errors.pop(path, None)

            try:
                await self._reload_workspace_source(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._workspace_errors[path] = exc

                logger.exception(
                    "Failed to load workspace MCP config: {}",
                    path,
                )

    async def _scan_workspace_locals(self) -> None:
        self.workspace_local_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_files = {
            path.resolve()
            for path in self.workspace_local_dir.rglob(
                "*.py"
            )
            if path.is_file()
            and not path.name.startswith("_")
            and self._has_tool_header(path)
        }

        known_files = set(
            self._local_sources
        )

        # --------------------------------------------------------------
        # Removed provider files
        # --------------------------------------------------------------

        for path in known_files - current_files:
            await self._remove_local_source(path)

        # --------------------------------------------------------------
        # New / changed provider files
        # --------------------------------------------------------------

        for path in sorted(current_files):
            fingerprint = self._fingerprint(path)

            previous = self._local_fingerprints.get(
                path
            )

            previous_error = self._local_errors.get(
                path
            )

            if (
                previous == fingerprint
                and previous_error is not None
            ):
                continue

            if (
                previous == fingerprint
                and path in self._local_sources
            ):
                continue

            self._local_fingerprints[path] = fingerprint
            self._local_errors.pop(path, None)

            try:
                await self._reload_local_source(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._local_errors[path] = exc

                logger.exception(
                    "Failed to load workspace local tool: {}",
                    path,
                )

    async def _reload_workspace_source(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        config = self._load_yaml(path)

        specs = self._parse_workspace_config(
            config,
            path,
        )

        names = [spec.name for spec in specs]

        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate MCP provider name in {path}"
            )

        # Workspace providers may not shadow builtin providers.
        for spec in specs:
            existing = self.providers.get(spec.name)

            if existing is not None:
                if existing.spec.origin == "builtin":
                    raise ValueError(
                        f"Workspace MCP '{spec.name}' "
                        f"cannot override builtin tool provider"
                    )

                source = Path(
                    existing.spec.source
                ).resolve()

                if (
                    existing.spec.origin == "workspace"
                    and source != path.resolve()
                ):
                    raise ValueError(
                        f"Workspace MCP '{spec.name}' "
                        f"is already provided by {source}"
                    )

        # --------------------------------------------------------------
        # Connect every candidate first.
        #
        # If any candidate fails, current source remains untouched.
        # --------------------------------------------------------------

        candidates: dict[str, Provider] = {}

        try:
            for spec in specs:
                candidate = await self._connect_provider(
                    spec
                )

                candidates[spec.name] = candidate

        except BaseException:
            for provider in candidates.values():
                await self._close_provider(provider)

            raise

        # --------------------------------------------------------------
        # Commit source replacement.
        # --------------------------------------------------------------

        old_names = self._workspace_sources.get(
            path.resolve(),
            set(),
        )

        self._workspace_sources[path.resolve()] = set(
            candidates
        )

        for name in old_names - set(candidates):
            provider = self.providers.pop(name, None)

            if provider is not None:
                await self._close_provider(provider)

        for name, candidate in candidates.items():
            old = self.providers.get(name)

            self.providers[name] = candidate

            if old is not None:
                await self._close_provider(old)

        self._workspace_fingerprints[
            path.resolve()
        ] = fingerprint

        self._workspace_errors.pop(
            path.resolve(),
            None,
        )

        logger.info(
            "Loaded workspace MCP source '{}': {}",
            path,
            sorted(candidates),
        )

    async def _remove_workspace_source(
        self,
        path: Path,
    ) -> None:
        path = path.resolve()

        names = self._workspace_sources.pop(
            path,
            set(),
        )

        for name in names:
            provider = self.providers.get(name)

            if provider is None:
                continue

            if provider.spec.origin != "workspace":
                continue

            if Path(
                provider.spec.source
            ).resolve() != path:
                continue

            self.providers.pop(name, None)

            await self._close_provider(provider)

        self._workspace_fingerprints.pop(
            path,
            None,
        )

        self._workspace_errors.pop(
            path,
            None,
        )

        logger.info(
            "Removed workspace MCP source '{}'",
            path,
        )

    # ==================================================================
    # Workspace local tools
    # ==================================================================

    async def _reload_local_source(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        path = path.resolve()

        old_names = self._local_sources.get(
            path,
            set(),
        )

        (
            cls,
            imported_name,
        ) = self._import_local_class(path)

        self._validate_local_tool_class(cls)

        provider_name = cls.id

        existing = self.providers.get(
            provider_name
        )

        if existing is not None:
            if existing.spec.origin == "builtin":
                raise ValueError(
                    f"Workspace local tool "
                    f"'{provider_name}' cannot "
                    f"override builtin tool provider"
                )

            source = Path(
                existing.spec.source
            ).resolve()

            if (
                existing.spec.origin == "workspace"
                and source != path
            ):
                raise ValueError(
                    f"Workspace local tool "
                    f"'{provider_name}' is already "
                    f"provided by {source}"
                )

        candidate_spec = ProviderSpec(
            name=provider_name,
            kind=PROVIDER_KIND_LOCAL,
            file=path,
            source=str(path),
            origin="workspace",
        )

        # Build the candidate first; failure leaves the
        # previously loaded source untouched.
        candidate = self._connect_local_provider(
            candidate_spec,
            cls,
        )

        previous_imported = (
            self._local_imported_names.get(path)
        )

        if (
            previous_imported is not None
            and previous_imported != imported_name
        ):
            sys.modules.pop(
                previous_imported,
                None,
            )

        for name in old_names - {provider_name}:
            stale = self.providers.pop(name, None)

            if stale is not None:
                await self._close_provider(stale)

        old = self.providers.get(provider_name)

        self.providers[provider_name] = candidate

        if old is not None:
            await self._close_provider(old)

        self._local_sources[path] = {
            provider_name,
        }

        self._local_imported_names[path] = (
            imported_name
        )

        self._local_fingerprints[path] = (
            fingerprint
        )

        self._local_errors.pop(path, None)

        logger.info(
            "Loaded workspace local tool source '{}': {}",
            path,
            provider_name,
        )

    async def _remove_local_source(
        self,
        path: Path,
    ) -> None:
        path = path.resolve()

        names = self._local_sources.pop(
            path,
            set(),
        )

        for name in names:
            provider = self.providers.get(name)

            if provider is None:
                continue

            if provider.spec.origin != "workspace":
                continue

            if provider.spec.kind != PROVIDER_KIND_LOCAL:
                continue

            if Path(
                provider.spec.source
            ).resolve() != path:
                continue

            self.providers.pop(name, None)

            await self._close_provider(provider)

        imported_name = (
            self._local_imported_names.pop(
                path,
                None,
            )
        )

        if imported_name is not None:
            sys.modules.pop(
                imported_name,
                None,
            )

        self._local_fingerprints.pop(
            path,
            None,
        )

        self._local_errors.pop(
            path,
            None,
        )

        logger.info(
            "Removed workspace local tool source '{}'",
            path,
        )

    @staticmethod
    def _has_tool_header(
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

    def _import_local_class(
        self,
        path: Path,
    ) -> tuple[
        type[LocalToolProvider],
        str,
    ]:
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

        spec = (
            importlib.util.spec_from_file_location(
                module_name,
                path,
            )
        )

        if (
            spec is None
            or spec.loader is None
        ):
            raise ImportError(
                f"Could not create import spec "
                f"for {path}"
            )

        module: ModuleType = (
            importlib.util.module_from_spec(
                spec
            )
        )

        sys.modules[
            module_name
        ] = module

        try:
            spec.loader.exec_module(
                module
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

    @staticmethod
    def _validate_local_tool_class(
        cls: Any,
    ) -> None:
        if not (
            isinstance(cls, type)
            and issubclass(
                cls,
                LocalToolProvider,
            )
            and cls is not LocalToolProvider
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

    # ==================================================================
    # YAML parsing
    # ==================================================================

    @staticmethod
    def _load_yaml(
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

    @staticmethod
    def _parse_workspace_config(
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
              browser:
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
                    ProviderRuntime._spec_from_mapping(
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
            ProviderRuntime._spec_from_mapping(
                name=name,
                raw=config,
                source=str(source),
                origin="workspace",
            )
        ]

    @staticmethod
    def _spec_from_mapping(
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

    # ==================================================================
    # Supervisor
    # ==================================================================

    async def _supervisor(self) -> None:
        while not self._stopping:
            try:
                await self._scan_workspace()
            except Exception:
                logger.exception(
                    "Provider workspace reconciliation failed"
                )

            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self.scan_interval,
                )
            except asyncio.TimeoutError:
                pass
            finally:
                self._wake.clear()

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _fingerprint(
        path: Path,
    ) -> tuple[int, int]:
        stat = path.stat()

        return (
            stat.st_mtime_ns,
            stat.st_size,
        )


# ============================================================================
# Agent view
# ============================================================================


class AgentToolView:
    """
    Tool exposure for one Agent.

    The underlying ProviderRuntime is shared.

    active_provider is intentionally local to this object.
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
            return _error_result(
                "No tool provider is active. "
                "Call route first."
            )

        provider = self.runtime.get_provider(
            self.active_provider
        )

        if provider is None:
            self.active_provider = None

            return _error_result(
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
                return _error_result(
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
            return _error_result(
                f"{ROUTE_TOOL_ARGUMENT} is required"
            )

        if provider_name not in self.runtime.providers:
            return _error_result(
                f"Unknown tool provider: {provider_name}. "
                f"Available: "
                f"{', '.join(self.runtime.provider_names())}"
            )

        if self.active_provider == provider_name:
            return _text_result(
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

        return _text_result(
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
                "出现在可见工具列表中。\n"
                "[Tool Groups] "
                + ", ".join(
                    self.runtime.provider_names()
                )
                + "\n"
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


# ============================================================================
# Facade
# ============================================================================


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
        builtin_tools: (
            tuple[type[LocalToolProvider], ...]
            | None
        ) = None,
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
        self.view: AgentToolView | None = None

        self._register_handlers()

    @property
    def servers(
        self,
    ) -> dict[str, ClientSession]:
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

        self.view = self.runtime.create_agent_view()

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
                return _error_result(
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
        config_path=Path(
            "./data/yaml_config/tools.yaml"
        )
    )

    await facade.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
