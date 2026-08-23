from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

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


MCP_SOURCE_BUILTIN = "builtin"
MCP_SOURCE_WORKSPACE = "workspace"


@dataclass(frozen=True)
class MCPProviderSpec:
    """
    Declarative configuration for one MCP server.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: MappingProxyType[str, str] | None = None
    cwd: str | None = None

    source: str = "<unknown>"
    origin: Literal["builtin", "workspace"] = "builtin"


@dataclass
class MCPProvider:
    """
    One live MCP connection.

    The provider owns its own AsyncExitStack so it can be stopped
    independently of other MCP providers.
    """

    spec: MCPProviderSpec
    stack: AsyncExitStack
    session: ClientSession
    tools: dict[str, types.Tool]


class MCPRuntime:
    """
    Shared MCP runtime.

    Responsibilities:
        - load builtin MCP configuration
        - discover workspace MCP configuration
        - maintain MCP server sessions
        - cache MCP tool definitions
        - hot reload workspace MCPs
        - expose per-Agent MCP views

    This object does NOT have an active_mcp field.

    Agent-specific MCP exposure lives in AgentMCPView.
    """

    def __init__(
        self,
        builtin_config_path: str | Path | None = None,
        workspace_mcp_dir: str | Path | None = None,
        *,
        scan_interval: float = 1.0,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]

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

        self.scan_interval = scan_interval

        # Provider name -> live provider.
        self.providers: dict[str, MCPProvider] = {}

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
        await self._scan_workspace()

        self._supervisor_task = asyncio.create_task(
            self._supervisor(),
            name="mcp-runtime-supervisor",
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

    # ==================================================================
    # Agent views
    # ==================================================================

    def create_agent_view(
        self,
        *,
        active_mcp: str | None = None,
    ) -> AgentMCPView:
        """
        Create an independent per-Agent MCP exposure view.

        The underlying MCP sessions remain shared.
        """
        return AgentMCPView(
            runtime=self,
            active_mcp=active_mcp,
        )

    # ==================================================================
    # Provider access
    # ==================================================================

    def provider_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.providers))

    def get_provider(
        self,
        name: str,
    ) -> MCPProvider | None:
        return self.providers.get(name)

    # ==================================================================
    # MCP lifecycle
    # ==================================================================

    async def _connect_provider(
        self,
        spec: MCPProviderSpec,
    ) -> MCPProvider:
        if spec.name in self.providers:
            raise ValueError(
                f"Duplicate MCP provider: {spec.name}"
            )

        command = shutil.which(spec.command) or spec.command

        env = None

        if spec.env is not None:
            env = {
                **os.environ,
                **dict(spec.env),
            }

        server_params = StdioServerParameters(
            command=command,
            args=list(spec.args),
            env=env,
            cwd=spec.cwd,
        )

        logger.info(
            "Starting MCP provider '{}': {} {}",
            spec.name,
            command,
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

            provider = MCPProvider(
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

    async def _replace_provider(
        self,
        spec: MCPProviderSpec,
    ) -> None:
        old = self.providers.get(spec.name)

        # Connect candidate first.
        candidate = await self._connect_provider(spec)

        self.providers[spec.name] = candidate

        if old is not None:
            await self._close_provider(old)

        logger.info(
            "MCP provider '{}' replaced",
            spec.name,
        )

        self._wake.set()

    async def _remove_provider(
        self,
        name: str,
    ) -> None:
        provider = self.providers.pop(name, None)

        if provider is None:
            return

        await self._close_provider(provider)

        logger.info(
            "MCP provider '{}' removed",
            name,
        )

    @staticmethod
    async def _close_provider(
        provider: MCPProvider,
    ) -> None:
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
            return self._error(
                f"Unknown MCP provider: {provider_name}"
            )

        arguments = arguments or {}

        try:
            return await provider.session.call_tool(
                tool_name,
                arguments,
            )

        except Exception as exc:
            logger.exception(
                "MCP tool call failed: %s.%s",
                provider_name,
                tool_name,
            )

            return self._error(str(exc))

    # ==================================================================
    # Builtin configuration
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
                    f"Duplicate builtin MCP provider: {spec.name}"
                )

            provider = await self._connect_provider(
                spec
            )

            self.providers[spec.name] = provider

    @staticmethod
    def _parse_builtin_config(
        config: dict[str, Any],
        source: Path,
    ) -> list[MCPProviderSpec]:
        servers = config.get(
            "mcp_servers",
            {},
        )

        if not isinstance(servers, dict):
            raise ValueError(
                f"'mcp_servers' must be a mapping: {source}"
            )

        specs: list[MCPProviderSpec] = []

        for name, raw in servers.items():
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Invalid MCP config for '{name}'"
                )

            specs.append(
                MCPRuntime._spec_from_mapping(
                    name=str(name),
                    raw=raw,
                    source=str(source),
                    origin="builtin",
                )
            )

        return specs

    # ==================================================================
    # Workspace discovery
    # ==================================================================

    async def _scan_workspace(self) -> None:
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
                        f"cannot override builtin MCP"
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

        candidates: dict[str, MCPProvider] = {}

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
    ) -> list[MCPProviderSpec]:
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

            result: list[MCPProviderSpec] = []

            for name, raw in servers.items():
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"Invalid MCP config for '{name}'"
                    )

                result.append(
                    MCPRuntime._spec_from_mapping(
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
            MCPRuntime._spec_from_mapping(
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
        origin: Literal["builtin", "workspace"],
    ) -> MCPProviderSpec:
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

        return MCPProviderSpec(
            name=name,
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
                    "MCP workspace reconciliation failed"
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

    @staticmethod
    def _text(
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

    @staticmethod
    def _error(
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


class AgentMCPView:
    """
    MCP tool exposure for one Agent.

    The underlying MCPRuntime is shared.

    active_mcp is intentionally local to this object.
    """

    def __init__(
        self,
        runtime: MCPRuntime,
        *,
        active_mcp: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.active_mcp = active_mcp

    def available_mcps(self) -> tuple[str, ...]:
        return self.runtime.provider_names()

    async def list_tools(self) -> list[types.Tool]:
        tools = [
            self._route_tool(),
        ]

        if self.active_mcp is None:
            return tools

        provider = self.runtime.get_provider(
            self.active_mcp
        )

        if provider is None:
            # The provider may have disappeared from the
            # workspace since this view was created.
            self.active_mcp = None
            return tools

        await self.runtime.refresh_provider_tools(
            self.active_mcp
        )

        provider = self.runtime.get_provider(
            self.active_mcp
        )

        if provider is None:
            self.active_mcp = None
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

        if name == "route_mcp":
            return await self._route_mcp(
                arguments.get("mcp_name")
            )

        if self.active_mcp is None:
            return MCPRuntime._error(
                "No MCP server is active. "
                "Call route_mcp first."
            )

        provider = self.runtime.get_provider(
            self.active_mcp
        )

        if provider is None:
            self.active_mcp = None

            return MCPRuntime._error(
                "The active MCP server is no longer available."
            )

        if name not in provider.tools:
            await self.runtime.refresh_provider_tools(
                self.active_mcp
            )

            provider = self.runtime.get_provider(
                self.active_mcp
            )

            if (
                provider is None
                or name not in provider.tools
            ):
                return MCPRuntime._error(
                    f"Tool '{name}' is not available "
                    f"from MCP '{self.active_mcp}'."
                )

        return await self.runtime.call_tool(
            self.active_mcp,
            name,
            arguments,
        )

    async def _route_mcp(
        self,
        mcp_name: str | None,
    ) -> types.CallToolResult:
        if not mcp_name:
            return MCPRuntime._error(
                "mcp_name is required"
            )

        if mcp_name not in self.runtime.providers:
            return MCPRuntime._error(
                f"Unknown MCP server: {mcp_name}. "
                f"Available: "
                f"{', '.join(self.runtime.provider_names())}"
            )

        if self.active_mcp == mcp_name:
            return MCPRuntime._text(
                f"MCP '{mcp_name}' is already active."
            )

        self.active_mcp = mcp_name

        await self.runtime.refresh_provider_tools(
            mcp_name
        )

        provider = self.runtime.get_provider(
            mcp_name
        )

        tool_count = (
            len(provider.tools)
            if provider is not None
            else 0
        )

        return MCPRuntime._text(
            f"MCP '{mcp_name}' activated. "
            f"{tool_count} tools are now available."
        )

    def _route_tool(self) -> types.Tool:
        names = list(
            self.runtime.provider_names()
        )

        return types.Tool(
            name="route_mcp",
            description=(
                "Select which MCP server is currently active. "
                "Only the selected MCP server's tools are exposed "
                "in this Agent tool view. "
                "MCP connections remain alive while routing changes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mcp_name": {
                        "type": "string",
                        "enum": names,
                        "description": (
                            "The MCP server to activate."
                        ),
                    }
                },
                "required": ["mcp_name"],
                "additionalProperties": False,
            },
        )


class MCPFacade:
    """
    Compatibility wrapper around MCPRuntime.

    This preserves the existing MCP stdio-server use case while the
    underlying runtime now supports independent Agent views.
    """

    def __init__(
        self,
        config_path: Path,
        workspace_mcp_dir: str | Path | None = None,
        *,
        scan_interval: float = 1.0,
    ) -> None:
        self.runtime = MCPRuntime(
            builtin_config_path=config_path,
            workspace_mcp_dir=workspace_mcp_dir,
            scan_interval=scan_interval,
        )

        self.server = Server("mcp-facade")
        self.view: AgentMCPView | None = None

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
                return MCPRuntime._error(
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