"""
Shared tool provider runtime.

Responsibilities:
    - load builtin MCP configuration and builtin local classes
    - discover workspace MCP configs and local Python tools
    - maintain provider sessions/instances (hot reload included)
    - cache tool definitions
    - bound every tool call with a timeout

This object does NOT know about Agent exposure: per-Agent tool
visibility lives in AgentToolView, which composes this runtime.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

import mcp.types as types

from src.nan_itself.tools import local as local_backend
from src.nan_itself.tools import mcp as mcp_backend
from src.nan_itself.tools.spec import (
    PROVIDER_KIND_LOCAL,
    DEFAULT_TOOL_TIMEOUT,
    ProviderSpec,
)
from src.nan_itself.tools.provider import (
    Provider,
)
from src.nan_itself.tools.results import (
    error_result,
)
from src.nan_itself.tools.watcher import (
    SourceTracker,
    file_fingerprint,
)

from src.nan_itself.tools.builtin import (
    BUILTIN_MCP_CONFIG,
    BUILTIN_TOOLS,
)


class ProviderRuntime:
    def __init__(
        self,
        builtin_config_path: str | Path | None = None,
        workspace_mcp_dir: str | Path | None = None,
        workspace_local_dir: str | Path | None = None,
        *,
        builtin_tools: (
            tuple[type[local_backend.LocalToolProvider], ...]
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
            else BUILTIN_MCP_CONFIG
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
        # Workspace bookkeeping.
        # ----------------------------------------------------------

        self._mcp_tracker = SourceTracker()

        # Workspace MCP source file -> provider names from it.
        self._mcp_sources: dict[Path, set[str]] = {}

        self._local_tracker = SourceTracker()

        # Workspace local source file -> provider name from it.
        self._local_sources: dict[Path, set[str]] = {}

        # Local source file -> imported module name.
        self._local_imported_names: dict[Path, str] = {}

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

        self._mcp_sources.clear()
        self._local_sources.clear()
        self._local_imported_names.clear()

        self._mcp_tracker = SourceTracker()
        self._local_tracker = SourceTracker()

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

        await mcp_backend.refresh_tools(
            provider
        )

    async def call_tool(
        self,
        provider_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        provider = self.providers.get(provider_name)

        if provider is None:
            return error_result(
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

            return error_result(
                f"Tool '{tool_name}' timed out "
                f"after {self.tool_timeout:g} seconds."
            )

        except Exception as exc:
            logger.exception(
                "Tool call failed: %s.%s",
                provider_name,
                tool_name,
            )

            return error_result(str(exc))

    # ==================================================================
    # Provider lifecycle
    # ==================================================================

    async def _connect_mcp(
        self,
        spec: ProviderSpec,
    ) -> Provider:
        if spec.name in self.providers:
            raise ValueError(
                f"Duplicate tool provider: {spec.name}"
            )

        return await mcp_backend.connect(spec)

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

    # ==================================================================
    # Builtin sources
    # ==================================================================

    async def _load_builtin_config(self) -> None:
        if not self.builtin_config_path.exists():
            logger.warning(
                "Builtin MCP config not found: {}",
                self.builtin_config_path,
            )
            return

        config = mcp_backend.load_yaml(
            self.builtin_config_path
        )

        specs = mcp_backend.parse_builtin_config(
            config,
            self.builtin_config_path,
        )

        for spec in specs:
            if spec.name in self.providers:
                raise ValueError(
                    f"Duplicate builtin tool provider: {spec.name}"
                )

            try:
                provider = await self._connect_mcp(
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

    def _load_builtin_tools(self) -> None:
        for cls in self.builtin_tools:
            local_backend.validate_class(cls)

            if cls.id in self.providers:
                raise ValueError(
                    f"Duplicate builtin tool provider: {cls.id}"
                )

            spec = local_backend.local_provider_spec(
                name=cls.id,
                source="<builtin>",
                origin="builtin",
            )

            self.providers[
                spec.name
            ] = local_backend.build_provider(
                spec,
                cls,
            )

    # ==================================================================
    # Workspace scanning
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

        for path in self._mcp_tracker.known_files() - current_files:
            await self._remove_workspace_source(path)

        for path in sorted(current_files):
            fingerprint = file_fingerprint(path)

            loaded = path in self._mcp_sources

            if not self._mcp_tracker.needs_load(
                path,
                fingerprint,
                loaded=loaded,
            ):
                continue

            self._mcp_tracker.mark_seen(path, fingerprint)

            try:
                await self._reload_workspace_source(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._mcp_tracker.mark_failed(path, exc)

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
            and local_backend.has_tool_header(path)
        }

        for path in self._local_tracker.known_files() - current_files:
            await self._remove_local_source(path)

        for path in sorted(current_files):
            fingerprint = file_fingerprint(path)

            loaded = path in self._local_sources

            if not self._local_tracker.needs_load(
                path,
                fingerprint,
                loaded=loaded,
            ):
                continue

            self._local_tracker.mark_seen(path, fingerprint)

            try:
                await self._reload_local_source(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._local_tracker.mark_failed(path, exc)

                logger.exception(
                    "Failed to load workspace local tool: {}",
                    path,
                )

    # ------------------------------------------------------------------
    # Workspace MCP reload
    # ------------------------------------------------------------------

    async def _reload_workspace_source(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        config = mcp_backend.load_yaml(path)

        specs = mcp_backend.parse_workspace_config(
            config,
            path,
        )

        names = [spec.name for spec in specs]

        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate MCP provider name in {path}"
            )

        # Workspace providers may not shadow builtin providers,
        # nor steal names owned by a different source file.
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
                candidate = await self._connect_mcp(
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

        old_names = self._mcp_sources.get(
            path.resolve(),
            set(),
        )

        self._mcp_sources[path.resolve()] = set(
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

        names = self._mcp_sources.pop(
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

        self._mcp_tracker.forget(path)

        logger.info(
            "Removed workspace MCP source '{}'",
            path,
        )

    # ------------------------------------------------------------------
    # Workspace local reload
    # ------------------------------------------------------------------

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
        ) = local_backend.load_class_from_file(path)

        local_backend.validate_class(cls)

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

        candidate_spec = local_backend.local_provider_spec(
            name=provider_name,
            file=path,
            source=str(path),
            origin="workspace",
        )

        # Build the candidate first; failure leaves the
        # previously loaded source untouched.
        candidate = local_backend.build_provider(
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

        self._local_tracker.forget(path)

        logger.info(
            "Removed workspace local tool source '{}'",
            path,
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


__all__ = [
    "BUILTIN_TOOLS",
    "Provider",
    "ProviderRuntime",
    "ProviderSpec",
]
