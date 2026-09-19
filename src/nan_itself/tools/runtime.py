"""
Shared tool provider runtime.

Responsibilities:
    - discover builtin and workspace MCP configs and local
      Python tools (both are hot-reloadable; deleting a source
      file disables its provider)
    - maintain provider sessions/instances (hot reload included)
    - resolve tools by 'provider/tool' composite name
    - bound every tool call with a timeout

MCP lifecycle model:

    ProviderRuntime
        |
        +-- supervisor task
        |       |
        |       +-- scans/reconciles configuration
        |       +-- starts/stops MCP workers
        |
        +-- MCP worker: files
        |       |
        |       +-- owns stdio_client
        |       +-- owns ClientSession
        |       +-- owns AsyncExitStack
        |
        +-- MCP worker: playwright
        |
        +-- MCP worker: command
        |
        +-- local providers
                |
                +-- in-process instances

IMPORTANT:

Each MCP server owns its own asyncio task.

That task is responsible for:

    connect -> run -> close

This prevents multiple mcp.client.stdio AnyIO cancel scopes from
being nested inside one task and later being torn down out of order.

HOT RELOAD IMPORTANT:

A replacement MCP worker is first created as an unregistered candidate:

    old worker: active mapping
    new worker: candidate only

Only after the candidate is ready is it committed into the live
provider / worker mappings. The old worker is then asked to stop.

This allows old and new generations with the same provider name to
coexist during the transaction.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import mcp.types as types
from loguru import logger

from nan_itself.utils import paths as _paths

from . import local as local_backend
from . import mcp as mcp_backend
from .provider import Provider
from .results import error_result
from .spec import (
    DEFAULT_TOOL_TIMEOUT,
    PROVIDER_KIND_LOCAL,
    ProviderSpec,
)
from .watcher import (
    SourceTracker,
    file_fingerprint,
)


class _MCPWorker:
    """
    Lifecycle owner for exactly one MCP provider.

    The worker task owns the MCP AsyncExitStack. No other task should
    ever call stack.aclose().

    The worker also tracks in-flight calls so shutdown can stop
    accepting new calls and wait for current calls to drain before
    exiting.
    """

    def __init__(
        self,
        spec: ProviderSpec,
    ) -> None:
        self.spec = spec

        self.stop_event = (
            asyncio.Event()
        )

        loop = asyncio.get_running_loop()

        self.ready: asyncio.Future[Provider] = (
            loop.create_future()
        )

        self.task: asyncio.Task[None] | None = None

        self.provider: Provider | None = None

        self.stopping = False

        self.active_calls = 0

        self._condition = asyncio.Condition()

    async def acquire(
        self,
    ) -> Provider:
        """
        Reserve the provider for one active operation.

        Once shutdown begins, new operations are rejected.
        """
        async with self._condition:
            if self.stopping:
                raise RuntimeError(
                    f"MCP provider '{self.spec.name}' "
                    f"is stopping."
                )

            provider = self.provider

            if provider is None:
                raise RuntimeError(
                    f"MCP provider '{self.spec.name}' "
                    f"is not ready."
                )

            self.active_calls += 1

            return provider

    async def release(
        self,
    ) -> None:
        """
        Release an active operation.
        """
        async with self._condition:
            if self.active_calls > 0:
                self.active_calls -= 1

            if self.active_calls == 0:
                self._condition.notify_all()

    async def begin_stop(
        self,
    ) -> None:
        """
        Prevent new calls and tell the worker to exit normally.
        """
        async with self._condition:
            self.stopping = True

        self.stop_event.set()

    async def wait_calls_drained(
        self,
    ) -> None:
        """
        Wait until all in-flight tool calls have completed.
        """
        async with self._condition:
            await self._condition.wait_for(
                lambda: self.active_calls == 0
            )


class ProviderRuntime:
    def __init__(
        self,
        builtin_tools_dir: str | Path | None = None,
        workspace_mcp_dir: str | Path | None = None,
        workspace_local_dir: str | Path | None = None,
        *,
        scan_interval: float = 1.0,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT,
        mcp_start_timeout: float = 60.0,
    ) -> None:
        project_root = _paths.repo_root()

        self.tool_timeout = tool_timeout
        self.mcp_start_timeout = (
            mcp_start_timeout
        )

        builtin_root = (
            Path(
                builtin_tools_dir
            ).resolve()
            if builtin_tools_dir is not None
            else (
                project_root
                / "builtin"
                / "tools"
            ).resolve()
        )

        # Builtin and workspace sources live in parallel directory
        # layouts and are scanned with exactly the same hot-reload
        # logic. Deleting a source file disables its provider.
        self.builtin_mcps_dir = (
            builtin_root / "mcps"
        )

        self.builtin_locals_dir = (
            builtin_root / "local"
        )

        self.workspace_mcp_dir = (
            Path(
                workspace_mcp_dir
            ).resolve()
            if workspace_mcp_dir is not None
            else (
                project_root
                / "workspace"
                / "tools"
                / "mcps"
            ).resolve()
        )

        self.workspace_local_dir = (
            Path(
                workspace_local_dir
            ).resolve()
            if workspace_local_dir is not None
            else (
                project_root
                / "workspace"
                / "tools"
                / "local"
            ).resolve()
        )

        self.scan_interval = (
            scan_interval
        )

        # ----------------------------------------------------------
        # Live providers.
        # ----------------------------------------------------------

        # Provider name -> live provider.
        #
        # Candidate MCP workers are deliberately NOT placed here
        # until their reload transaction commits.
        self.providers: dict[
            str,
            Provider,
        ] = {}

        # Provider name -> MCP worker.
        #
        # Local providers do not have workers.
        #
        # Candidate MCP workers are deliberately NOT placed here
        # until their reload transaction commits.
        self._mcp_workers: dict[
            str,
            _MCPWorker,
        ] = {}

        # ----------------------------------------------------------
        # Source bookkeeping (builtin + workspace).
        # ----------------------------------------------------------

        self._mcp_tracker = SourceTracker()

        # MCP source file -> provider names from it.
        self._mcp_sources: dict[
            Path,
            set[str],
        ] = {}

        self._local_tracker = SourceTracker()

        # Local source file -> provider name from it.
        self._local_sources: dict[
            Path,
            set[str],
        ] = {}

        # Local source file -> imported module name.
        self._local_imported_names: dict[
            Path,
            str,
        ] = {}

        # ----------------------------------------------------------
        # Supervisor lifecycle.
        # ----------------------------------------------------------

        self._supervisor_task: (
            asyncio.Task[None] | None
        ) = None

        # start() waits until the supervisor finishes initial loading.
        self._startup_future: (
            asyncio.Future[None] | None
        ) = None

        self._wake = asyncio.Event()

        self._stopping = False

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(
        self,
    ) -> None:
        """
        Start the runtime.

        A dedicated supervisor task owns reconciliation.
        Every MCP server gets its own dedicated worker task.
        """
        if self._supervisor_task is not None:
            return

        self._stopping = False
        self._wake.clear()

        loop = asyncio.get_running_loop()

        self._startup_future = (
            loop.create_future()
        )

        self._supervisor_task = (
            asyncio.create_task(
                self._supervisor(),
                name=(
                    "provider-runtime-supervisor"
                ),
            )
        )

        try:
            await self._startup_future

        except BaseException:
            self._stopping = True
            self._wake.set()

            supervisor = (
                self._supervisor_task
            )

            if supervisor is not None:
                try:
                    await supervisor
                except BaseException:
                    pass

            self._supervisor_task = None
            self._startup_future = None

            raise

    async def stop(
        self,
    ) -> None:
        """
        Stop the runtime cleanly.

        IMPORTANT:

        Do not cancel MCP worker tasks directly.

        Instead, the supervisor is asked to stop. Its finally block
        signals each worker's stop event and waits for each worker task
        to exit. The worker itself then closes its own MCP context.
        """
        supervisor = (
            self._supervisor_task
        )

        if supervisor is None:
            return

        self._stopping = True

        self._wake.set()

        self._supervisor_task = None

        try:
            await supervisor

        except asyncio.CancelledError:
            logger.warning(
                "Provider runtime supervisor was cancelled "
                "during shutdown"
            )

        except Exception:
            logger.exception(
                "Provider runtime supervisor failed during shutdown"
            )

        finally:
            self._startup_future = None

            self._mcp_sources.clear()
            self._local_sources.clear()
            self._local_imported_names.clear()

            self._mcp_tracker = SourceTracker()
            self._local_tracker = SourceTracker()

            self.providers.clear()
            self._mcp_workers.clear()

    # ==================================================================
    # Provider access
    # ==================================================================

    def provider_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.providers
            )
        )

    def get_provider(
        self,
        name: str,
    ) -> Provider | None:
        return self.providers.get(
            name
        )

    def attach_provider(
        self,
        provider: Provider,
    ) -> None:
        """
        Register an externally-owned live provider.

        The caller owns the full lifecycle; the runtime neither
        starts nor stops it. Used e.g. by the modules runtime to
        expose module action faces as ordinary providers named
        `module:<id>`.
        """
        self.providers[
            provider.spec.name
        ] = provider

    def detach_provider(
        self,
        name: str,
    ) -> None:
        """
        Remove an externally-owned provider.
        """
        self.providers.pop(
            name,
            None,
        )

    def list_all_tools(
        self,
    ) -> list[
        tuple[
            str,
            types.Tool,
        ]
    ]:
        """
        Every tool of every provider as
        (provider_name, tool) pairs, in stable order.
        """
        pairs: list[
            tuple[
                str,
                types.Tool,
            ]
        ] = []

        for provider_name in sorted(
            self.providers
        ):
            provider = self.providers[
                provider_name
            ]

            for tool_name in sorted(
                provider.tools
            ):
                pairs.append(
                    (
                        provider_name,
                        provider.tools[
                            tool_name
                        ],
                    )
                )

        return pairs

    async def resolve_tool(
        self,
        name: str,
    ) -> tuple[
        Provider,
        types.Tool,
    ] | None:
        """
        Resolve a 'provider/tool' composite name.

        MCP tool tables are refreshed once on a miss, so a tool
        added to a live server becomes visible without restart.
        """
        provider_name, _, tool_name = (
            name.partition("/")
        )

        if not tool_name:
            return None

        provider = self.providers.get(
            provider_name
        )

        if provider is None:
            return None

        tool = provider.tools.get(
            tool_name
        )

        if (
            tool is None
            and provider.spec.kind
            != PROVIDER_KIND_LOCAL
        ):
            await self.refresh_provider_tools(
                provider_name
            )

            provider = self.providers.get(
                provider_name
            )

            if provider is None:
                return None

            tool = provider.tools.get(
                tool_name
            )

        if tool is None:
            return None

        return (
            provider,
            tool,
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
            return

        worker = (
            self._mcp_workers.get(
                name
            )
        )

        if worker is None:
            raise RuntimeError(
                f"MCP provider '{name}' "
                f"is not running."
            )

        reserved = await worker.acquire()

        try:
            if reserved is not provider:
                raise RuntimeError(
                    f"MCP provider '{name}' changed while "
                    f"refreshing tools."
                )

            await mcp_backend.refresh_tools(
                provider
            )

        finally:
            await worker.release()

    async def call_tool(
        self,
        provider_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        provider = self.providers.get(
            provider_name
        )

        if provider is None:
            return error_result(
                f"Unknown tool provider: "
                f"{provider_name}"
            )

        arguments = arguments or {}

        try:
            if (
                provider.spec.kind
                == PROVIDER_KIND_LOCAL
            ):
                return await asyncio.wait_for(
                    provider.call_tool(
                        tool_name,
                        arguments,
                    ),
                    timeout=self.tool_timeout,
                )

            worker = (
                self._mcp_workers.get(
                    provider_name
                )
            )

            if worker is None:
                return error_result(
                    f"MCP provider '{provider_name}' "
                    f"is not running."
                )

            reserved = await worker.acquire()

            try:
                if reserved is not provider:
                    return error_result(
                        f"MCP provider '{provider_name}' "
                        f"was replaced during the tool call."
                    )

                return await asyncio.wait_for(
                    provider.call_tool(
                        tool_name,
                        arguments,
                    ),
                    timeout=self.tool_timeout,
                )

            finally:
                await worker.release()

        except asyncio.TimeoutError:
            logger.warning(
                "Tool call timed out: %s.%s (%gs)",
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

            return error_result(
                str(exc)
            )

    # ==================================================================
    # MCP worker lifecycle
    # ==================================================================

    async def _start_mcp_worker(
        self,
        spec: ProviderSpec,
        *,
        register: bool = True,
    ) -> _MCPWorker:
        """
        Start exactly one worker for one MCP provider.

        By default the worker is installed into the live runtime.

        During hot reload:

            register=False

        creates a candidate worker that owns its own MCP connection
        but is NOT visible through self.providers or
        self._mcp_workers until the transaction commits.

        This allows old and new generations with the same provider
        name to coexist safely during replacement.
        """
        if (
            register
            and spec.name in self._mcp_workers
        ):
            raise ValueError(
                f"Duplicate MCP worker: "
                f"{spec.name}"
            )

        worker = _MCPWorker(
            spec
        )

        worker.task = asyncio.create_task(
            self._run_mcp_worker(
                worker
            ),
            name=(
                f"mcp-worker:"
                f"{spec.name}"
            ),
        )

        try:
            provider = await asyncio.wait_for(
                asyncio.shield(
                    worker.ready
                ),
                timeout=self.mcp_start_timeout,
            )

        except BaseException:
            await self._stop_mcp_worker(
                worker
            )
            raise

        worker.provider = provider

        if register:
            # The worker is fully ready before it becomes visible
            # to the rest of the runtime.
            self._mcp_workers[
                spec.name
            ] = worker

            self.providers[
                spec.name
            ] = provider

            logger.info(
                "MCP worker '{}' started",
                spec.name,
            )

        else:
            logger.info(
                "MCP worker '{}' candidate started",
                spec.name,
            )

        return worker

    async def _run_mcp_worker(
        self,
        worker: _MCPWorker,
    ) -> None:
        """
        Own one MCP connection from creation to destruction.

        This function MUST keep the MCP AsyncExitStack inside this
        worker task for its entire lifetime.
        """
        spec = worker.spec
        provider: Provider | None = None

        try:
            logger.info(
                "Connecting MCP worker '{}'",
                spec.name,
            )

            provider = await mcp_backend.connect(
                spec
            )

            worker.provider = provider

            if not worker.ready.done():
                worker.ready.set_result(
                    provider
                )

            # Stay alive until this worker is asked to stop.
            await worker.stop_event.wait()

        except BaseException as exc:
            if not worker.ready.done():
                worker.ready.set_exception(
                    exc
                )

            if not isinstance(
                exc,
                asyncio.CancelledError,
            ):
                logger.exception(
                    "MCP worker '{}' exited unexpectedly",
                    spec.name,
                )

            raise

        finally:
            # No new calls are allowed from this point.
            async with worker._condition:
                worker.stopping = True

            # Wait for already-running tool calls to leave.
            try:
                await worker.wait_calls_drained()

            except BaseException:
                logger.exception(
                    "Failed while draining calls for "
                    "MCP worker '{}'",
                    spec.name,
                )

            # ------------------------------------------------------
            # CRITICAL:
            #
            # The worker task itself closes its own AsyncExitStack.
            # ------------------------------------------------------

            if provider is not None:
                stack = provider.stack

                provider.stack = None

                if stack is not None:
                    try:
                        logger.info(
                            "Closing MCP worker '{}'",
                            spec.name,
                        )

                        await stack.aclose()

                    except BaseException:
                        logger.exception(
                            "Failed to close MCP worker '{}'",
                            spec.name,
                        )

                provider.session = None

            # ------------------------------------------------------
            # Only remove mappings if THIS worker still owns them.
            #
            # This is important during hot reload:
            #
            # old worker A
            #     ↓
            # new worker A installed
            #     ↓
            # old worker exits
            #
            # old worker must not delete new worker A.
            #
            # Candidate workers that were never committed do not
            # appear in these mappings, so this is naturally safe.
            # ------------------------------------------------------

            current_worker = (
                self._mcp_workers.get(
                    spec.name
                )
            )

            if current_worker is worker:
                self._mcp_workers.pop(
                    spec.name,
                    None,
                )

            current_provider = (
                self.providers.get(
                    spec.name
                )
            )

            if (
                provider is not None
                and current_provider is provider
            ):
                self.providers.pop(
                    spec.name,
                    None,
                )

            logger.info(
                "MCP worker '{}' stopped",
                spec.name,
            )

    async def _stop_mcp_worker(
        self,
        worker: _MCPWorker,
    ) -> None:
        """
        Ask one worker to shut itself down.

        We do not directly close its stack from this task.
        """
        await worker.begin_stop()

        task = worker.task

        if task is None:
            return

        try:
            await task

        except asyncio.CancelledError:
            logger.warning(
                "MCP worker '{}' was cancelled",
                worker.spec.name,
            )

        except Exception:
            logger.debug(
                "MCP worker '{}' finished with an exception",
                worker.spec.name,
            )

    async def _stop_mcp_workers(
        self,
        workers: list[_MCPWorker],
    ) -> None:
        """
        Stop multiple independent workers concurrently.

        Every worker closes its own stack inside its own task.
        """
        if not workers:
            return

        for worker in workers:
            await worker.begin_stop()

        await asyncio.gather(
            *(
                self._await_mcp_worker(
                    worker
                )
                for worker in workers
            ),
            return_exceptions=True,
        )

    async def _await_mcp_worker(
        self,
        worker: _MCPWorker,
    ) -> None:
        task = worker.task

        if task is None:
            return

        try:
            await task

        except asyncio.CancelledError:
            logger.warning(
                "MCP worker '{}' was cancelled",
                worker.spec.name,
            )

        except Exception:
            logger.exception(
                "MCP worker '{}' failed during shutdown",
                worker.spec.name,
            )

    async def _stop_all_mcp_workers(
        self,
    ) -> None:
        workers = list(
            self._mcp_workers.values()
        )

        if not workers:
            return

        for worker in workers:
            await worker.begin_stop()

        await asyncio.gather(
            *(
                self._await_mcp_worker(
                    worker
                )
                for worker in workers
            ),
            return_exceptions=True,
        )

        self._mcp_workers.clear()

        for name, provider in list(
            self.providers.items()
        ):
            if (
                provider.spec.kind
                == PROVIDER_KIND_LOCAL
            ):
                self.providers.pop(
                    name,
                    None,
                )

    async def _reap_dead_workers(
        self,
    ) -> None:
        """
        Detect unexpectedly terminated current MCP workers.

        The worker's source file is marked for reload by removing
        its source bookkeeping, so the next scan reconnects it —
        or removes it cleanly when the file is gone. This policy
        is the same for builtin and workspace sources.
        """
        dead: list[
            tuple[str, _MCPWorker]
        ] = []

        for name, worker in list(
            self._mcp_workers.items()
        ):
            task = worker.task

            if task is None:
                continue

            if not task.done():
                continue

            if worker.stopping:
                continue

            dead.append(
                (
                    name,
                    worker,
                )
            )

        for name, worker in dead:
            task = worker.task

            if task is not None:
                try:
                    task.result()

                except asyncio.CancelledError:
                    pass

                except Exception:
                    logger.exception(
                        "MCP worker '{}' crashed",
                        name,
                    )

            current_worker = (
                self._mcp_workers.get(
                    name
                )
            )

            if current_worker is not worker:
                continue

            self._mcp_workers.pop(
                name,
                None,
            )

            provider = (
                self.providers.get(
                    name
                )
            )

            if (
                provider is not None
                and provider is worker.provider
            ):
                self.providers.pop(
                    name,
                    None,
                )

            source = Path(
                worker.spec.source
            ).resolve()

            self._mcp_sources.pop(
                source,
                None,
            )

            self._mcp_tracker.forget(
                source
            )

            logger.warning(
                "MCP worker '{}' died; source '{}' "
                "will be reloaded",
                name,
                source,
            )

    # ==================================================================
    # Source scanning (builtin + workspace, identical logic)
    # ==================================================================

    async def _scan_sources(
        self,
    ) -> None:
        await self._scan_mcps()
        await self._scan_locals()

    async def _scan_mcps(
        self,
    ) -> None:
        await self._scan_mcp_root(
            self.builtin_mcps_dir
        )

        await self._scan_mcp_root(
            self.workspace_mcp_dir
        )

    async def _scan_mcp_root(
        self,
        root: Path,
    ) -> None:
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_files = {
            path.resolve()
            for path in root.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in {".yaml", ".yml"}
                and not path.name.startswith("_")
            )
        }

        removed = {
            path
            for path in self._mcp_tracker.known_files()
            if path.is_relative_to(
                root
            )
        } - current_files

        for path in removed:
            await self._remove_mcp_source(
                path
            )

        for path in sorted(
            current_files
        ):
            fingerprint = (
                file_fingerprint(path)
            )

            loaded = (
                path in self._mcp_sources
            )

            if not self._mcp_tracker.needs_load(
                path,
                fingerprint,
                loaded=loaded,
            ):
                continue

            self._mcp_tracker.mark_seen(
                path,
                fingerprint,
            )

            try:
                await self._reload_mcp_source(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._mcp_tracker.mark_failed(
                    path,
                    exc,
                )

                logger.exception(
                    "Failed to load MCP config: {}",
                    path,
                )

    async def _scan_locals(
        self,
    ) -> None:
        await self._scan_local_root(
            self.builtin_locals_dir
        )

        await self._scan_local_root(
            self.workspace_local_dir
        )

    async def _scan_local_root(
        self,
        root: Path,
    ) -> None:
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_files = {
            path.resolve()
            for path in root.rglob(
                "*.py"
            )
            if (
                path.is_file()
                and not path.name.startswith("_")
                and local_backend.has_tool_header(
                    path
                )
            )
        }

        removed = {
            path
            for path in self._local_tracker.known_files()
            if path.is_relative_to(
                root
            )
        } - current_files

        for path in removed:
            await self._remove_local_source(
                path
            )

        for path in sorted(
            current_files
        ):
            fingerprint = (
                file_fingerprint(path)
            )

            loaded = (
                path in self._local_sources
            )

            if not self._local_tracker.needs_load(
                path,
                fingerprint,
                loaded=loaded,
            ):
                continue

            self._local_tracker.mark_seen(
                path,
                fingerprint,
            )

            try:
                await self._reload_local_source(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._local_tracker.mark_failed(
                    path,
                    exc,
                )

                logger.exception(
                    "Failed to load local tool: {}",
                    path,
                )

    # ------------------------------------------------------------------
    # MCP source reload
    # ------------------------------------------------------------------

    async def _reload_mcp_source(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        path = path.resolve()

        config = mcp_backend.load_yaml(
            path
        )

        specs = (
            mcp_backend.parse_config(
                config,
                path,
            )
        )

        names = [
            spec.name
            for spec in specs
        ]

        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate MCP provider name in {path}"
            )

        # --------------------------------------------------------------
        # Validate ownership before starting candidates:
        # one provider name belongs to exactly one source file.
        # --------------------------------------------------------------

        for spec in specs:
            existing = (
                self.providers.get(
                    spec.name
                )
            )

            if existing is None:
                continue

            source = Path(
                existing.spec.source
            ).resolve()

            if source != path:
                raise ValueError(
                    f"MCP provider '{spec.name}' "
                    f"is already provided by {source}"
                )

        # --------------------------------------------------------------
        # Connect candidates.
        #
        # IMPORTANT:
        #
        # Candidate workers are started with register=False.
        # This permits:
        #
        #     old worker: example
        #     candidate:  example
        #
        # to coexist before commit.
        # --------------------------------------------------------------

        candidates: dict[
            str,
            _MCPWorker,
        ] = {}

        try:
            for spec in specs:
                worker = (
                    await self._start_mcp_worker(
                        spec,
                        register=False,
                    )
                )

                candidates[
                    spec.name
                ] = worker

        except BaseException:
            # Candidates are independent workers. Stop only those
            # created by this failed transaction.
            await self._stop_mcp_workers(
                list(
                    candidates.values()
                )
            )

            raise

        # --------------------------------------------------------------
        # Commit source replacement.
        # --------------------------------------------------------------

        old_names = self._mcp_sources.get(
            path,
            set(),
        )

        new_names = set(
            candidates
        )

        old_workers_to_stop: list[
            _MCPWorker
        ] = []

        # --------------------------------------------------------------
        # Remove stale names from this source.
        # --------------------------------------------------------------

        for name in (
            old_names - new_names
        ):
            old_worker = (
                self._mcp_workers.get(
                    name
                )
            )

            if (
                old_worker is not None
                and Path(
                    old_worker.spec.source
                ).resolve()
                == path
            ):
                self._mcp_workers.pop(
                    name,
                    None,
                )

                current_provider = (
                    self.providers.get(
                        name
                    )
                )

                if (
                    current_provider is not None
                    and current_provider
                    is old_worker.provider
                ):
                    self.providers.pop(
                        name,
                        None,
                    )

                old_workers_to_stop.append(
                    old_worker
                )

        # --------------------------------------------------------------
        # Install candidates FIRST, then stop old workers.
        #
        # This is the generation handoff point:
        #
        #     old provider
        #          ↓
        #     candidate provider
        #
        # Existing views remember only provider name, so they naturally
        # see the new generation after this point.
        # --------------------------------------------------------------

        for name, candidate in (
            candidates.items()
        ):
            old_worker = (
                self._mcp_workers.get(
                    name
                )
            )

            if (
                old_worker is not None
                and old_worker is not candidate
            ):
                self._mcp_workers.pop(
                    name,
                    None,
                )

                old_provider = (
                    self.providers.get(
                        name
                    )
                )

                if (
                    old_provider is not None
                    and old_provider
                    is old_worker.provider
                ):
                    self.providers.pop(
                        name,
                        None,
                    )

                old_workers_to_stop.append(
                    old_worker
                )

            self._mcp_workers[
                name
            ] = candidate

            provider = (
                candidate.provider
            )

            if provider is None:
                raise RuntimeError(
                    f"MCP worker '{name}' "
                    f"became ready without a provider."
                )

            self.providers[
                name
            ] = provider

        self._mcp_sources[
            path
        ] = new_names

        # --------------------------------------------------------------
        # Stop replaced/stale workers concurrently.
        #
        # Old workers close their own stack in their own tasks.
        # --------------------------------------------------------------

        await self._stop_mcp_workers(
            old_workers_to_stop
        )

        logger.info(
            "Loaded MCP source '{}': {}",
            path,
            sorted(candidates),
        )

    async def _remove_mcp_source(
        self,
        path: Path,
    ) -> None:
        path = path.resolve()

        names = self._mcp_sources.pop(
            path,
            set(),
        )

        workers: list[
            _MCPWorker
        ] = []

        for name in names:
            worker = (
                self._mcp_workers.get(
                    name
                )
            )

            if worker is None:
                continue

            if Path(
                worker.spec.source
            ).resolve() != path:
                continue

            self._mcp_workers.pop(
                name,
                None,
            )

            provider = (
                self.providers.get(
                    name
                )
            )

            if (
                provider is not None
                and provider
                is worker.provider
            ):
                self.providers.pop(
                    name,
                    None,
                )

            workers.append(
                worker
            )

        await self._stop_mcp_workers(
            workers
        )

        self._mcp_tracker.forget(
            path
        )

        logger.info(
            "Removed MCP source '{}'",
            path,
        )

    # ------------------------------------------------------------------
    # Local source reload
    # ------------------------------------------------------------------

    async def _reload_local_source(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        path = path.resolve()

        old_names = (
            self._local_sources.get(
                path,
                set(),
            )
        )

        (
            cls,
            imported_name,
        ) = local_backend.load_class_from_file(
            path
        )

        local_backend.validate_class(
            cls
        )

        provider_name = cls.id

        existing = (
            self.providers.get(
                provider_name
            )
        )

        if existing is not None:
            source = Path(
                existing.spec.source
            ).resolve()

            if source != path:
                raise ValueError(
                    f"Local tool "
                    f"'{provider_name}' is already "
                    f"provided by {source}"
                )

        candidate_spec = (
            local_backend.local_provider_spec(
                name=provider_name,
                file=path,
                source=str(path),
            )
        )

        candidate = (
            local_backend.build_provider(
                candidate_spec,
                cls,
            )
        )

        previous_imported = (
            self._local_imported_names.get(
                path
            )
        )

        if (
            previous_imported is not None
            and previous_imported
            != imported_name
        ):
            sys.modules.pop(
                previous_imported,
                None,
            )

        for name in (
            old_names - {provider_name}
        ):
            stale = (
                self.providers.pop(
                    name,
                    None,
                )
            )

            _ = stale

        old = (
            self.providers.get(
                provider_name
            )
        )

        self.providers[
            provider_name
        ] = candidate

        _ = old

        self._local_sources[
            path
        ] = {
            provider_name,
        }

        self._local_imported_names[
            path
        ] = imported_name

        logger.info(
            "Loaded local tool source '{}': {}",
            path,
            provider_name,
        )

    async def _remove_local_source(
        self,
        path: Path,
    ) -> None:
        path = path.resolve()

        names = (
            self._local_sources.pop(
                path,
                set(),
            )
        )

        for name in names:
            provider = (
                self.providers.get(
                    name
                )
            )

            if provider is None:
                continue

            if Path(
                provider.spec.source
            ).resolve() != path:
                continue

            self.providers.pop(
                name,
                None,
            )

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

        self._local_tracker.forget(
            path
        )

        logger.info(
            "Removed local tool source '{}'",
            path,
        )

    # ==================================================================
    # Supervisor
    # ==================================================================

    async def _supervisor(
        self,
    ) -> None:
        """
        Reconciliation loop.

        The supervisor never directly closes an MCP AsyncExitStack.
        It only tells worker tasks when they should stop.
        """
        try:
            try:
                await self._scan_sources()

            except Exception:
                # A degraded start (e.g. one MCP provider whose
                # dependencies cannot be fetched yet) must not
                # kill the runtime. The supervisor loop keeps
                # reconciling and failed sources are retried
                # with backoff.
                logger.exception(
                    "Initial provider source scan failed"
                )

            startup_future = (
                self._startup_future
            )

            if (
                startup_future is not None
                and not startup_future.done()
            ):
                startup_future.set_result(
                    None
                )

            while not self._stopping:
                try:
                    await self._reap_dead_workers()

                    await self._scan_sources()

                except Exception:
                    logger.exception(
                        "Provider source reconciliation failed"
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

        except BaseException as exc:
            startup_future = (
                self._startup_future
            )

            if (
                startup_future is not None
                and not startup_future.done()
            ):
                startup_future.set_exception(
                    exc
                )

            raise

        finally:
            await self._stop_all_mcp_workers()

            self.providers.clear()

            self._mcp_workers.clear()

            self._mcp_sources.clear()
            self._local_sources.clear()
            self._local_imported_names.clear()

            self._mcp_tracker = SourceTracker()
            self._local_tracker = SourceTracker()


__all__ = [
    "Provider",
    "ProviderRuntime",
    "ProviderSpec",
]