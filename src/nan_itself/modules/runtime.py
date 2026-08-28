from __future__ import annotations

import asyncio
import sys
import time
from loguru import logger
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping



from ..modules import (
    deps as _deps,
    loading as _loading,
    persistence as _persistence,
)
from .reload import (
    hot_reload,
)
from .model import (  # noqa: F401
    DataSpace,
    DuplicateModuleError,
    Module,
    ModuleRecord,
    ModuleState,
    ModuleTurn,
)

from .builtin import (
    BUILTIN_MODULES,
)


# ============================================================================
# Facade
# ============================================================================


class Facade:
    """
    Runtime supervisor for Modules.

    Responsibilities:
        - builtin Module registration
        - workspace Module discovery
        - Python Module loading
        - dependency graph
        - lifecycle supervision
        - retry
        - hot reload
        - DataSpace ownership
        - persistence
        - Agent-turn DataSpace snapshots
        - Module querying
    """

    # ------------------------------------------------------------------
    # Turn delivery
    # ------------------------------------------------------------------

    _DELIVERY_TASKS: set = set()

    def deliver_turn(
        self,
        record,
    ) -> None:
        """
        Broadcast one completed TurnRecord to every Module.

        Each handler runs in its own task: a slow or failing
        module can never delay its peers, and never delays the
        agents either (this call returns immediately).
        """

        for module_record in list(
            self.modules.values()
        ):

            task = asyncio.create_task(
                self._safe_on_turn(
                    module_record.instance,
                    record,
                )
            )

            self._DELIVERY_TASKS.add(task)

            task.add_done_callback(
                self._DELIVERY_TASKS.discard
            )

    @staticmethod
    async def _safe_on_turn(
        instance,
        record,
    ) -> None:
        try:
            await instance.on_turn(record)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                f"Module on_turn failed: "
                f"{getattr(instance, 'id', '?')}"
            )

    def __init__(
        self,
        workspace_modules: str | Path | None = None,
        *,
        builtin_modules: (
            tuple[type[Module], ...] | None
        ) = None,
        data_dir: str | Path = "./data/modules",
        retry_interval: float = 1.0,
        scan_interval: float = 1.0,
        llm=None,
    ) -> None:

        # Provisioned to every Module instance as `module.llm`
        # right after construction; modules that do not need an
        # LLM simply ignore it.
        self.llm = llm
        project_root = (
            Path(__file__)
            .resolve()
            .parents[3]
        )

        self.workspace_modules = (
            Path(
                workspace_modules
            ).resolve()
            if workspace_modules is not None
            else (
                project_root
                / "workspace"
                / "modules"
            ).resolve()
        )

        self.data_dir = (
            Path(data_dir).resolve()
        )

        self.private_dir = (
            self.data_dir / "private"
        )

        self.dataspace_dir = (
            self.data_dir / "dataspace"
        )

        self.builtin_modules = (
            BUILTIN_MODULES
            if builtin_modules is None
            else tuple(builtin_modules)
        )

        self.retry_interval = (
            retry_interval
        )

        self.scan_interval = (
            scan_interval
        )

        self.modules: dict[
            str,
            ModuleRecord,
        ] = {}

        self.dataspaces: dict[
            str,
            DataSpace,
        ] = {}

        # module_id -> required module_ids
        self.dependencies: dict[
            str,
            set[str],
        ] = {}

        # module_id -> dependent module_ids
        self.dependents: dict[
            str,
            set[str],
        ] = {}

        # workspace file -> latest fingerprint
        self._workspace_fingerprints: dict[
            Path,
            tuple[int, int],
        ] = {}

        # workspace file -> last load error for fingerprint
        self._workspace_load_errors: dict[
            Path,
            BaseException,
        ] = {}

        self._supervisor_task: (
            asyncio.Task[None] | None
        ) = None

        self._wake = asyncio.Event()

        self._stopping = False

    # ==================================================================
    # Public lifecycle
    # ==================================================================

    async def start(self) -> None:
        if self._supervisor_task is not None:
            return

        self._stopping = False

        self._ensure_data_dirs()

        # Builtins are already concrete classes.
        self._load_builtins()

        # Workspace classes must be discovered by Facade itself.
        await self._scan_workspace_modules()

        self._rebuild_dependency_graph()
        self._validate_dependency_graph()

        self._supervisor_task = asyncio.create_task(
            self._supervisor(),
            name="module-facade-supervisor",
        )

        await self._reconcile()

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

        # Stop dependents before dependencies.
        records = self.modules.copy()

        for module_id in reversed(
            self._safe_topological_order()
        ):
            record = records.get(module_id)

            if record is not None:
                await self._stop_record(
                    record
                )

        # Persist final state after Modules have stopped.
        self.save_state()

    # ==================================================================
    # Agent / turn snapshots
    # ==================================================================

    def snapshot(
        self,
    ) -> Mapping[
        str,
        Mapping[str, Any],
    ]:
        """
        Capture one detached global DataSpace snapshot.

        The returned object can safely be shared by the Main Agent
        and all descendant Subagents participating in the same turn.
        """
        snapshot = {
            module_id: record.data.snapshot()
            for module_id, record
            in self.modules.items()
        }

        return MappingProxyType(
            snapshot
        )

    async def query(
        self,
        turn: Any,
    ) -> list[str]:
        """
        Compatibility convenience API.

        Captures a fresh DataSpace snapshot and queries all RUNNING
        Modules against it.
        """
        snapshot = self.snapshot()

        return await self.query_snapshot(
            turn,
            snapshot,
        )

    async def query_snapshot(
        self,
        turn: Any,
        snapshot: Mapping[
            str,
            Mapping[str, Any],
        ],
        *,
        on_start: Any = None,
        on_result: Any = None,
    ) -> list[str]:
        """
        Query all RUNNING Modules against an existing snapshot.

        The supplied snapshot is not recaptured.

        This is the API used by Core Agent and Subagents so the entire
        dispatch tree can observe exactly the same world state.
        """
        module_turn = ModuleTurn(
            turn=turn,
            data=snapshot,
        )

        records = [
            record
            for record in self.modules.values()
            if record.state
            is ModuleState.RUNNING
        ]

        async def run_one(record: Any) -> Any:
            if on_start is not None:
                on_start(record.id)

            started = time.time()

            try:
                result = await record.instance.query(
                    module_turn,
                )
                failed = False
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    f"Module query failed: {record.id}[generation={record.generation}]",
                )
                result, failed = None, True

            if on_result is not None:
                on_result(
                    record.id,
                    result,
                    time.time() - started,
                    failed,
                )

            return result

        results = await asyncio.gather(
            *(
                run_one(record)
                for record in records
            ),
        )

        return [
            result
            for result in results
            if result is not None
            and result != ""
        ]

    async def _query_record(
        self,
        record: ModuleRecord,
        turn: ModuleTurn,
    ) -> str | None:
        try:
            return await record.instance.query(
                turn
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(f'''Module query failed: {record.id}[generation={record.generation}]''')

            return None

    # ==================================================================
    # Persistence
    # ==================================================================

    def _ensure_data_dirs(self) -> None:
        _persistence.ensure_data_dirs(
            self.private_dir,
            self.dataspace_dir,
        )

    def _private_state_path(
        self,
        module_id: str,
    ) -> Path:
        return _persistence.private_state_path(
            self.private_dir,
            module_id,
        )

    def _dataspace_path(
        self,
        module_id: str,
    ) -> Path:
        return _persistence.dataspace_path(
            self.dataspace_dir,
            module_id,
        )

    def _read_json_file(
        self,
        path: Path,
    ) -> Any:
        return _persistence.read_json_file(path)

    def _atomic_write_json(
        self,
        path: Path,
        value: Any,
    ) -> None:
        _persistence.atomic_write_json(path, value)

    def _load_dataspace_state(
        self,
        record: ModuleRecord,
    ) -> bool:
        return _persistence.load_dataspace_state(
            record,
            self.dataspace_dir,
        )

    def _load_private_state(
        self,
        record: ModuleRecord,
    ) -> bool:
        return _persistence.load_private_state(
            record,
            self.private_dir,
        )

    def _restore_record_state(
        self,
        record: ModuleRecord,
    ) -> None:
        _persistence.restore_record_state(
            record,
            private_dir=self.private_dir,
            dataspace_dir=self.dataspace_dir,
        )

    def _save_record_state(
        self,
        record: ModuleRecord,
    ) -> None:
        _persistence.save_record_state(
            record,
            private_dir=self.private_dir,
            dataspace_dir=self.dataspace_dir,
        )

    def save_state(self) -> None:
        self._ensure_data_dirs()

        for record in self.modules.values():
            try:
                self._save_record_state(
                    record
                )

            except Exception:
                logger.exception(f'''Failed to persist Module state: {record.id}''')

    # ==================================================================
    # Discovery / loading
    # ==================================================================

    def _load_builtins(self) -> None:
        for cls in self.builtin_modules:
            if cls.id in self.modules:
                continue

            self._register_module_class(
                cls,
                source="<builtin>",
                origin="builtin",
            )

    async def _scan_workspace_modules(
        self,
    ) -> None:
        self.workspace_modules.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_files = {
            path.resolve()
            for path in self.workspace_modules.rglob(
                "*.py"
            )
            if path.is_file()
            and not path.name.startswith("_")
        }

        known_files = {
            Path(record.source).resolve()
            for record in self.modules.values()
            if record.origin == "workspace"
        }

        # --------------------------------------------------------------
        # Removed files
        # --------------------------------------------------------------

        for removed in (
            known_files - current_files
        ):
            record = next(
                (
                    item
                    for item in self.modules.values()
                    if (
                        item.origin
                        == "workspace"
                        and Path(
                            item.source
                        ).resolve()
                        == removed
                    )
                ),
                None,
            )

            if record is not None:
                await self._remove_workspace_record(
                    record
                )

        # --------------------------------------------------------------
        # New / changed files
        # --------------------------------------------------------------

        for path in sorted(
            current_files
        ):
            if not self._has_module_header(
                path
            ):
                continue

            fingerprint = self._fingerprint(
                path
            )

            previous = (
                self._workspace_fingerprints.get(
                    path
                )
            )

            previous_error = (
                self._workspace_load_errors.get(
                    path
                )
            )

            # Don't repeatedly retry an unchanged broken file.
            if (
                previous == fingerprint
                and previous_error is not None
            ):
                continue

            existing = (
                self._find_record_by_source(
                    path
                )
            )

            if (
                previous == fingerprint
                and existing is not None
            ):
                continue

            self._workspace_fingerprints[
                path
            ] = fingerprint

            self._workspace_load_errors.pop(
                path,
                None,
            )

            try:
                await self._load_or_reload_workspace_file(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._workspace_load_errors[
                    path
                ] = exc

                logger.exception(f'''Failed to load workspace Module: {path}''')

                self._wake.set()

    @staticmethod
    def _has_module_header(
        path: Path,
    ) -> bool:
        return _loading.has_module_header(path)

    def _fingerprint(
        self,
        path: Path,
    ) -> tuple[int, int]:
        return _loading.fingerprint(path)

    async def _load_or_reload_workspace_file(
        self,
        path: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        old = self._find_record_by_source(
            path
        )

        (
            cls,
            imported_name,
            _,
        ) = self._import_workspace_class(
            path
        )

        self._validate_module_class(
            cls
        )

        if old is None:
            self._register_module_class(
                cls,
                source=str(path),
                origin="workspace",
                source_fingerprint=fingerprint,
                imported_module_name=(
                    imported_name
                ),
            )

            self._rebuild_dependency_graph()
            self._validate_dependency_graph()

            self._wake.set()

            return

        if old.id != cls.id:
            raise RuntimeError(
                f"hot reload changed Module id "
                f"in {path}: "
                f"{old.id!r} -> {cls.id!r}"
            )

        await hot_reload(
            self,
            old=old,
            cls=cls,
            imported_name=imported_name,
            fingerprint=fingerprint,
        )

    def _import_workspace_class(
        self,
        path: Path,
    ) -> tuple[
        type[Module],
        str,
        ModuleType,
    ]:
        cls, imported_name, module = (
            _loading.import_module_class(path)
        )

        return cls, imported_name, module

    def _validate_module_class(
        self,
        cls: type[Module],
    ) -> None:
        _loading.validate_module_class(cls)

    def _register_module_class(
        self,
        cls: type[Module],
        *,
        source: str,
        origin: str,
        source_fingerprint: (
            tuple[int, int] | None
        ) = None,
        imported_module_name: str | None = None,
    ) -> ModuleRecord:
        self._validate_module_class(
            cls
        )

        module_id = cls.id

        if module_id in self.modules:
            existing = self.modules[
                module_id
            ]

            raise DuplicateModuleError(
                f"duplicate Module id "
                f"{module_id!r}: "
                f"{existing.source} "
                f"and {source}"
            )

        data = self.dataspaces.get(
            module_id
        )

        if data is None:
            data = DataSpace(
                owner=module_id
            )

            self.dataspaces[
                module_id
            ] = data

        instance = cls()

        instance.llm = self.llm

        record = ModuleRecord(
            id=module_id,
            cls=cls,
            instance=instance,
            data=data,
            source=source,
            origin=origin,
            generation=0,
            source_fingerprint=(
                source_fingerprint
            ),
            imported_module_name=(
                imported_module_name
            ),
        )

        self._bind_instance(
            record
        )

        self.modules[
            module_id
        ] = record

        self._restore_record_state(
            record
        )

        if (
            origin == "workspace"
            and source_fingerprint is not None
        ):
            self._workspace_fingerprints[
                Path(source).resolve()
            ] = source_fingerprint

        return record

    def _bind_instance(
        self,
        record: ModuleRecord,
    ) -> None:
        _deps.bind_instance(
            record,
            self.dataspaces,
        )

    def _find_record_by_source(
        self,
        path: Path,
    ) -> ModuleRecord | None:
        path = path.resolve()

        for record in self.modules.values():
            if (
                record.origin
                == "workspace"
                and Path(
                    record.source
                ).resolve()
                == path
            ):
                return record

        return None

    # ==================================================================
    # Dependency graph
    # ==================================================================

    def _rebuild_dependency_graph(
        self,
    ) -> None:
        (
            self.dependencies,
            self.dependents,
        ) = _deps.build_dependency_maps(
            self.modules
        )

        for record in self.modules.values():
            self._bind_instance(
                record
            )

    def _validate_dependency_graph(
        self,
    ) -> None:
        self._topological_order()

    def _topological_order(
        self,
    ) -> list[str]:
        return _deps.topological_order(
            self.modules,
            self.dependencies,
            self.dependents,
        )

    def _safe_topological_order(
        self,
    ) -> list[str]:
        return _deps.safe_topological_order(
            self.modules,
            self.dependencies,
            self.dependents,
        )

    # ==================================================================
    # Supervisor
    # ==================================================================

    async def _supervisor(
        self,
    ) -> None:
        next_scan = 0.0

        while not self._stopping:
            now = time.monotonic()

            if now >= next_scan:
                try:
                    await self._scan_workspace_modules()

                    self._rebuild_dependency_graph()
                    self._validate_dependency_graph()

                    await self._reconcile()

                except Exception:
                    logger.exception(
                        "Module reconciliation scan failed"
                    )

                next_scan = (
                    time.monotonic()
                    + self.scan_interval
                )

            try:
                await self._reconcile()

            except Exception:
                logger.exception(
                    "Module reconcile failed"
                )

            timeout = min(
                max(
                    next_scan
                    - time.monotonic(),
                    0.0,
                ),
                self.retry_interval,
            )

            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=timeout,
                )

            except asyncio.TimeoutError:
                pass

            finally:
                self._wake.clear()

    async def _reconcile(
        self,
    ) -> None:
        now = time.monotonic()

        for module_id in self._topological_order():
            record = self.modules.get(
                module_id
            )

            if record is None:
                continue

            if record.state in {
                ModuleState.STARTING,
                ModuleState.RUNNING,
                ModuleState.STOPPING,
            }:
                continue

            if record.retry_at > now:
                continue

            await self._try_start(
                record
            )

    async def _try_start(
        self,
        record: ModuleRecord,
    ) -> None:
        if record.task is not None:
            if not record.task.done():
                return

            record.task = None

        self._bind_instance(
            record
        )

        record.error = None

        record.state = (
            ModuleState.STARTING
        )

        started = asyncio.Event()

        record.task = asyncio.create_task(
            self._run_module(
                record,
                started,
            ),
            name=(
                f"module:"
                f"{record.id}:"
                f"{record.generation}"
            ),
        )

        await started.wait()

    async def _run_module(
        self,
        record: ModuleRecord,
        started: asyncio.Event | None = None,
    ) -> None:
        record.state = (
            ModuleState.RUNNING
        )

        if started is not None:
            started.set()

        try:
            await record.instance.start()

        except asyncio.CancelledError:
            if not self._stopping:
                record.state = (
                    ModuleState.DOWN
                )

                record.error = None

                record.retry_at = (
                    time.monotonic()
                    + self.retry_interval
                )

                logger.warning(f'''Module cancelled unexpectedly: {record.id}''')

                self._wake.set()

            raise

        except Exception as exc:
            record.state = (
                ModuleState.DOWN
            )

            record.error = exc

            record.retry_at = (
                time.monotonic()
                + self.retry_interval
            )

            logger.exception(f'''Module crashed: {record.id}[generation={record.generation}]''')

            self._wake.set()

        else:
            record.state = (
                ModuleState.DOWN
            )

            record.error = None

            record.retry_at = (
                time.monotonic()
                + self.retry_interval
            )

            logger.info(f'''Module exited normally: {record.id}[generation={record.generation}]''')

            self._wake.set()

        finally:
            current_task = (
                asyncio.current_task()
            )

            if record.task is current_task:
                record.task = None

    async def _stop_record(
        self,
        record: ModuleRecord,
    ) -> None:
        record.state = (
            ModuleState.STOPPING
        )

        try:
            await record.instance.stop()

        except Exception as exc:
            record.error = exc

            logger.exception(f'''Module stop failed: {record.id}[generation={record.generation}]''')

        task = record.task

        if (
            task is not None
            and not task.done()
        ):
            task.cancel()

            try:
                await task

            except asyncio.CancelledError:
                pass

            except Exception:
                pass

        record.task = None

        if self._stopping:
            record.state = (
                ModuleState.STOPPING
            )
        else:
            record.state = (
                ModuleState.DOWN
            )

            record.retry_at = (
                time.monotonic()
                + self.retry_interval
            )

    # ==================================================================
    # Hot Reload
    # ==================================================================

    async def _remove_workspace_record(
        self,
        record: ModuleRecord,
    ) -> None:
        logger.info(f'''Removing workspace Module: {record.id}''')

        await self._stop_record(
            record
        )

        self.modules.pop(
            record.id,
            None,
        )

        if record.imported_module_name:
            sys.modules.pop(
                record.imported_module_name,
                None,
            )

        source_path = Path(
            record.source
        ).resolve()

        self._workspace_fingerprints.pop(
            source_path,
            None,
        )

        self._workspace_load_errors.pop(
            source_path,
            None,
        )

        # DataSpace intentionally survives
        # Module removal.

        self._rebuild_dependency_graph()

        self._wake.set()