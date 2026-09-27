from __future__ import annotations

import asyncio
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from loguru import logger

from . import deps as _deps
from . import loading as _loading
from . import persistence as _persistence
from nan_itself.utils import paths as _paths
from .model import (
    DataSpace,
    DuplicateModuleError,
    Module,
    ModuleRecord,
    ModuleState,
    Turn,
)
from .reload import (
    hot_reload,
)


# ============================================================================
# Facade
# ============================================================================


class Facade:
    """
    Runtime supervisor for Modules.

    Responsibilities:
        - builtin and workspace Module discovery (same
          hot-reload logic per root)
        - Python Module loading
        - dependency graph
        - lifecycle supervision
        - retry
        - hot reload
        - DataSpace ownership
        - persistence
        - Agent-turn DataSpace snapshots
        - Module querying
        - channel downlink routing (never raises)

    Important isolation rule:

        Dependency graph bookkeeping and Module instance binding
        are separate operations.

        Rebuilding the graph MUST NOT implicitly mutate every
        Module instance.

        A Module is bound:
            - when its instance is first created
            - when its replacement generation is created

        Reloading Module A must not re-bind Module B, C, ...
        even when A declares them in `requires`.
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
        Broadcast one completed Turn to every Module.

        Each handler runs in its own task: a slow or failing
        module can never delay its peers, and never delays the
        agents either (this call returns immediately).

        A failing tell() crashes its module (DOWN + supervised
        restart); the failure never propagates to the agents.
        """
        for module_record in list(
            self.modules.values()
        ):
            # Only a RUNNING module receives turns: ask() filters
            # the same way, and a module that has not finished
            # starting (or is being replaced / torn down) must not
            # crash from a delivery aimed at a service that was
            # never fully up.
            if (
                module_record.state
                is not ModuleState.RUNNING
            ):
                continue

            task = asyncio.create_task(
                self._tell_module(
                    module_record,
                    record,
                )
            )

            self._DELIVERY_TASKS.add(task)

            task.add_done_callback(
                self._DELIVERY_TASKS.discard
            )

    async def _tell_module(
        self,
        module_record,
        turn,
    ) -> None:
        try:
            await module_record.instance.tell(
                turn
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self._crash_module(
                module_record,
                exc,
            )

    def __init__(
        self,
        workspace_modules: str | Path | None = None,
        *,
        builtin_modules_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        retry_interval: float = 1.0,
        scan_interval: float = 1.0,
        llm=None,
    ) -> None:
        self.llm = llm

        project_root = _paths.repo_root()

        # Builtin and workspace module files live in parallel
        # directory layouts and are scanned with exactly the same
        # hot-reload logic. Deleting a source file removes its
        # Module.
        self.builtin_modules_dir = (
            Path(
                builtin_modules_dir
            ).resolve()
            if builtin_modules_dir is not None
            else (
                project_root
                / "builtin"
                / "modules"
            ).resolve()
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
            if data_dir is not None
            else _paths.data_dir() / "modules"
        )

        self.private_dir = (
            self.data_dir / "private"
        )

        self.dataspace_dir = (
            self.data_dir / "dataspace"
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

        # Builtin and workspace module files share one identical
        # hot-reload discovery path.
        await self._scan_modules()

        # Graph construction is pure bookkeeping.
        # It must not re-bind already existing instances.
        self._rebuild_dependency_graph(
            bind=False
        )

        self._validate_dependency_graph()

        self._supervisor_task = (
            asyncio.create_task(
                self._supervisor(),
                name="module-facade-supervisor",
            )
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
            self._topological_order()
        ):
            record = records.get(
                module_id
            )

            if record is not None:
                await self._stop_record(
                    record
                )

        # Persist final state after Modules have stopped.
        self.save_state()

    # ==================================================================
    # Agent / turn snapshots
    # ==================================================================

    def get(
        self,
        module_id: str,
    ):
        """
        Return the RUNNING instance of a Module, or None.

        This is how the composition root reaches special Modules
        (e.g. the inbox) without importing them.
        """
        record = self.modules.get(
            module_id
        )

        if record is None:
            return None

        if (
            record.state
            is not ModuleState.RUNNING
        ):
            return None

        return record.instance

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

    async def query_snapshot(
        self,
        turn: Turn,
        *,
        on_start: Any = None,
        on_result: Any = None,
    ) -> list[str]:
        """
        Query all RUNNING Modules against the turn's world.

        The turn carries the DataSpace snapshot captured when the
        turn started, so the entire dispatch tree observes exactly
        the same world state.
        """
        records = [
            record
            for record in self.modules.values()
            if record.state
            is ModuleState.RUNNING
        ]

        async def run_one(
            record: Any,
        ) -> Any:
            if on_start is not None:
                on_start(record.id)

            started = time.time()

            try:
                result = (
                    await record.instance.ask(
                        turn
                    )
                )
                failed = False

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                # Let-it-crash: the module goes DOWN and is
                # restarted by the supervisor; this turn just
                # sees no ambient contribution from it.
                self._crash_module(
                    record,
                    exc,
                )

                result, failed = (
                    None,
                    True,
                )

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

    # ==================================================================
    # Channel downlink routing
    # ==================================================================
    #
    # Routing never raises: every failure (unknown module /
    # channel, module not running, channel-free module, schema
    # rejection) comes back as a result string.

    def list_module_channels(self) -> str:
        """
        Enumerate every RUNNING module's channels.
        """
        lines: list[str] = []

        for record in self.modules.values():
            if record.state is not ModuleState.RUNNING:
                continue

            for name in (
                record.instance.channels
            ):
                lines.append(
                    f"{record.id}/{name}"
                )

        if not lines:
            return (
                "No module exposes channels. "
                "Channels appear when a module "
                "declares them and is RUNNING."
            )

        return "\n".join(lines)

    def show_module_channel(
        self,
        module_id: str,
        channel: str | None = None,
    ) -> str:
        """
        Show channel details: description and schema.
        """
        record = self.modules.get(module_id)

        if record is None:
            return (
                f"Unknown module '{module_id}'. "
                "Use list_channels to see "
                "available channels."
            )

        if record.state is not ModuleState.RUNNING:
            return (
                f"Module '{module_id}' is not "
                "RUNNING."
            )

        channels = record.instance.channels

        if not channels:
            return (
                f"Module '{module_id}' exposes "
                "no channels."
            )

        if channel is not None:
            if channel not in channels:
                return (
                    f"Unknown channel "
                    f"'{module_id}/{channel}'."
                )

            selected = {channel: channels[channel]}

        else:
            selected = dict(channels)

        lines: list[str] = []

        for name, spec in selected.items():
            detail: dict[str, Any] = {
                "name": f"{module_id}/{name}",
                "description": spec.description,
                "schema": spec.json_schema(),
            }

            lines.append(
                json.dumps(
                    detail,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        return "\n".join(lines)

    def write_module_channel(
        self,
        module_id: str,
        channel: str,
        payload: Any,
    ) -> str:
        """
        Validate one payload and feed it to the module.

        Returns 'written' / 'rejected' or an error string.
        Validation failures are rejected at the boundary; the
        module only ever receives well-formed payloads. An
        exception out of feed() crashes the module (DOWN +
        supervised restart) and comes back as a failure string.
        """
        record = self.modules.get(module_id)

        if record is None:
            return (
                f"Unknown module '{module_id}'. "
                "Use list_channels to see "
                "available channels."
            )

        if record.state is not ModuleState.RUNNING:
            return (
                f"Module '{module_id}' is not "
                "RUNNING; payload not delivered."
            )

        channels = record.instance.channels

        if not channels:
            return (
                f"Module '{module_id}' exposes "
                "no channels."
            )

        if channel not in channels:
            return (
                f"Unknown channel "
                f"'{module_id}/{channel}'. "
                "Use show_channels for schemas."
            )

        validated, error = channels[
            channel
        ].validate(payload)

        if error is not None:
            return (
                f"rejected: payload failed "
                f"schema validation for "
                f"'{module_id}/{channel}': "
                f"{error}"
            )

        try:
            return record.instance.feed(
                channel,
                deepcopy(validated),
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            # Let-it-crash: the module is restarted by the
            # supervisor; the model only sees a failure string.
            self._crash_module(record, exc)

            return (
                f"Module '{module_id}' crashed "
                "while accepting the write; "
                "it will be restarted."
            )

    # ==================================================================
    # Persistence
    # ==================================================================

    def _ensure_data_dirs(
        self,
    ) -> None:
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
        return _persistence.read_json_file(
            path
        )

    def _atomic_write_json(
        self,
        path: Path,
        value: Any,
    ) -> None:
        _persistence.atomic_write_json(
            path,
            value,
        )

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

    def save_state(
        self,
    ) -> None:
        self._ensure_data_dirs()

        for record in self.modules.values():
            try:
                self._save_record_state(
                    record
                )

            except Exception:
                logger.exception(
                    f"Failed to persist Module state: "
                    f"{record.id}"
                )

    # ==================================================================
    # Discovery / loading
    # ==================================================================

    async def _scan_modules(
        self,
    ) -> None:
        await self._scan_module_root(
            self.builtin_modules_dir
        )

        await self._scan_module_root(
            self.workspace_modules
        )

    async def _scan_module_root(
        self,
        root: Path,
    ) -> None:
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        current_files = {
            path.resolve()
            for path in (
                root.rglob(
                    "*.py"
                )
            )
            if (
                path.is_file()
                and not path.name.startswith("_")
            )
        }

        known_files = {
            Path(record.source).resolve()
            for record in self.modules.values()
            if Path(
                record.source
            ).resolve().is_relative_to(
                root
            )
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
                        Path(
                            item.source
                        ).resolve()
                        == removed
                    )
                ),
                None,
            )

            if record is not None:
                await self._remove_record(
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
                await self._load_or_reload_file(
                    path,
                    fingerprint,
                )

            except Exception as exc:
                self._workspace_load_errors[
                    path
                ] = exc

                logger.exception(
                    f"Failed to load Module file: "
                    f"{path}"
                )

                self._wake.set()

    @staticmethod
    def _has_module_header(
        path: Path,
    ) -> bool:
        return _loading.has_module_header(
            path
        )

    def _fingerprint(
        self,
        path: Path,
    ) -> tuple[int, int]:
        return _loading.fingerprint(
            path
        )

    async def _load_or_reload_file(
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
        ) = self._import_module_file(
            path
        )

        self._validate_module_class(
            cls
        )

        if old is None:
            self._register_module_class(
                cls,
                source=str(path),
                source_fingerprint=(
                    fingerprint
                ),
                imported_module_name=(
                    imported_name
                ),
            )

            # The new instance was already bound by
            # _register_module_class().
            #
            # Only rebuild pure graph bookkeeping here.
            self._rebuild_dependency_graph(
                bind=False
            )

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

    def _import_module_file(
        self,
        path: Path,
    ) -> tuple[
        type[Module],
        str,
        Any,
    ]:
        (
            cls,
            imported_name,
            module,
        ) = _loading.import_module_class(
            path
        )

        return (
            cls,
            imported_name,
            module,
        )

    def _validate_module_class(
        self,
        cls: type[Module],
    ) -> None:
        _loading.validate_module_class(
            cls
        )

    def _register_module_class(
        self,
        cls: type[Module],
        *,
        source: str,
        source_fingerprint: (
            tuple[int, int]
            | None
        ) = None,
        imported_module_name: (
            str
            | None
        ) = None,
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
            generation=0,
            source_fingerprint=(
                source_fingerprint
            ),
            imported_module_name=(
                imported_module_name
            ),
        )

        # Only the new instance is bound.
        self._bind_instance(
            record
        )

        self.modules[
            module_id
        ] = record

        self._restore_record_state(
            record
        )

        if source_fingerprint is not None:
            self._workspace_fingerprints[
                Path(
                    source
                ).resolve()
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
                Path(
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
        *,
        bind: bool = False,
    ) -> None:
        """
        Rebuild dependency metadata.

        By default this is PURE graph bookkeeping.

        `bind=True` exists only for explicit bulk-rebinding callers.
        Normal reconciliation, validation, removal and hot reload must
        leave existing Module instance bindings untouched.
        """
        (
            self.dependencies,
            self.dependents,
        ) = _deps.build_dependency_maps(
            self.modules
        )

        if bind:
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
                    await self._scan_modules()

                    # Pure graph reconstruction.
                    self._rebuild_dependency_graph(
                        bind=False
                    )

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

        for module_id in (
            self._topological_order()
        ):
            record = self.modules.get(
                module_id
            )

            if record is None:
                continue

            if record.state not in {
                ModuleState.NEW,
                ModuleState.DOWN,
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

        # Explicitly bind only this record.
        #
        # This is required because a Module might be retried after
        # construction, and this method is also the lifecycle entry
        # point for that exact record.
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

        # A task cancelled before its first step never runs its
        # body, so _run_module could not set the event itself;
        # this callback guarantees the wait below terminates
        # whatever happened to the task (done callbacks always
        # run). Double-set on the success path is harmless.
        record.task.add_done_callback(
            lambda _task: started.set()
        )

        await started.wait()

    def _crash_module(
        self,
        record: ModuleRecord,
        exc: BaseException,
    ) -> None:
        """
        Let-it-crash: treat one ask/tell/feed failure exactly
        like a start() crash.

        The module goes DOWN with the recorded error and its
        live start task (any realtime service it runs) is
        cancelled; the supervisor restarts it after backoff.
        The failure never propagates to the caller.

        Fields are set BEFORE the task is cancelled, so the
        task's own cleanup cannot overwrite them.
        """
        record.error = exc

        record.state = (
            ModuleState.DOWN
        )

        record.retry_at = (
            time.monotonic()
            + self.retry_interval
        )

        logger.exception(
            f"Module crashed: "
            f"{record.id}"
            f"[generation={record.generation}]"
        )

        task = record.task

        if (
            task is not None
            and not task.done()
        ):
            task.cancel()

        self._wake.set()

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
                # A crash already recorded by _crash_module()
                # must survive this cleanup: only a plain
                # unexpected cancel (no recorded error) writes
                # the default fields.
                if record.error is None:
                    record.state = (
                        ModuleState.DOWN
                    )

                    record.error = None

                    record.retry_at = (
                        time.monotonic()
                        + self.retry_interval
                    )

                    logger.warning(
                        f"Module cancelled unexpectedly: "
                        f"{record.id}"
                    )

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

            logger.exception(
                f"Module crashed: "
                f"{record.id}"
                f"[generation={record.generation}]"
            )

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

            logger.info(
                f"Module exited normally: "
                f"{record.id}"
                f"[generation={record.generation}]"
            )

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

            logger.exception(
                f"Module stop failed: "
                f"{record.id}"
                f"[generation={record.generation}]"
            )

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
                logger.exception(
                    f"Module task crashed "
                    f"during stop: {record.id}"
                )

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

    async def _remove_record(
        self,
        record: ModuleRecord,
    ) -> None:
        logger.info(
            f"Removing Module: "
            f"{record.id}"
        )

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

        # Pure dependency graph update.
        # Do NOT re-bind remaining Modules.
        self._rebuild_dependency_graph(
            bind=False
        )

        self._wake.set()