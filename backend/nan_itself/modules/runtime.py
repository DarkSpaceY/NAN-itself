from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

from loguru import logger

from nan_itself.utils import blocklist as _blocklist
from nan_itself.utils import hotload as _hotload
from nan_itself.utils import paths as _paths
from nan_itself.utils.hotload import (
    SourceTracker,
    directory_fingerprint,
)

from . import deps as _deps
from . import loading as _loading
from . import persistence as _persistence
from .model import (
    _MODULE_STOP_TIMEOUT,
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

# The stop budget itself lives in model.py (shared with the
# hot-reload commit path, which cannot import it from here
# without a cycle).


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
    """

    def __init__(
        self,
        *,
        retry_interval: float = 1.0,
        scan_interval: float = 1.0,
        llm=None,
    ) -> None:
        self.llm = llm

        project_root = _paths.repo_root()

        # Builtin and workspace module folders live in parallel
        # directory layouts and are scanned with exactly the same
        # hot-reload logic. Deleting a module folder (or its
        # main.py) removes its Module.
        self.builtin_modules_dir = (project_root / "builtin" / "modules").resolve()

        self.workspace_modules_dir = (project_root / "workspace" / "modules").resolve()

        # Legacy state layout (data/modules/{private,dataspace}/
        # <id>.json); migrated into each folder's data/ on first
        # restore, then only cleaned up.
        self._legacy_private_dir = _paths.data_dir() / "modules" / "private"

        self._legacy_dataspace_dir = _paths.data_dir() / "modules" / "dataspace"

        self.retry_interval = retry_interval

        self.scan_interval = scan_interval

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

        # Change bookkeeping for module folders, shared with the
        # tools and skills subsystems (fingerprints, sticky load
        # errors, escalating retry).
        self._source_tracker = SourceTracker()

        self._supervisor_task: asyncio.Task[None] | None = None

        # Strong references to running fire-and-forget
        # delivery tasks so they are not garbage-collected
        # mid-flight.
        self._delivery_tasks: set[asyncio.Task[None]] = set()

        self._wake = asyncio.Event()

        self._stopping = False

    # ==================================================================
    # Public lifecycle
    # ==================================================================

    async def start(self) -> None:
        if self._supervisor_task is not None:
            return

        self._stopping = False

        # Builtin and workspace module files share one identical
        # hot-reload discovery path.
        await self._scan_modules()

        # Graph construction is pure bookkeeping.
        # It must not re-bind already existing instances.
        self._refresh_dependency_graph()

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

            with contextlib.suppress(asyncio.CancelledError):
                await supervisor

        # Stop dependents before dependencies.
        records = self.modules.copy()

        for module_id in reversed(self._topological_order()):
            record = records.get(module_id)

            if record is not None:
                await self._stop_record(record)

        # Persist final state after Modules have stopped.
        self.save_state()

    # ------------------------------------------------------------------
    # Turn delivery
    # ------------------------------------------------------------------

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
        for module_record in list(self.modules.values()):
            # Only a RUNNING module receives turns: ask() filters
            # the same way, and a module that has not finished
            # starting (or is being replaced / torn down) must not
            # crash from a delivery aimed at a service that was
            # never fully up.
            if module_record.state is not ModuleState.RUNNING:
                continue

            task = asyncio.create_task(
                self._tell_module(
                    module_record,
                    record,
                )
            )

            self._delivery_tasks.add(task)

            task.add_done_callback(self._delivery_tasks.discard)

    async def _tell_module(
        self,
        module_record,
        turn,
    ) -> None:
        try:
            await module_record.instance.tell(turn)

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self._crash_module(
                module_record,
                exc,
            )

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
        record = self.modules.get(module_id)

        if record is None:
            return None

        if record.state is not ModuleState.RUNNING:
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
        """
        snapshot = {module_id: record.data.snapshot() for module_id, record in self.modules.items()}

        return MappingProxyType(snapshot)

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
        turn started, handed to each Module as `turn.world`.
        Note: a Module that reads its dependencies through the
        readers in `self.dependencies` observes their *live*
        DataSpace state, not the frozen turn snapshot — the
        snapshot is only guaranteed for code that reads
        `turn.world` itself.
        """
        records = [
            record for record in self.modules.values() if record.state is ModuleState.RUNNING
        ]

        async def run_one(
            record: Any,
        ) -> Any:
            if on_start is not None:
                on_start(record.id)

            started = time.time()

            try:
                result = await record.instance.ask(turn)
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
            *(run_one(record) for record in records),
        )

        return [result for result in results if result is not None and result != ""]

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

            for name in record.instance.channels:
                lines.append(f"{record.id}/{name}")

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
            return f"Unknown module '{module_id}'. Use list_channels to see available channels."

        if record.state is not ModuleState.RUNNING:
            return f"Module '{module_id}' is not RUNNING."

        channels = record.instance.channels

        if not channels:
            return f"Module '{module_id}' exposes no channels."

        if channel is not None:
            if channel not in channels:
                return f"Unknown channel '{module_id}/{channel}'."

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
            return f"Unknown module '{module_id}'. Use list_channels to see available channels."

        if record.state is not ModuleState.RUNNING:
            return f"Module '{module_id}' is not RUNNING; payload not delivered."

        channels = record.instance.channels

        if not channels:
            return f"Module '{module_id}' exposes no channels."

        if channel not in channels:
            return f"Unknown channel '{module_id}/{channel}'. Use show_channels for schemas."

        validated, error = channels[channel].validate(payload)

        if error is not None:
            return (
                f"rejected: payload failed schema validation for '{module_id}/{channel}': {error}"
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

            return f"Module '{module_id}' crashed while accepting the write; it will be restarted."

    # ==================================================================
    # Persistence
    # ==================================================================

    def save_state(
        self,
    ) -> None:
        for record in self.modules.values():
            try:
                self._save_record_state(record)

            except Exception:
                logger.exception(f"Failed to persist Module state: {record.id}")

    def _restore_record_state(
        self,
        record: ModuleRecord,
    ) -> None:
        _persistence.migrate_legacy_state(
            record.id,
            record.instance.data_dir,
            legacy_private_dir=self._legacy_private_dir,
            legacy_dataspace_dir=self._legacy_dataspace_dir,
        )

        _persistence.restore_record_state(record)

    def _save_record_state(
        self,
        record: ModuleRecord,
    ) -> None:
        _persistence.save_record_state(record)

    # ==================================================================
    # Supervisor
    # ==================================================================

    async def _supervisor(
        self,
    ) -> None:
        # The initial scan+reconcile happens in start()
        # before this task runs, so the first periodic
        # scan is one full interval away.
        next_scan = time.monotonic() + self.scan_interval

        while not self._stopping:
            now = time.monotonic()

            if now >= next_scan:
                try:
                    await self._scan_modules()

                    # Pure graph reconstruction.
                    self._refresh_dependency_graph()

                except Exception:
                    logger.exception("Module reconciliation scan failed")

                next_scan = time.monotonic() + self.scan_interval

            try:
                await self._reconcile()

            except Exception:
                logger.exception("Module reconcile failed")

            timeout = min(
                max(
                    next_scan - time.monotonic(),
                    0.0,
                ),
                self.retry_interval,
            )

            try:
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=timeout,
                )

            except TimeoutError:
                pass

            finally:
                self._wake.clear()

    async def _reconcile(
        self,
    ) -> None:
        now = time.monotonic()

        for module_id in self._topological_order():
            record = self.modules.get(module_id)

            if record is None:
                continue

            if record.state not in {
                ModuleState.NEW,
                ModuleState.DOWN,
            }:
                continue

            if record.retry_at > now:
                continue

            await self._try_start(record)

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
        self._bind_instance(record)

        record.error = None

        record.state = ModuleState.STARTING

        started = asyncio.Event()

        record.task = asyncio.create_task(
            self._run_module(
                record,
                started,
            ),
            name=(f"module:{record.id}:{record.generation}"),
        )

        # A task cancelled before its first step never runs its
        # body, so _run_module could not set the event itself;
        # this callback guarantees the wait below terminates
        # whatever happened to the task (done callbacks always
        # run). Double-set on the success path is harmless.
        record.task.add_done_callback(lambda _task: started.set())

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

        record.state = ModuleState.DOWN

        record.retry_at = time.monotonic() + self.retry_interval

        logger.exception(f"Module crashed: {record.id}[generation={record.generation}]")

        task = record.task

        if task is not None and not task.done():
            task.cancel()

        self._wake.set()

    async def _run_module(
        self,
        record: ModuleRecord,
        started: asyncio.Event | None = None,
    ) -> None:
        record.state = ModuleState.RUNNING

        if started is not None:
            started.set()

        try:
            await record.instance.start()

        except asyncio.CancelledError:
            # A crash already recorded by _crash_module() must survive
            # this cleanup: only a plain unexpected cancel (no recorded
            # error) writes the default fields.
            if not self._stopping and record.error is None:
                record.state = ModuleState.DOWN

                record.error = None

                record.retry_at = time.monotonic() + self.retry_interval

                logger.warning(f"Module cancelled unexpectedly: {record.id}")

                self._wake.set()

            raise

        except Exception as exc:
            record.state = ModuleState.DOWN

            record.error = exc

            record.retry_at = time.monotonic() + self.retry_interval

            logger.exception(f"Module crashed: {record.id}[generation={record.generation}]")

            self._wake.set()

        else:
            record.state = ModuleState.DOWN

            record.error = None

            record.retry_at = time.monotonic() + self.retry_interval

            logger.info(f"Module exited normally: {record.id}[generation={record.generation}]")

            self._wake.set()

        finally:
            current_task = asyncio.current_task()

            if record.task is current_task:
                record.task = None

    async def _stop_record(
        self,
        record: ModuleRecord,
    ) -> None:
        record.state = ModuleState.STOPPING

        try:
            await asyncio.wait_for(
                record.instance.stop(),
                timeout=_MODULE_STOP_TIMEOUT,
            )

        except TimeoutError:
            logger.error(
                f"Module stop timed out after "
                f"{_MODULE_STOP_TIMEOUT:g}s: "
                f"{record.id}"
                f"[generation={record.generation}]"
            )

        except Exception as exc:
            record.error = exc

            logger.exception(f"Module stop failed: {record.id}[generation={record.generation}]")

        task = record.task

        if task is not None and not task.done():
            task.cancel()

            try:
                await asyncio.wait_for(
                    task,
                    timeout=_MODULE_STOP_TIMEOUT,
                )

            except TimeoutError:
                logger.error(
                    f"Module task did not exit after "
                    f"cancellation: "
                    f"{record.id}"
                    f"[generation={record.generation}]"
                )

            except asyncio.CancelledError:
                pass

            except Exception:
                logger.exception(f"Module task crashed during stop: {record.id}")

        record.task = None

        if self._stopping:
            record.state = ModuleState.STOPPING

        else:
            record.state = ModuleState.DOWN

            record.retry_at = time.monotonic() + self.retry_interval

    # ==================================================================
    # Discovery / loading
    # ==================================================================

    async def _scan_modules(
        self,
    ) -> None:
        await self._scan_module_root(self.builtin_modules_dir)

        await self._scan_module_root(self.workspace_modules_dir)

    async def _scan_module_root(
        self,
        root: Path,
    ) -> None:
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Blocked names never enter the candidate set (which also
        # unloads an already-loaded one, as a plain removal).
        blocked_names = _blocklist.blocked("modules")

        # One resolved-source index for the whole scan. Both the
        # removal lookup and the already-loaded lookup below would
        # otherwise walk every record once per folder.
        by_source = {Path(record.source).resolve(): record for record in self.modules.values()}

        # A folder is a candidate exactly when it carries main.py;
        # a folder whose main.py is missing counts as removed (its
        # data/ stays on disk, so a re-add restores state).
        current_folders = {
            path.resolve()
            for path in root.iterdir()
            if (
                path.is_dir()
                and not path.name.startswith("_")
                and path.name not in blocked_names
                and (path / "main.py").is_file()
            )
        }

        known_folders = {source for source in by_source if source.is_relative_to(root)}

        # --------------------------------------------------------------
        # Removed folders
        # --------------------------------------------------------------

        for removed in known_folders - current_folders:
            record = by_source.get(removed)

            if record is not None:
                await self._remove_record(record)

        # --------------------------------------------------------------
        # New / changed folders
        # --------------------------------------------------------------

        for folder in sorted(current_folders):
            fingerprint = directory_fingerprint(
                folder,
                exclude_dirs=("data",),
            )

            loaded = folder in by_source

            if not self._source_tracker.needs_load(
                folder,
                fingerprint,
                loaded=loaded,
            ):
                continue

            self._source_tracker.mark_seen(
                folder,
                fingerprint,
            )

            try:
                await self._load_or_reload_folder(
                    folder,
                    fingerprint,
                )

            except Exception as exc:
                self._source_tracker.mark_failed(
                    folder,
                    exc,
                )

                logger.exception(f"Failed to load Module folder: {folder}")

                self._wake.set()

    async def _load_or_reload_folder(
        self,
        folder: Path,
        fingerprint: tuple[int, int],
    ) -> None:
        old = self._find_record_by_source(folder)

        (
            cls,
            imported_name,
        ) = _loading.import_module_class(folder)

        _loading.validate_module_class(cls)

        if old is None:
            try:
                # Validate-then-install: a Module rejected by graph
                # validation never reaches the live tables, so the
                # cycle error cannot leave the graph cyclic for
                # start(), stop() or _reconcile().
                self._register_module_class(
                    cls,
                    source=str(folder),
                    source_fingerprint=(fingerprint),
                    imported_module_name=(imported_name),
                )

            except Exception:
                # The folder was imported before the Facade was
                # involved; evict that synthetic package so a
                # rejected folder leaves nothing behind.
                _hotload.evict_package(imported_name)

                raise

            self._wake.set()

            return

        if old.id != folder.name:
            raise RuntimeError(
                f"hot reload changed Module folder identity in {folder}: "
                f"{old.id!r} -> {folder.name!r}"
            )

        await hot_reload(
            self,
            old=old,
            cls=cls,
            imported_name=imported_name,
            fingerprint=fingerprint,
        )

    def _register_module_class(
        self,
        cls: type[Module],
        *,
        source: str,
        source_fingerprint: (tuple[int, int] | None) = None,
        imported_module_name: (str | None) = None,
    ) -> ModuleRecord:
        """
        Validate one new Module, then install it.

        The dependency graph is checked BEFORE the record reaches any
        live table, so a Module whose arrival would close a cycle is
        rejected while nothing has been written. Rejection therefore
        needs no undo step, and it cannot leave the graph cyclic for
        start(), stop() or _reconcile().
        """
        record = self._build_module_record(
            cls,
            source=source,
            imported_module_name=imported_module_name,
        )

        _deps.validate_addition(
            self.modules,
            record,
        )

        self._install_module_record(
            record,
            source_fingerprint=source_fingerprint,
        )

        # Validation already proved the graph acyclic, so this is
        # pure bookkeeping and cannot fail.
        self._rebuild_dependency_graph()

        return record

    def _build_module_record(
        self,
        cls: type[Module],
        *,
        source: str,
        imported_module_name: (str | None) = None,
    ) -> ModuleRecord:
        """
        Build one ModuleRecord without touching the live tables.

        The instance is constructed and bound to its own `data` and
        readers; nothing shared is written. A record built here can
        therefore still be discarded once validation rejects it.
        """
        _loading.validate_module_class(cls)

        # Identity is the folder name; folders under one root are
        # naturally unique.
        module_id = Path(source).resolve().name

        if module_id in self.modules:
            existing = self.modules[module_id]

            raise DuplicateModuleError(
                f"duplicate Module id {module_id!r}: {existing.source} and {source}"
            )

        data = self.dataspaces.get(module_id)

        if data is None:
            data = DataSpace(owner=module_id)

        instance = cls()

        instance.llm = self.llm

        record = ModuleRecord(
            id=module_id,
            cls=cls,
            instance=instance,
            data=data,
            source=source,
            generation=0,
            imported_module_name=(imported_module_name),
        )

        # Bind the new instance: it receives its own `data` and
        # one reader per declared dependency.
        self._bind_instance(record)

        return record

    def _install_module_record(
        self,
        record: ModuleRecord,
        *,
        source_fingerprint: (tuple[int, int] | None) = None,
    ) -> None:
        """
        Commit a validated record into the live tables.

        Installs the DataSpace, registers the record, restores its
        persisted state, and attaches it to already-registered
        dependents that declare it.
        """
        module_id = record.id

        self.dataspaces[module_id] = record.data

        self.modules[module_id] = record

        self._restore_record_state(record)

        # Attach this Module to already-registered dependents
        # that declare it and currently lack the reader. This is
        # a targeted single-edge operation: only the missing
        # edge is added, nothing else is re-bound.
        #
        # The new record itself is skipped: a Module never
        # depends on its own id.
        for dependent in self.modules.values():
            if dependent is record:
                continue

            _deps.attach_dependency(
                dependent,
                module_id,
                dataspaces=self.dataspaces,
            )

        if source_fingerprint is not None:
            self._source_tracker.mark_seen(
                Path(record.source).resolve(),
                source_fingerprint,
            )

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
            if Path(record.source).resolve() == path:
                return record

        return None

    # ==================================================================
    # Dependency graph
    # ==================================================================

    def _refresh_dependency_graph(
        self,
    ) -> None:
        """
        Rebuild the dependency metadata and reject cycles.

        Pure graph bookkeeping: existing Module instance bindings
        are never touched.
        """
        self._rebuild_dependency_graph()

        self._topological_order()

    def _rebuild_dependency_graph(
        self,
    ) -> None:
        """
        Rebuild the dependency graph maps.

        Pure graph bookkeeping: existing Module instance bindings
        are never touched.
        """
        (
            self.dependencies,
            self.dependents,
        ) = _deps.build_dependency_maps(self.modules)

    def _topological_order(
        self,
    ) -> list[str]:
        return _deps.topological_order(
            self.modules,
            self.dependencies,
            self.dependents,
        )

    # ==================================================================
    # Source removal
    # ==================================================================

    async def _remove_record(
        self,
        record: ModuleRecord,
    ) -> None:
        logger.info(f"Removing Module: {record.id}")

        await self._stop_record(record)

        # Persist this record's state before releasing it, so a
        # later re-add restores the state as of this moment.
        # Best effort: a serialization failure must not abort
        # removal.
        try:
            self._save_record_state(record)

        except Exception:
            logger.exception(f"Failed to persist Module state: {record.id}")

        # The destructive part follows awaits, during which a hot
        # reload may have installed a new generation for this id.
        # Tearing the new generation down would destroy the
        # DataSpace it shares with the old one, so only proceed
        # while this exact record is still the installed one.
        if self.modules.get(record.id) is not record:
            logger.warning(
                "Ignoring stale removal of Module {} (generation={})",
                record.id,
                record.generation,
            )

            _hotload.evict_package(record.imported_module_name)

            return

        self.modules.pop(
            record.id,
            None,
        )

        _hotload.evict_package(record.imported_module_name)

        source_path = Path(record.source).resolve()

        self._source_tracker.forget(source_path)

        # Release the removed Module's in-memory DataSpace. Its
        # state lives in the folder's data/ directory: when only
        # main.py was deleted the folder (and state) survives a
        # re-add; when the whole folder is gone, so is the state.
        self.dataspaces.pop(
            record.id,
            None,
        )

        # Detach the removed Module from every remaining
        # dependent's reader map. Targeted single-edge operation:
        # nothing else is re-bound.
        for remaining in self.modules.values():
            _deps.detach_dependency(
                remaining,
                record.id,
            )

        # Pure dependency graph update.
        # Do NOT re-bind remaining Modules.
        self._rebuild_dependency_graph()

        self._wake.set()
