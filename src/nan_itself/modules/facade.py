from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, ClassVar, Mapping


logger = logging.getLogger(__name__)

MODULE_HEADER = "# @module"


# ============================================================================
# DataSpace
# ============================================================================


class DataSpace:
    """
    A Module-owned published state.

    The owner Module is the only writer.
    Readers receive detached deep copies.
    """

    def __init__(
        self,
        owner: str,
        initial: Mapping[str, Any] | None = None,
    ) -> None:
        self.owner = owner
        self._value: dict[str, Any] = deepcopy(
            dict(initial or {})
        )
        self.revision = 0

    def publish(
        self,
        value: Mapping[str, Any],
    ) -> None:
        """
        Atomically publish a new state.

        The input is deep-copied so the caller cannot retain
        references into DataSpace internals.
        """
        if not isinstance(value, Mapping):
            raise TypeError(
                "DataSpace value must be a Mapping"
            )

        value_copy = deepcopy(
            dict(value)
        )

        self._value = value_copy
        self.revision += 1

    def snapshot(self) -> dict[str, Any]:
        """
        Return a completely detached snapshot.
        """
        return deepcopy(self._value)

    def dump(self) -> dict[str, Any]:
        """
        Alias used by Facade persistence.
        """
        return self.snapshot()


class DataSpaceReader:
    """
    Read-only handle exposed to dependent Modules.

    It deliberately does not expose publish().
    """

    __slots__ = ("_space",)

    def __init__(
        self,
        space: DataSpace,
    ) -> None:
        self._space = space

    @property
    def owner(self) -> str:
        return self._space.owner

    @property
    def revision(self) -> int:
        return self._space.revision

    def snapshot(self) -> dict[str, Any]:
        return self._space.snapshot()


# ============================================================================
# Module
# ============================================================================


@dataclass(frozen=True)
class ModuleTurn:
    """
    Lightweight query view supplied to a Module.

    `turn` is the opaque Agent turn object.

    `data` is one consistent detached DataSpace snapshot.

    Core may put additional identity information in `turn`, such as:
        - agent_hash
        - depth
        - turn_id

    Module is free to interpret those fields.
    """

    turn: Any
    data: Mapping[str, Mapping[str, Any]]

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            self.turn,
            name,
        )


class Module:
    """
    Base class for all Modules.

    A Module owns:
        self.data

    A Module can read:
        self.dependencies

    A Module's start() represents its service lifetime.
    """

    id: ClassVar[str]

    requires: ClassVar[
        tuple[str, ...]
    ] = ()

    data: DataSpace

    dependencies: Mapping[
        str,
        DataSpaceReader,
    ]

    async def start(self) -> None:
        """
        Enter the Module's running state.

        Normally this should be a long-running coroutine.

        A state-only Module may return immediately; Facade will treat
        it as exited and retry it later.
        """
        raise NotImplementedError

    async def stop(self) -> None:
        """
        Stop all work owned by this Module.

        Exceptions are recorded by Facade, but do not prevent
        cleanup of other Modules.
        """

    async def query(
        self,
        turn: ModuleTurn,
    ) -> str | None:
        """
        Produce Agent-facing context/prompt for this turn.

        Query failure does not automatically bring the Module down.
        """
        return None

    def serialize_state(self) -> Any:
        """
        Return JSON-serializable private Module state.

        Facade stores the result under:
            ./data/modules/private/<module_id>.json
        """
        return None

    def restore_state(
        self,
        state: Any,
    ) -> None:
        """
        Restore private Module state.
        """
        pass


class ModuleState(Enum):
    NEW = auto()
    STARTING = auto()
    RUNNING = auto()
    DOWN = auto()
    STOPPING = auto()


@dataclass
class ModuleRecord:
    id: str
    cls: type[Module]
    instance: Module
    data: DataSpace

    source: str
    origin: str  # "builtin" | "workspace"

    generation: int = 0

    task: asyncio.Task[None] | None = None

    state: ModuleState = ModuleState.NEW

    error: BaseException | None = None

    retry_at: float = 0.0

    source_fingerprint: tuple[int, int] | None = None

    imported_module_name: str | None = None


# ============================================================================
# Builtin Modules
# ============================================================================

# Add builtin Module classes here.
#
# Example:
#
# class ExampleModule(Module):
#     id = "example"
#
#     async def start(self):
#         while True:
#             self.data.publish({"status": "alive"})
#             await asyncio.sleep(1)
#
#
# BUILTIN_MODULES = (ExampleModule,)

BUILTIN_MODULES: tuple[
    type[Module],
    ...
] = ()


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
    ) -> None:
        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
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

        results = await asyncio.gather(
            *(
                self._query_record(
                    record,
                    module_turn,
                )
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
            logger.exception(
                "Module query failed: "
                "%s[generation=%d]",
                record.id,
                record.generation,
            )

            return None

    # ==================================================================
    # Persistence
    # ==================================================================

    def _ensure_data_dirs(self) -> None:
        self.private_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.dataspace_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _private_state_path(
        self,
        module_id: str,
    ) -> Path:
        return (
            self.private_dir
            / f"{module_id}.json"
        )

    def _dataspace_path(
        self,
        module_id: str,
    ) -> Path:
        return (
            self.dataspace_dir
            / f"{module_id}.json"
        )

    @staticmethod
    def _read_json_file(
        path: Path,
    ) -> Any:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _atomic_write_json(
        path: Path,
        value: Any,
    ) -> None:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = path.with_name(
            f".{path.name}.{os.getpid()}.tmp"
        )

        try:
            with temp_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                file.write(encoded)
                file.write("\n")
                file.flush()
                os.fsync(
                    file.fileno()
                )

            os.replace(
                temp_path,
                path,
            )

        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    def _load_dataspace_state(
        self,
        record: ModuleRecord,
    ) -> bool:
        path = self._dataspace_path(
            record.id
        )

        if not path.exists():
            return True

        try:
            value = self._read_json_file(
                path
            )

            if not isinstance(
                value,
                dict,
            ):
                raise TypeError(
                    "DataSpace file must contain "
                    f"a JSON object: {path}"
                )

            record.data.publish(
                value
            )

            logger.info(
                "Restored DataSpace: %s "
                "revision=%d",
                record.id,
                record.data.revision,
            )

            return True

        except Exception:
            logger.exception(
                "Failed to restore DataSpace "
                "for Module %s",
                record.id,
            )

            return False

    def _load_private_state(
        self,
        record: ModuleRecord,
    ) -> bool:
        path = self._private_state_path(
            record.id
        )

        if not path.exists():
            return True

        try:
            state = self._read_json_file(
                path
            )

            record.instance.restore_state(
                state
            )

            logger.info(
                "Restored private state: %s",
                record.id,
            )

            return True

        except Exception:
            logger.exception(
                "Failed to restore private state "
                "for Module %s",
                record.id,
            )

            return False

    def _restore_record_state(
        self,
        record: ModuleRecord,
    ) -> None:
        self._load_dataspace_state(
            record
        )

        self._load_private_state(
            record
        )

    def _save_record_state(
        self,
        record: ModuleRecord,
    ) -> None:
        private_state = (
            record.instance.serialize_state()
        )

        self._atomic_write_json(
            self._private_state_path(
                record.id
            ),
            private_state,
        )

        self._atomic_write_json(
            self._dataspace_path(
                record.id
            ),
            record.data.dump(),
        )

    def save_state(self) -> None:
        self._ensure_data_dirs()

        for record in self.modules.values():
            try:
                self._save_record_state(
                    record
                )

            except Exception:
                logger.exception(
                    "Failed to persist Module state: %s",
                    record.id,
                )

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

                logger.exception(
                    "Failed to load workspace Module: %s",
                    path,
                )

                self._wake.set()

    @staticmethod
    def _has_module_header(
        path: Path,
    ) -> bool:
        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except OSError:
            return False

        head = "\n".join(
            text.splitlines()[:20]
        )

        return MODULE_HEADER in head

    @staticmethod
    def _fingerprint(
        path: Path,
    ) -> tuple[int, int]:
        stat = path.stat()

        return (
            stat.st_mtime_ns,
            stat.st_size,
        )

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

        await self._hot_reload(
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
        digest = hashlib.sha1(
            str(path).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:12]

        module_name = (
            f"_workspace_module_"
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

        module = (
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
                and issubclass(cls, Module)
                and cls is not Module
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
                f"one concrete Module; "
                f"found {len(candidates)}"
            )

        return (
            candidates[0],
            module_name,
            module,
        )

    @staticmethod
    def _validate_module_class(
        cls: type[Module],
    ) -> None:
        module_id = getattr(
            cls,
            "id",
            None,
        )

        if not isinstance(
            module_id,
            str,
        ):
            raise TypeError(
                f"Module {cls.__name__} "
                "must define a string id"
            )

        if not module_id:
            raise ValueError(
                f"Module {cls.__name__} "
                "has empty id"
            )

        requires = getattr(
            cls,
            "requires",
            (),
        )

        if not isinstance(
            requires,
            tuple,
        ):
            raise TypeError(
                f"Module {module_id!r}.requires "
                "must be tuple[str, ...]"
            )

        if any(
            not isinstance(
                item,
                str,
            )
            or not item
            for item in requires
        ):
            raise TypeError(
                f"Module {module_id!r}.requires "
                "contains invalid ids"
            )

        if len(
            set(requires)
        ) != len(requires):
            raise ValueError(
                f"Module {module_id!r}.requires "
                "contains duplicates"
            )

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

            raise RuntimeError(
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
        record.instance.data = (
            record.data
        )

        dependencies: dict[
            str,
            DataSpaceReader,
        ] = {}

        for dependency_id in (
            record.cls.requires
        ):
            space = self.dataspaces.get(
                dependency_id
            )

            if space is None:
                # Missing dependencies are allowed at bind time.
                # Facade may still start this Module.
                continue

            dependencies[
                dependency_id
            ] = DataSpaceReader(
                space
            )

        record.instance.dependencies = (
            MappingProxyType(
                dependencies
            )
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
        self.dependencies = {
            module_id: set(
                record.cls.requires
            )
            for module_id, record
            in self.modules.items()
        }

        self.dependents = {
            module_id: set()
            for module_id in self.modules
        }

        for (
            module_id,
            required_ids,
        ) in self.dependencies.items():
            for dependency_id in required_ids:
                if (
                    dependency_id
                    in self.dependents
                ):
                    self.dependents[
                        dependency_id
                    ].add(
                        module_id
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
        indegree = {
            module_id: 0
            for module_id in self.modules
        }

        for (
            module_id,
            required_ids,
        ) in self.dependencies.items():
            for dependency_id in required_ids:
                if (
                    dependency_id
                    in indegree
                ):
                    indegree[
                        module_id
                    ] += 1
                else:
                    logger.warning(
                        "Module %s requires missing "
                        "Module %s",
                        module_id,
                        dependency_id,
                    )

        queue = sorted(
            module_id
            for module_id, degree
            in indegree.items()
            if degree == 0
        )

        result: list[str] = []

        while queue:
            current = queue.pop(0)

            result.append(
                current
            )

            for dependent in sorted(
                self.dependents.get(
                    current,
                    (),
                )
            ):
                indegree[
                    dependent
                ] -= 1

                if (
                    indegree[dependent]
                    == 0
                ):
                    queue.append(
                        dependent
                    )

            queue.sort()

        if len(result) != len(
            self.modules
        ):
            remaining = sorted(
                module_id
                for module_id in self.modules
                if module_id not in result
            )

            raise RuntimeError(
                "Module dependency cycle "
                "detected: "
                + " -> ".join(
                    remaining
                )
            )

        return result

    def _safe_topological_order(
        self,
    ) -> list[str]:
        try:
            return self._topological_order()

        except Exception:
            return list(
                self.modules
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

                logger.warning(
                    "Module cancelled "
                    "unexpectedly: %s",
                    record.id,
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
                "Module crashed: "
                "%s[generation=%d]",
                record.id,
                record.generation,
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
                "Module exited normally: "
                "%s[generation=%d]",
                record.id,
                record.generation,
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
                "Module stop failed: "
                "%s[generation=%d]",
                record.id,
                record.generation,
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

    async def _hot_reload(
        self,
        old: ModuleRecord,
        cls: type[Module],
        imported_name: str,
        fingerprint: tuple[int, int],
    ) -> None:
        logger.info(
            "Reloading Module %s: "
            "generation %d -> %d",
            old.id,
            old.generation,
            old.generation + 1,
        )

        # --------------------------------------------------------------
        # Capture live private state.
        # --------------------------------------------------------------

        try:
            private_state = (
                old.instance.serialize_state()
            )

        except Exception:
            logger.exception(
                "Hot reload rejected for Module %s: "
                "failed to serialize current "
                "private state",
                old.id,
            )

            sys.modules.pop(
                imported_name,
                None,
            )

            return

        # --------------------------------------------------------------
        # Create candidate generation.
        # --------------------------------------------------------------

        candidate_instance = cls()

        candidate = ModuleRecord(
            id=old.id,
            cls=cls,
            instance=candidate_instance,
            data=old.data,
            source=old.source,
            origin=old.origin,
            generation=(
                old.generation + 1
            ),
            source_fingerprint=fingerprint,
            imported_module_name=(
                imported_name
            ),
        )

        self._bind_instance(
            candidate
        )

        try:
            candidate.instance.restore_state(
                deepcopy(
                    private_state
                )
            )

        except Exception:
            logger.exception(
                "Hot reload rejected for "
                "Module %s: failed to restore "
                "private state",
                old.id,
            )

            sys.modules.pop(
                imported_name,
                None,
            )

            return

        # --------------------------------------------------------------
        # Start candidate.
        # --------------------------------------------------------------

        started = asyncio.Event()

        candidate.state = (
            ModuleState.STARTING
        )

        candidate.task = asyncio.create_task(
            self._run_module(
                candidate,
                started,
            ),
            name=(
                f"module:"
                f"{candidate.id}:"
                f"{candidate.generation}:"
                f"candidate"
            ),
        )

        await started.wait()

        # --------------------------------------------------------------
        # Reject failed candidates.
        # --------------------------------------------------------------

        if candidate.state is ModuleState.DOWN:
            logger.error(
                "Hot reload rejected for "
                "Module %s: candidate failed "
                "during startup",
                old.id,
            )

            if (
                candidate.task is not None
                and not candidate.task.done()
            ):
                candidate.task.cancel()

                try:
                    await candidate.task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            sys.modules.pop(
                imported_name,
                None,
            )

            return

        if candidate.state is not (
            ModuleState.RUNNING
        ):
            logger.error(
                "Hot reload candidate entered "
                "unexpected state: %s -> %s",
                old.id,
                candidate.state,
            )

            if (
                candidate.task is not None
                and not candidate.task.done()
            ):
                candidate.task.cancel()

                try:
                    await candidate.task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            sys.modules.pop(
                imported_name,
                None,
            )

            return

        # --------------------------------------------------------------
        # Validate new dependency graph.
        # --------------------------------------------------------------

        self.modules[
            old.id
        ] = candidate

        self._rebuild_dependency_graph()

        try:
            self._validate_dependency_graph()

        except Exception:
            self.modules[
                old.id
            ] = old

            self._rebuild_dependency_graph()

            if (
                candidate.task is not None
                and not candidate.task.done()
            ):
                candidate.task.cancel()

                try:
                    await candidate.task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            sys.modules.pop(
                imported_name,
                None,
            )

            logger.exception(
                "Hot reload rejected for "
                "Module %s: invalid dependency graph",
                old.id,
            )

            return

        # --------------------------------------------------------------
        # Commit.
        # --------------------------------------------------------------

        old_task = old.task

        try:
            await old.instance.stop()

        except Exception:
            logger.exception(
                "Old Module generation stop failed: "
                "%s[generation=%d]",
                old.id,
                old.generation,
            )

        if (
            old_task is not None
            and not old_task.done()
        ):
            old_task.cancel()

            try:
                await old_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if (
            old.imported_module_name
            and old.imported_module_name
            != imported_name
        ):
            sys.modules.pop(
                old.imported_module_name,
                None,
            )

        self._workspace_fingerprints[
            Path(
                old.source
            ).resolve()
        ] = fingerprint

        self._workspace_load_errors.pop(
            Path(
                old.source
            ).resolve(),
            None,
        )

        self._wake.set()

    async def _remove_workspace_record(
        self,
        record: ModuleRecord,
    ) -> None:
        logger.info(
            "Removing workspace Module: %s",
            record.id,
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

        self._rebuild_dependency_graph()

        self._wake.set()