"""
Hot reload transaction.

Seven-step protocol replacing one running module generation
with another, keeping the DataSpace alive throughout:

    1 serialize old private state        (fail -> reject)
    2 build candidate generation        (shares the DataSpace)
    3 restore candidate private state   (fail -> reject)
    4 start candidate                  (fail -> reject)
    5 health-check candidate state     (fail -> reject)
    6 swap pointer + rebuild graph     (fail -> roll back)
    7 commit: stop old, evict import, update books

The candidate starts BEFORE the old generation stops, so there
is a brief overlap window; single-writer ownership of the
DataSpace transfers during it.

IMPORTANT ISOLATION CONTRACT:

    Reloading Module A must only manage Module A.

    A's `requires` relationship is a DataSpace dependency,
    not a lifecycle dependency.

    Therefore this module MUST NOT:
        - stop a dependency Module
        - start a dependency Module
        - restart a dependency Module
        - re-bind unrelated Module instances

    Dependency graph rebuild is pure bookkeeping.
    Candidate binding is done only for the candidate itself.

IMPORTANT CONCURRENCY CONTRACT:

    There may be multiple callers attempting to reload the same
    Module concurrently.

    Only one transaction may operate on one Module ID at a time.

    A transaction that captured an old ModuleRecord but acquires
    the lock after another transaction already replaced that record
    is stale and MUST NOT commit.

IMPORTANT START CONTRACT:

    Module.start() is a lifetime coroutine and normally does not
    return.

    Therefore reload does not wait for start() to finish.

    Instead, after the candidate runner enters its lifecycle, one
    scheduler turn is allowed for an immediate failure to surface.
    A candidate that is still alive is considered running.
"""

from __future__ import annotations

import asyncio
import sys
from copy import deepcopy
from typing import Any

from loguru import logger

from .model import (
    Module,
    ModuleRecord,
    ModuleState,
)


def _reject_reload(
    facade,
    old: ModuleRecord,
    imported_name: str,
    reason: BaseException,
) -> None:
    """
    Abort one reload: record why it was rejected and drop the
    candidate import.

    A rejected reload keeps its updated fingerprint, so the
    rejection must be cached in the table the scanner consults --
    otherwise the unchanged file would be skipped silently
    forever.
    """
    facade._module_load_errors[
        old.source_path
    ] = reason

    _evict_import(imported_name)


def _evict_import(
    imported_name: str | None,
) -> None:
    if imported_name:
        sys.modules.pop(
            imported_name,
            None,
        )


def _reload_lock(
    facade,
    module_id: str,
) -> asyncio.Lock:
    """
    Return the per-Facade, per-Module reload lock.

    The table lives on the Facade, so independent Facade
    instances never share a lock.
    """
    lock = facade._reload_locks.get(
        module_id
    )

    if lock is None:
        lock = asyncio.Lock()
        facade._reload_locks[module_id] = lock

    return lock


async def _stop_candidate(
    candidate: ModuleRecord,
) -> None:
    """
    Stop/cancel a rejected candidate.

    The candidate is never installed as the active generation
    when this helper is called.
    """
    task = candidate.task

    if task is not None and not task.done():
        try:
            task.cancel()
            await task

        except asyncio.CancelledError:
            pass

        except Exception:
            logger.exception(
                f"Module task crashed during "
                f"hot-reload stop: {candidate.id}"
            )


async def _candidate_failed_early(
    candidate: ModuleRecord,
    candidate_task: asyncio.Task,
) -> bool:
    """
    Give the candidate one scheduler turn to expose an immediate
    startup failure.

    Module.start() is a lifetime coroutine, so a still-running
    candidate is considered started.

    A candidate whose task has already completed without remaining
    RUNNING is rejected.
    """
    await asyncio.sleep(0)

    if not candidate_task.done():
        return False

    return (
        candidate.state
        is not ModuleState.RUNNING
    )


async def _hot_reload_locked(
    facade: Any,
    *,
    old: ModuleRecord,
    cls: type[Module],
    imported_name: str,
    fingerprint: tuple[int, int],
) -> None:
    """
    Execute one reload transaction while holding the per-Module lock.
    """

    # --------------------------------------------------------------
    # Stale transaction guard.
    #
    # A caller may have captured `old` before another concurrent
    # transaction replaced it.
    #
    # Once the lock is acquired, only the currently installed
    # generation is allowed to start a reload transaction.
    # --------------------------------------------------------------

    current = facade.modules.get(
        old.id
    )

    if current is not old:
        logger.warning(
            "Ignoring stale hot reload transaction for Module {} "
            "(generation={})",
            old.id,
            old.generation,
        )

        _evict_import(
            imported_name
        )

        return

    logger.info(
        "Reloading Module {}: generation {} -> {}",
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

    except Exception as exc:
        logger.exception(
            "Hot reload rejected for Module {}: "
            "failed to serialize current private state",
            old.id,
        )

        _reject_reload(
            facade,
            old,
            imported_name,
            exc,
        )

        return

    # --------------------------------------------------------------
    # Create candidate generation.
    # --------------------------------------------------------------

    try:
        candidate_instance = cls()

    except Exception as exc:
        logger.exception(
            "Hot reload rejected for Module {}: "
            "failed to construct candidate",
            old.id,
        )

        _reject_reload(
            facade,
            old,
            imported_name,
            exc,
        )

        return

    candidate_instance.llm = (
        facade.llm
    )

    candidate = ModuleRecord(
        id=old.id,
        cls=cls,
        instance=candidate_instance,
        data=old.data,
        source_path=old.source_path,
        generation=(
            old.generation + 1
        ),
        source_fingerprint=fingerprint,
        imported_module_name=(
            imported_name
        ),
    )

    # --------------------------------------------------------------
    # Bind ONLY the candidate.
    # --------------------------------------------------------------

    try:
        facade._bind_instance(
            candidate
        )

    except Exception as exc:
        logger.exception(
            "Hot reload rejected for Module {}: "
            "failed to bind candidate",
            old.id,
        )

        _reject_reload(
            facade,
            old,
            imported_name,
            exc,
        )

        return

    # --------------------------------------------------------------
    # Restore candidate private state.
    # --------------------------------------------------------------

    try:
        candidate.instance.restore_state(
            deepcopy(
                private_state
            )
        )

    except Exception as exc:
        logger.exception(
            "Hot reload rejected for Module {}: "
            "failed to restore private state",
            old.id,
        )

        _reject_reload(
            facade,
            old,
            imported_name,
            exc,
        )

        return

    # --------------------------------------------------------------
    # Start candidate.
    #
    # start() is a lifetime coroutine. We do not wait for it to
    # return. We only detect immediate startup failure.
    # --------------------------------------------------------------

    started = asyncio.Event()

    candidate.state = (
        ModuleState.STARTING
    )

    candidate_task = asyncio.create_task(
        facade._run_module(
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

    candidate.task = candidate_task

    await started.wait()

    if await _candidate_failed_early(
        candidate,
        candidate_task,
    ):
        reason = RuntimeError(
            "hot reload rejected: "
            "candidate failed during startup"
        )

        logger.error(
            "Hot reload rejected for Module {}: "
            "candidate failed during startup",
            old.id,
        )

        await _stop_candidate(
            candidate
        )

        _reject_reload(
            facade,
            old,
            imported_name,
            reason,
        )

        return

    # --------------------------------------------------------------
    # Unexpected candidate state.
    # --------------------------------------------------------------

    if candidate.state is not (
        ModuleState.RUNNING
    ):
        reason = RuntimeError(
            "hot reload rejected: "
            f"candidate entered unexpected state "
            f"{candidate.state}"
        )

        logger.error(
            "Hot reload candidate entered "
            "unexpected state: {} -> {}",
            old.id,
            candidate.state,
        )

        await _stop_candidate(
            candidate
        )

        _reject_reload(
            facade,
            old,
            imported_name,
            reason,
        )

        return

    # --------------------------------------------------------------
    # Validate new dependency graph.
    #
    # Pure graph bookkeeping only.
    # Existing Modules are NOT re-bound.
    # --------------------------------------------------------------

    facade.modules[
        old.id
    ] = candidate

    facade._rebuild_dependency_graph()

    try:
        facade._check_dependency_graph_acyclic()

    except Exception as exc:
        # Roll back pointer and graph metadata.
        facade.modules[
            old.id
        ] = old

        facade._rebuild_dependency_graph()

        await _stop_candidate(
            candidate
        )

        logger.exception(
            "Hot reload rejected for Module {}: "
            "invalid dependency graph",
            old.id,
        )

        _reject_reload(
            facade,
            old,
            imported_name,
            exc,
        )

        return

    # --------------------------------------------------------------
    # Commit.
    #
    # Only the generation being replaced is lifecycle-managed.
    # No dependency Module is touched.
    # --------------------------------------------------------------

    old_task = old.task

    try:
        await old.instance.stop()

    except Exception:
        logger.exception(
            "Old Module generation stop failed: "
            "{}[generation={}]",
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
            logger.exception(
                f"Old module generation crashed "
                f"during hot-reload swap: {old.id}"
            )

    if (
        old.imported_module_name
        and old.imported_module_name
        != imported_name
    ):
        sys.modules.pop(
            old.imported_module_name,
            None,
        )

    facade._module_fingerprints[
        old.source_path
    ] = fingerprint

    facade._module_load_errors.pop(
        old.source_path,
        None,
    )

    facade._wake.set()


async def hot_reload(
    facade,
    *,
    old: ModuleRecord,
    cls: type[Module],
    imported_name: str,
    fingerprint: tuple[int, int],
) -> None:
    """
    Public hot reload entry point.

    Reloads for different Module IDs may proceed concurrently.

    Reloads for the same Module ID are serialized.
    """
    lock = _reload_lock(
        facade,
        old.id,
    )

    async with lock:
        await _hot_reload_locked(
            facade,
            old=old,
            cls=cls,
            imported_name=imported_name,
            fingerprint=fingerprint,
        )