"""
Hot reload transaction.

Seven-step protocol replacing one running module generation
with another, keeping the DataSpace alive throughout:

    1 serialize old private state        (fail -> reject)
    2 build candidate generation          (shares the DataSpace)
    3 restore candidate private state     (fail -> reject)
    4 start candidate                     (fail -> reject)
    5 health-check candidate state        (fail -> reject)
    6 swap pointer + rebuild graph        (fail -> roll back)
    7 commit: stop old, evict import, update books

The candidate starts BEFORE the old generation stops, so there
is a brief overlap window; single-writer ownership of the
DataSpace transfers during it.
"""

from __future__ import annotations

import asyncio
import sys
from copy import deepcopy
from pathlib import Path

from loguru import logger

from src.nan_itself.modules.model import (
    Module,
    ModuleRecord,
    ModuleState,
)


def _reject(facade, old, reason) -> None:
    """
    Cache a rejection so the sticky-error book reflects it.

    Without this, a rejected reload is skipped silently forever
    (fingerprint already updated, no error recorded).
    """
    facade._workspace_load_errors[
        Path(old.source).resolve()
    ] = reason


async def hot_reload(
    facade,
    *
    self,
    old: ModuleRecord,
    cls: type[Module],
    imported_name: str,
    fingerprint: tuple[int, int],
) -> None:
    logger.info(f'''Reloading Module {old.id}: generation {old.generation} -> {old.generation + 1}''')

    # --------------------------------------------------------------
    # Capture live private state.
    # --------------------------------------------------------------

    try:
        private_state = (
            old.instance.serialize_state()
        )

    except Exception as exc:
        logger.exception(f'''Hot reload rejected for Module {old.id}: failed to serialize current private state''')

        _reject(facade, old, exc)

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

    facade._bind_instance(
        candidate
    )

    try:
        candidate.instance.restore_state(
            deepcopy(
                private_state
            )
        )

    except Exception as exc:
        logger.exception(f'''Hot reload rejected for Module {old.id}: failed to restore private state''')

        _reject(facade, old, exc)

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

    await started.wait()

    # --------------------------------------------------------------
    # Reject failed candidates.
    # --------------------------------------------------------------

    if candidate.state is ModuleState.DOWN:
        logger.error(f'''Hot reload rejected for Module {old.id}: candidate failed during startup''')

        _reject(
            facade,
            old,
            RuntimeError("hot reload rejected: candidate failed to start"),
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
        logger.error(f'''Hot reload candidate entered unexpected state: {old.id} -> {candidate.state}''')

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

    facade.modules[
        old.id
    ] = candidate

    facade._rebuild_dependency_graph()

    try:
        facade._validate_dependency_graph()

    except Exception as exc:
        facade.modules[
            old.id
        ] = old

        facade._rebuild_dependency_graph()

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

        logger.exception(f'''Hot reload rejected for Module {old.id}: invalid dependency graph''')

        _reject(facade, old, exc)

        return

    # --------------------------------------------------------------
    # Commit.
    # --------------------------------------------------------------

    old_task = old.task

    try:
        await old.instance.stop()

    except Exception:
        logger.exception(f'''Old Module generation stop failed: {old.id}[generation={old.generation}]''')

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

    facade._workspace_fingerprints[
        Path(
            old.source
        ).resolve()
    ] = fingerprint

    facade._workspace_load_errors.pop(
        Path(
            old.source
        ).resolve(),
        None,
    )

    facade._wake.set()
