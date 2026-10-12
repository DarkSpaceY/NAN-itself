"""
Crash-safe persistence for module state.

Two artifacts per module, written atomically inside the
module's own folder:

    <home>/data/private.json     serialize_state() payload
    <home>/data/dataspace.json   published DataSpace snapshot

A corrupt file only blocks its own module, never the others.

Legacy layouts (data/modules/{private,dataspace}/<id>.json) are
migrated into the folder on first restore, so an update from the
old flat layout keeps its state.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

PRIVATE_FILENAME = "private.json"

DATASPACE_FILENAME = "dataspace.json"


def private_state_path(
    data_dir: Path,
) -> Path:
    return data_dir / PRIVATE_FILENAME


def dataspace_path(
    data_dir: Path,
) -> Path:
    return data_dir / DATASPACE_FILENAME


def read_json_file(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def atomic_write_json(
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

    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(encoded)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temp_path,
            path,
        )

    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def _migrate_legacy_file(
    legacy: Path,
    target: Path,
) -> None:
    """
    Move one legacy state file into the module folder (best
    effort): only when the legacy file exists, the target does
    not, and both sit on the same filesystem (os.replace).
    """
    if not legacy.is_file() or target.exists():
        return

    try:
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.replace(
            legacy,
            target,
        )

        logger.info(
            "Migrated legacy Module state: {} -> {}",
            legacy,
            target,
        )

    except OSError:
        logger.exception(
            "Failed to migrate legacy Module state: {}",
            legacy,
        )


def migrate_legacy_state(
    module_id: str,
    data_dir: Path,
    *,
    legacy_private_dir: Path,
    legacy_dataspace_dir: Path,
) -> None:
    _migrate_legacy_file(
        legacy_private_dir / f"{module_id}.json",
        private_state_path(data_dir),
    )

    _migrate_legacy_file(
        legacy_dataspace_dir / f"{module_id}.json",
        dataspace_path(data_dir),
    )


def load_dataspace_state(
    record,
) -> bool:
    path = dataspace_path(record.instance.data_dir)

    if not path.exists():
        return True

    try:
        value = read_json_file(path)

        if not isinstance(value, dict):
            raise TypeError(f"DataSpace file must contain a JSON object: {path}")

        record.data.publish(value)

        logger.info(f"""Restored DataSpace: {record.id} revision={record.data.revision}""")

        return True

    except Exception:
        logger.exception(f"""Failed to restore DataSpace for Module {record.id}""")

        return False


def load_private_state(
    record,
) -> bool:
    path = private_state_path(record.instance.data_dir)

    if not path.exists():
        return True

    try:
        state = read_json_file(path)

        record.instance.restore_state(state)

        logger.info(f"""Restored private state: {record.id}""")

        return True

    except Exception:
        logger.exception(f"""Failed to restore private state for Module {record.id}""")

        return False


def restore_record_state(
    record,
) -> None:
    load_dataspace_state(record)

    load_private_state(record)


def save_record_state(
    record,
) -> None:
    data_dir = record.instance.data_dir

    private_state = record.instance.serialize_state()

    atomic_write_json(
        private_state_path(data_dir),
        private_state,
    )

    atomic_write_json(
        dataspace_path(data_dir),
        record.data.snapshot(),
    )
