"""
Crash-safe persistence for module state.

Two artifacts per module id, written atomically:

    <data_dir>/private/<id>.json     serialize_state() payload
    <data_dir>/dataspace/<id>.json   published DataSpace snapshot

A corrupt file only blocks its own module, never the others.
"""

from __future__ import annotations

import json
import os
from loguru import logger
from pathlib import Path
from typing import Any



def ensure_data_dirs(
    private_dir: Path,
    dataspace_dir: Path,
) -> None:
    private_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataspace_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def private_state_path(
    private_dir: Path,
    module_id: str,
) -> Path:
    return (
        private_dir
        / f"{module_id}.json"
    )


def dataspace_path(
    dataspace_dir: Path,
    module_id: str,
) -> Path:
    return (
        dataspace_dir
        / f"{module_id}.json"
    )


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


def load_dataspace_state(
    record,
    dataspace_dir: Path,
) -> bool:
    path = dataspace_path(
        dataspace_dir,
        record.id,
    )

    if not path.exists():
        return True

    try:
        value = read_json_file(path)

        if not isinstance(value, dict):
            raise TypeError(
                "DataSpace file must contain "
                f"a JSON object: {path}"
            )

        record.data.publish(value)

        logger.info(f'''Restored DataSpace: {record.id} revision={record.data.revision}''')

        return True

    except Exception:
        logger.exception(f'''Failed to restore DataSpace for Module {record.id}''')

        return False


def load_private_state(
    record,
    private_dir: Path,
) -> bool:
    path = private_state_path(
        private_dir,
        record.id,
    )

    if not path.exists():
        return True

    try:
        state = read_json_file(path)

        record.instance.restore_state(state)

        logger.info(f'''Restored private state: {record.id}''')

        return True

    except Exception:
        logger.exception(f'''Failed to restore private state for Module {record.id}''')

        return False


def restore_record_state(
    record,
    *,
    private_dir: Path,
    dataspace_dir: Path,
) -> None:
    load_dataspace_state(record, dataspace_dir)

    load_private_state(record, private_dir)


def save_record_state(
    record,
    *,
    private_dir: Path,
    dataspace_dir: Path,
) -> None:
    private_state = (
        record.instance.serialize_state()
    )

    atomic_write_json(
        private_state_path(private_dir, record.id),
        private_state,
    )

    atomic_write_json(
        dataspace_path(dataspace_dir, record.id),
        record.data.dump(),
    )
