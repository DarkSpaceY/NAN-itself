"""
Folder-based module lifecycle against a live Facade:

    - writes under data/ never trigger a reload
    - sibling/config edits trigger a hot reload that keeps the
      DataSpace and private state across generations
    - deleting main.py unloads the module while the folder (and
      its data/) survives; re-adding main.py registers a fresh
      instance that restores the persisted state
    - legacy data/modules/{private,dataspace}/<id>.json files are
      migrated into the folder on first restore

Every Facade is pointed at tmp dirs only (never the real
builtin/modules -- real modules would start).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nan_itself.modules.runtime import Facade
from nan_itself.utils import paths as _paths
from nan_itself.utils.hotload import directory_fingerprint


def run(coro):
    return asyncio.run(coro)


def _facade(tmp_path: Path, monkeypatch) -> Facade:
    monkeypatch.setattr(_paths, "repo_root", lambda: tmp_path)

    return Facade()


def _write_entry(folder: Path, text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "main.py").write_text(text, encoding="utf-8")


def _scan(facade: Facade) -> None:
    run(facade._scan_module_root(facade.workspace_modules_dir))


MAIN_V1 = """import asyncio

COUNTER = 1


class Example(Module):
    def __init__(self):
        self.counter = COUNTER

    def serialize_state(self):
        return {"counter": self.counter}

    def restore_state(self, state):
        self.counter = state["counter"]

    async def start(self):
        self.data.publish({"v": COUNTER})
        await asyncio.Event().wait()

    async def ask(self, turn):
        return None
"""

MAIN_V2 = MAIN_V1.replace("COUNTER = 1", "COUNTER = 2")

BROKEN = "class Example(Module):\n    async def start(self"


def test_data_writes_do_not_trigger_reload(tmp_path, monkeypatch):
    facade = _facade(tmp_path, monkeypatch)

    folder = tmp_path / "workspace" / "modules" / "example"

    _write_entry(folder, MAIN_V1)

    _scan(facade)

    record = facade.modules.get("example")

    assert record is not None

    # Runtime writes under data/.
    (folder / "data").mkdir(exist_ok=True)

    (folder / "data" / "private.json").write_text("{}", encoding="utf-8")

    (folder / "data" / "blobs.bin").write_text("payload", encoding="utf-8")

    _scan(facade)

    assert facade.modules.get("example") is record

    assert record.generation == 0


def test_sibling_and_config_edits_hot_reload_keeping_state(tmp_path, monkeypatch):
    facade = _facade(tmp_path, monkeypatch)

    folder = tmp_path / "workspace" / "modules" / "example"

    _write_entry(folder, MAIN_V1)

    (folder / "helper.py").write_text("VALUE = 'a'\n", encoding="utf-8")

    _scan(facade)

    old = facade.modules["example"]

    assert old.generation == 0

    # Sibling edit.
    (folder / "helper.py").write_text("VALUE = 'b'\n", encoding="utf-8")

    _scan(facade)

    new = facade.modules["example"]

    assert new is not old

    assert new.generation == 1

    # The DataSpace object survives the generation swap.
    assert new.data is old.data


def test_main_py_removal_unloads_and_readd_restores_state(tmp_path, monkeypatch):
    facade = _facade(tmp_path, monkeypatch)

    folder = tmp_path / "workspace" / "modules" / "example"

    _write_entry(folder, MAIN_V1)

    _scan(facade)

    record = facade.modules["example"]

    # Advance live state and persist it (as _remove_record does).
    record.instance.counter = 7

    record.data.publish({"v": 7})

    facade._save_record_state(record)

    # Delete the entry; the folder and its data/ survive.
    (folder / "main.py").unlink()

    _scan(facade)

    assert facade.modules.get("example") is None

    assert (folder / "data" / "private.json").is_file()

    # Re-add: a fresh instance restores the persisted state.
    _write_entry(folder, MAIN_V1)

    _scan(facade)

    readded = facade.modules["example"]

    assert readded is not None

    assert readded.instance.counter == 7

    assert readded.data.snapshot() == {"v": 7}


def test_broken_intermediate_keeps_old_generation(tmp_path, monkeypatch):
    facade = _facade(tmp_path, monkeypatch)

    folder = tmp_path / "workspace" / "modules" / "example"

    _write_entry(folder, MAIN_V1)

    _scan(facade)

    old = facade.modules["example"]

    _write_entry(folder, BROKEN)

    _scan(facade)

    assert facade.modules["example"] is old

    # Restore a valid source: reload succeeds.
    _write_entry(folder, MAIN_V2)

    _scan(facade)

    assert facade.modules["example"] is not old

    assert facade.modules["example"].generation == 1


def test_legacy_state_files_are_migrated_into_folder(tmp_path, monkeypatch):
    facade = _facade(tmp_path, monkeypatch)

    folder = tmp_path / "workspace" / "modules" / "example"

    _write_entry(folder, MAIN_V1)

    # Pre-seed the legacy layout the old Facade wrote.
    legacy_private = tmp_path / "data" / "modules" / "private"

    legacy_dataspace = tmp_path / "data" / "modules" / "dataspace"

    legacy_private.mkdir(parents=True)

    legacy_dataspace.mkdir(parents=True)

    (legacy_private / "example.json").write_text(
        '{"counter": 42}',
        encoding="utf-8",
    )

    (legacy_dataspace / "example.json").write_text(
        '{"legacy": true}',
        encoding="utf-8",
    )

    _scan(facade)

    record = facade.modules["example"]

    assert record.instance.counter == 42

    assert record.data.snapshot() == {"legacy": True}

    # The files now live in the folder; the legacy ones are gone.
    assert (folder / "data" / "private.json").is_file()

    assert (folder / "data" / "dataspace.json").is_file()

    assert not (legacy_private / "example.json").exists()

    assert not (legacy_dataspace / "example.json").exists()


def test_directory_fingerprint_matches_runtime_exclusion(tmp_path, monkeypatch):
    """
    The runtime scans with the same fingerprint helper the tests
    use; this pins the exclusion contract (data/ excluded).
    """
    facade = _facade(tmp_path, monkeypatch)

    folder = tmp_path / "workspace" / "modules" / "example"

    _write_entry(folder, MAIN_V1)

    _scan(facade)

    recorded = facade._source_tracker.fingerprints[folder.resolve()]

    assert recorded == directory_fingerprint(folder, exclude_dirs=("data",))
