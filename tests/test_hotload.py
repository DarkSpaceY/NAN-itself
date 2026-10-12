"""
Tests for the shared hot-reload primitives (utils/hotload.py):

    - detected_files exclusion rules (hidden, __pycache__, data/)
    - directory_fingerprint sensitivity (add/remove/edit; data/
      is inert)
    - folder package loading: vocabulary seeding, sibling
      imports, main.py-only entry extraction, and failure
      eviction leaving no sys.modules residue
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from nan_itself.utils.hotload import (
    detected_files,
    directory_fingerprint,
    evict_package,
    load_folder_package,
)

# ============================================================================
# detected_files / directory_fingerprint
# ============================================================================


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(text, encoding="utf-8")


def test_detected_files_exclusions(tmp_path: Path) -> None:
    _touch(tmp_path / "main.py")
    _touch(tmp_path / "helper.py")
    _touch(tmp_path / "data" / "weights.bin")
    _touch(tmp_path / ".hidden.py")
    _touch(tmp_path / "__pycache__" / "main.cpython-312.pyc")
    _touch(tmp_path / "sub" / ".venv" / "lib.py")

    files = detected_files(tmp_path)

    names = [path.name for path in files]

    assert "main.py" in names
    assert "helper.py" in names
    assert "weights.bin" not in names
    assert ".hidden.py" not in names
    assert "main.cpython-312.pyc" not in names
    assert "lib.py" not in names


def test_detected_files_custom_exclude_dirs(tmp_path: Path) -> None:
    _touch(tmp_path / "main.py")
    _touch(tmp_path / "data" / "keep.json")

    # Skills detect everything.
    assert (tmp_path / "data" / "keep.json") in set(detected_files(tmp_path, exclude_dirs=()))

    # Modules/tools exclude data/.
    assert (tmp_path / "data" / "keep.json") not in set(detected_files(tmp_path))


def test_fingerprint_detects_edit_add_remove(tmp_path: Path) -> None:
    _touch(tmp_path / "main.py", "v1")

    base = directory_fingerprint(tmp_path)

    # Edit.
    _touch(tmp_path / "main.py", "v2-longer")

    assert directory_fingerprint(tmp_path) != base

    after_edit = directory_fingerprint(tmp_path)

    # Add.
    _touch(tmp_path / "sibling.py", "s")

    assert directory_fingerprint(tmp_path) != after_edit

    after_add = directory_fingerprint(tmp_path)

    # Remove.
    (tmp_path / "sibling.py").unlink()

    assert directory_fingerprint(tmp_path) != after_add


def test_fingerprint_ignores_data_and_is_stable(tmp_path: Path) -> None:
    _touch(tmp_path / "main.py", "v1")

    base = directory_fingerprint(tmp_path)

    # Writes under data/ never change the fingerprint.
    _touch(tmp_path / "data" / "private.json", "{}")

    _touch(tmp_path / "data" / "models" / "w.bin", "weights")

    assert directory_fingerprint(tmp_path) == base

    # Repeated reads are stable.
    assert directory_fingerprint(tmp_path) == base


# ============================================================================
# Package loading
# ============================================================================


MODULE_MAIN = """import asyncio

from . import helper


class Thing(Module):
    size = helper.SIZE

    async def start(self):
        await asyncio.Event().wait()

    async def ask(self, turn):
        return helper.greet()
"""

MODULE_HELPER = """SIZE = 7


def greet():
    return "hi from helper"
"""


def _make_entity(tmp_path: Path, main: str, helper: str | None = None) -> Path:
    folder = tmp_path / "thing"

    folder.mkdir()

    (folder / "main.py").write_text(main, encoding="utf-8")

    if helper is not None:
        (folder / "helper.py").write_text(helper, encoding="utf-8")

    return folder


class _Base:
    async def start(self) -> None:  # pragma: no cover - shape only
        raise NotImplementedError


def test_load_folder_package_seeds_and_links_siblings(tmp_path: Path) -> None:
    folder = _make_entity(tmp_path, MODULE_MAIN, MODULE_HELPER)

    pkg_name, main = load_folder_package(
        folder,
        kind="module",
        seeds={"Module": _Base},
    )

    try:
        cls = main.Thing

        assert issubclass(cls, _Base)

        # Vocabulary was seeded into the sibling too.
        helper = sys.modules[f"{pkg_name}.helper"]

        assert helper.Module is _Base

        # Sibling import resolved through the package.
        assert cls.size == 7

    finally:
        evict_package(pkg_name)

    assert not any(name.startswith(pkg_name) for name in sys.modules)


def test_load_folder_package_failure_leaves_no_residue(tmp_path: Path) -> None:
    broken = """class Broken(Module):
    async def start(self):
        return syntax error
"""

    folder = _make_entity(tmp_path, broken)

    with pytest.raises(SyntaxError):
        load_folder_package(
            folder,
            kind="module",
            seeds={"Module": _Base},
        )

    assert not any(name.startswith("_nan_pkg_module_thing") for name in sys.modules)


def test_load_folder_package_requires_main(tmp_path: Path) -> None:
    folder = tmp_path / "empty"

    folder.mkdir()

    with pytest.raises(FileNotFoundError):
        load_folder_package(
            folder,
            kind="module",
            seeds={"Module": _Base},
        )


def test_data_directory_never_enters_the_package(tmp_path: Path) -> None:
    folder = _make_entity(tmp_path, MODULE_MAIN, MODULE_HELPER)

    _touch(folder / "data" / "payload.py", "raise RuntimeError('must not import')")

    pkg_name, main = load_folder_package(
        folder,
        kind="module",
        seeds={"Module": _Base},
    )

    try:
        with pytest.raises(ImportError):
            __import__(f"{pkg_name}.payload")

    finally:
        evict_package(pkg_name)
