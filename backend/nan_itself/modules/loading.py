"""
Module folder discovery and import.

One folder = one Module:

    <name>/
        main.py       entry; exactly one concrete Module subclass
        other.py      siblings, seeded with the same vocabulary
        config.yaml   optional (read by the Module itself)
        data/         never imported, never fingerprinted

Pure functions; no registry state.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from nan_itself.utils import hotload

from .model import (
    Module,
    Turn,
)

MAIN_FILENAME = "main.py"


def validate_module_class(
    cls: Any,
) -> None:
    requires = getattr(cls, "requires", ())

    if not isinstance(requires, tuple):
        raise TypeError(f"Module {cls.__name__}.requires must be tuple[str, ...]")

    if any(
        not isinstance(item, str) or not item
        for item in requires
    ):
        raise TypeError(f"Module {cls.__name__!r}.requires contains invalid ids")

    if len(set(requires)) != len(requires):
        raise ValueError(f"Module {cls.__name__!r}.requires contains duplicates")


def import_module_class(
    folder: Path,
) -> tuple[
    Any,
    str,
]:
    """
    Import one Module folder and extract its class.

    The folder is imported as a synthetic package seeded with the
    module vocabulary (Module, Turn), so main.py and every
    sibling work without any nan_itself imports. Returns the
    single concrete Module subclass defined in main.py -- classes
    defined in siblings are not entry candidates -- plus the
    package root name for later eviction from sys.modules.
    """
    pkg_name, main = hotload.load_folder_package(
        folder,
        kind="module",
        seeds={"Module": Module, "Turn": Turn},
    )

    try:
        candidates = [
            cls
            for _, cls in inspect.getmembers(main, inspect.isclass)
            if (
                cls.__module__ == main.__name__
                and issubclass(cls, Module)
                and cls is not Module
                and not inspect.isabstract(cls)
            )
        ]

    except BaseException:
        hotload.evict_package(pkg_name)

        raise

    if len(candidates) != 1:
        hotload.evict_package(pkg_name)

        raise RuntimeError(
            f"{folder / MAIN_FILENAME} must contain exactly one concrete Module; "
            f"found {len(candidates)}"
        )

    return (
        candidates[0],
        pkg_name,
    )
