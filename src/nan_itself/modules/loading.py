"""
Workspace module discovery primitives.

Header sniffing, change fingerprints, dynamic import of a
workspace file and Module class validation. Pure functions.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from src.nan_itself.modules.model import (
    MODULE_HEADER,
    Module,
)


def has_module_header(
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


def fingerprint(
    path: Path,
) -> tuple[int, int]:
    stat = path.stat()

    return (
        stat.st_mtime_ns,
        stat.st_size,
    )


def validate_module_class(
    cls: Any,
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


def import_module_class(
    path: Path,
) -> tuple[
    Any,
    str,
    ModuleType,
]:
    """
    Import a workspace module file.

    Returns the single concrete Module subclass it defines,
    the synthetic module name it was imported under, and the
    module object itself.
    """
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

    # Compile from freshly read source instead of going through
    # the import system: the bytecode cache (__pycache__) keys on
    # coarse mtime + size, so an edit that keeps both (a one-
    # character change within the same timestamp tick) would
    # silently execute STALE code. Hot reload must never do that.
    try:
        source = path.read_text(
            encoding="utf-8"
        )

        code = compile(
            source,
            str(path),
            "exec",
        )

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    module = ModuleType(module_name)

    module.__file__ = str(path)

    sys.modules[
        module_name
    ] = module

    try:
        exec(
            code,
            module.__dict__,
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
