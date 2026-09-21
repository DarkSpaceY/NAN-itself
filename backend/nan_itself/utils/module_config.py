"""
Module-private configuration loading.

Each Module owns exactly one YAML file:
config/modules/<module_id>.yaml. The module declares a pydantic
config model whose defaults are the shipped constants; the YAML
(when present) overrides them and is validated at construction
time -- a malformed module config is a loud construction failure,
never a silent half default. Modules read their config in
__init__, so a hot reload (instance rebuild) re-reads the file.

Path-valued fields follow one convention (resolve_path):
absolute paths pass through unchanged, anything else resolves
against the repository root, so configs stay repo-relative and
launch-directory independent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from nan_itself.utils.paths import repo_root

T = TypeVar("T", bound=BaseModel)


def module_config_path(module_id: str) -> Path:
    """Config file location for one module id."""
    return repo_root() / "config" / "modules" / f"{module_id}.yaml"


def load_module_config(
    module_id: str,
    model: type[T],
    path: Path | None = None,
) -> T:
    """
    Validate config/modules/<module_id>.yaml against the
    module's pydantic model. Missing or empty file = pure
    defaults. `path` override exists for tests.
    """
    config_path = (
        path if path is not None else module_config_path(module_id)
    )

    if not config_path.is_file():
        return model()

    data = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    if data is None:
        return model()

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid module config YAML: {config_path}"
        )

    return model(**data)


def resolve_path(value: str | Path) -> Path:
    """
    Absolute passes through; relative resolves against the
    repository root.
    """
    resolved = Path(value).expanduser()

    if resolved.is_absolute():
        return resolved

    return repo_root() / resolved
