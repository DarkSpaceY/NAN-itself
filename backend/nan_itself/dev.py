"""
Development-time block list.

A developer can silence individual Modules, Tools or Skills for
local debugging by listing their names in config/dev.yaml. A
blocked source behaves exactly as if it did not exist: it is
never loaded, and an already-loaded one is unloaded by the next
scan. This works because every runtime discovers sources as
"candidate set minus already-loaded set" and unloads the
difference -- filtering the candidate set is therefore enough to
get both "never load" and "unload if already loaded".

Matching rules:

    - modules / tools: source-file stem
      builtin/modules/voice.py            -> "voice"
      builtin/tools/mcps/calculation.yaml -> "calculation"
    - skills: directory name
      builtin/skills/writing-modules      -> "writing-modules"

The file is re-read on every call. It is tiny, and reading it per
scan is exactly what makes edits take effect at runtime without a
restart. There is deliberately no long-lived cache.

A missing, malformed or invalid file is never fatal: it is
treated as an empty block list and a warning is logged. This is a
development aid; a typo in it must never be able to take the
agent down.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from nan_itself.utils import paths as _paths


# The source kinds a block list may target. Used to validate the
# `kind` argument of blocked().
_BLOCK_KINDS = frozenset(
    {
        "modules",
        "tools",
        "skills",
    }
)


class BlockConfig(BaseModel):
    """Names of sources to keep unloaded, one list per kind."""

    modules: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class DevSection(BaseModel):
    """The `dev:` mapping of the file."""

    block: BlockConfig = Field(default_factory=BlockConfig)


class DevConfig(BaseModel):
    """Contents of config/dev.yaml."""

    dev: DevSection = Field(default_factory=DevSection)


def load_dev_config(
    path: Path | None = None,
) -> DevConfig:
    """
    Read the development block list.

    `path` defaults to config/dev.yaml under the repository root;
    an explicit path is accepted for tests.

    A missing file yields an empty config. A malformed or invalid
    file is logged as a warning and also yields an empty config,
    so a broken file degrades to the previous (unblocked)
    behaviour instead of raising.
    """
    config_path = (
        path
        if path is not None
        else _paths.repo_root()
        / "config"
        / "dev.yaml"
    )

    if not config_path.exists():
        return DevConfig()

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"dev config must be a mapping: "
                f"{config_path}"
            )

        return DevConfig(**data)

    except Exception as exc:
        logger.warning(
            "Ignoring invalid dev config '{}': {}",
            config_path,
            exc,
        )

        return DevConfig()


def blocked(
    kind: str,
) -> frozenset[str]:
    """
    Return the blocked names for one source kind.

    `kind` is one of "modules", "tools" or "skills". The config
    file is re-read on every call, so a name added or removed
    while the process is running takes effect on the next scan.
    """
    if kind not in _BLOCK_KINDS:
        raise ValueError(
            f"Unknown block kind {kind!r}; "
            f"expected one of {sorted(_BLOCK_KINDS)}"
        )

    config = load_dev_config()

    return frozenset(
        getattr(
            config.dev.block,
            kind,
        )
    )
