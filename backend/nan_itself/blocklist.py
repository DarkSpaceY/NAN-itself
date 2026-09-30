"""
Source block list (config/sources.yaml).

Listing a Module, Tool or Skill name under `block:` makes the
runtime treat that source as if it did not exist: it is never
loaded, and an already-loaded one is unloaded by the next scan.
This works because every runtime discovers sources as
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
restart. There is deliberately no long-lived cache -- in
particular it is NOT part of the cached Settings object.

A missing, malformed or invalid file is never fatal: it is
treated as an empty block list and a warning is logged, so a typo
degrades to "nothing blocked" instead of taking the agent down.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from nan_itself.utils import paths as _paths

CONFIG_FILENAME = "sources.yaml"

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


class BlockListConfig(BaseModel):
    """Contents of config/sources.yaml."""

    block: BlockConfig = Field(default_factory=BlockConfig)


def load_blocklist(
    path: Path | None = None,
) -> BlockListConfig:
    """
    Read the source block list.

    `path` defaults to config/sources.yaml under the repository
    root; an explicit path is accepted for tests.

    A missing file yields an empty block list. A malformed or
    invalid file is logged as a warning and also yields an empty
    block list, so a broken file degrades to "nothing blocked"
    instead of raising.
    """
    config_path = (
        path
        if path is not None
        else _paths.repo_root()
        / "config"
        / CONFIG_FILENAME
    )

    if not config_path.exists():
        return BlockListConfig()

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"sources config must be a mapping: "
                f"{config_path}"
            )

        return BlockListConfig(**data)

    except Exception as exc:
        logger.warning(
            "Ignoring invalid sources config '{}': {}",
            config_path,
            exc,
        )

        return BlockListConfig()


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

    config = load_blocklist()

    return frozenset(
        getattr(
            config.block,
            kind,
        )
    )
