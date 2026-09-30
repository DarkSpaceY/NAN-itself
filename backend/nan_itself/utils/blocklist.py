"""
Source block list (config/sources.yaml).

A discovery pass consults this to treat listed sources as absent:
`block.modules` / `block.tools` name source-file stems, and
`block.skills` names skill directories, that must not be loaded.
Filtering the candidate set is enough for both "never load" and
"unload if already loaded", because every discovery pass
reconciles "candidate set minus already-loaded set".

The file is read per call -- it is tiny, and that is exactly what
makes an edit take effect on the next discovery scan without a
restart. A missing or broken file blocks nothing and never raises.
"""

from __future__ import annotations

import yaml
from loguru import logger

from . import paths as _paths

CONFIG_FILENAME = "sources.yaml"


def blocked(
    kind: str,
) -> set[str]:
    """
    Names of sources to keep unloaded for one kind.

    `kind` is a key under `block:` -- "modules", "tools" or
    "skills". Anything missing or malformed yields an empty set.
    """
    path = (
        _paths.repo_root()
        / "config"
        / CONFIG_FILENAME
    )

    if not path.exists():
        return set()

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(file) or {}

        block = (
            data.get("block")
            if isinstance(data, dict)
            else None
        )

        names = (
            block.get(kind)
            if isinstance(block, dict)
            else None
        )

        return (
            set(names)
            if isinstance(names, list)
            else set()
        )

    except Exception as exc:
        logger.warning(
            "Ignoring invalid sources config '{}': {}",
            path,
            exc,
        )

        return set()
