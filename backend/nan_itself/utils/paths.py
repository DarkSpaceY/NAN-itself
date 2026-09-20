"""
Canonical filesystem anchors.

Every repo-relative location (data/, models/) resolves through
this module so the process never depends on its working
directory: launching nan from any cwd still writes inside the
repository. Generic overrides stay env-driven (NAN_DATA_DIR,
NAN_MODELS_DIR); feature-specific envs (NAN_AUDIO_SPEAKER_MODEL,
NAN_VISION_VLM_DIR, ...) keep precedence at their call sites.

Anchoring relies on the editable install: nan_itself.__file__
resolves into <repo>/src/nan_itself, so three levels up from
this file is the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repository root, anchored at this file's location."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """
    Root for all persistent module data (env: NAN_DATA_DIR).
    """
    override = os.getenv("NAN_DATA_DIR")

    return (
        Path(override).expanduser().resolve()
        if override
        else repo_root() / "data"
    )


def models_dir() -> Path:
    """
    Root for all model weights (env: NAN_MODELS_DIR).
    """
    override = os.getenv("NAN_MODELS_DIR")

    return (
        Path(override).expanduser().resolve()
        if override
        else repo_root() / "models"
    )
