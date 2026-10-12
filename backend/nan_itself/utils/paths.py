"""
Canonical filesystem anchors.

Every repo-relative location (data/, config/) resolves through
this module so the process never depends on its working
directory: launching nan from any cwd still writes inside the
repository. Locations are fixed derivatives of the repository
root -- there are no overrides. Core configuration lives in
config/settings.yaml; each Module owns config.yaml inside its
own folder and loads it itself.

Anchoring relies on the editable install: nan_itself.__file__
resolves into <repo>/backend/nan_itself, so three levels up from
this file is the repository root.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Repository root, anchored at this file's location."""
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Root for runtime data that is not owned by a Module folder."""
    return repo_root() / "data"


def persona_path() -> Path:
    """The Agent's persona prompt file."""
    return repo_root() / "workspace" / "persona.md"
