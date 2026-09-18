"""
Path-anchoring convention (utils/paths.py).

The whole repository must be launch-directory independent: from
ANY cwd, every repo-relative location (data/, models/, config/,
builtin/, workspace/) resolves inside the repository. These tests
re-run the anchoring contract under a foreign cwd so a future
cwd-relative default or a new parents[N] copy cannot sneak back
in unnoticed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

from nan_itself.utils import paths as _paths


REPO = Path(__file__).resolve().parents[1]


def _chdir_out(monkeypatch: Any, tmp_path: Path) -> None:
    """Park the process cwd far outside the repository."""
    monkeypatch.chdir(tmp_path)


# ============================================================================
# paths.py contract
# ============================================================================


def test_repo_root_is_stable_regardless_of_cwd(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _chdir_out(monkeypatch, tmp_path)

    assert _paths.repo_root() == REPO


def test_data_and_models_dirs_anchor_to_repo(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NAN_DATA_DIR", raising=False)

    monkeypatch.delenv("NAN_MODELS_DIR", raising=False)

    _chdir_out(monkeypatch, tmp_path)

    assert _paths.data_dir() == REPO / "data"

    assert _paths.models_dir() == REPO / "models"


# ============================================================================
# Runtime defaults under a foreign cwd
# ============================================================================


def test_facade_defaults_anchor_from_foreign_cwd(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from nan_itself.modules.runtime import Facade

    _chdir_out(monkeypatch, tmp_path)

    facade = Facade()

    assert facade.data_dir == REPO / "data" / "modules"

    assert (
        facade.builtin_modules_dir
        == REPO / "builtin" / "modules"
    )


def test_settings_yaml_found_from_foreign_cwd(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from nan_itself.config import load_settings

    _chdir_out(monkeypatch, tmp_path)

    settings = load_settings()

    with (REPO / "config" / "settings.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file) or {}

    expected = data.get("llm", {}).get("model")

    if expected:
        assert settings.llm.model == expected


def test_mcp_relative_cwd_anchors_to_repo(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from nan_itself.tools.mcp import load_yaml, parse_config

    _chdir_out(monkeypatch, tmp_path)

    path = REPO / "builtin" / "tools" / "mcps" / "files.yaml"

    spec = parse_config(load_yaml(path), path)[0]

    assert spec.cwd == str(REPO)


# ============================================================================
# Builtin modules (loading-time injection reproduced by hand)
# ============================================================================


def _load_builtin_module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"path_anchor_{name}",
        REPO / "builtin" / "modules" / f"{name}.py",
    )

    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    from nan_itself.modules.model import Module, Turn

    module.Module = Module

    module.Turn = Turn

    spec.loader.exec_module(module)

    return module


def test_builtin_modules_anchor_from_foreign_cwd(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("NAN_MEMORY_DIR", raising=False)

    monkeypatch.delenv("NAN_PLAN_DIR", raising=False)

    monkeypatch.delenv("NAN_VISION_FACES_REGISTRY", raising=False)

    monkeypatch.delenv("NAN_AUDIO_VOICES_REGISTRY", raising=False)

    _chdir_out(monkeypatch, tmp_path)

    memory = _load_builtin_module("memory")

    plan = _load_builtin_module("plan")

    vision = _load_builtin_module("vision")

    audio = _load_builtin_module("audio")

    assert (
        memory.MemoryModule().base
        == REPO / "data" / "databases" / "memory"
    )

    assert (
        plan.PlanModule().base
        == REPO / "data" / "databases" / "plan"
    )

    assert (
        vision.VisionModule().registry_path
        == REPO / "data" / "databases" / "vision" / "faces.json"
    )

    assert (
        audio.AudioModule().registry_path
        == REPO / "data" / "databases" / "audio" / "voices.json"
    )
