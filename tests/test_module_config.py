"""
utils/module_config.py: per-module YAML loading contract.

Each module owns config/modules/<id>.yaml validated against its
pydantic config model; missing or empty file = pure defaults, a
malformed file is a loud failure, and path fields resolve
repo-relative through resolve_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from nan_itself.utils import paths as _paths
from nan_itself.utils.module_config import (
    load_module_config,
    module_config_path,
    resolve_path,
)


class _Demo(BaseModel):
    device: int | str | None = None

    model_path: str = "models/demo/model.onnx"


def test_missing_file_yields_defaults(tmp_path: Path) -> None:
    cfg = load_module_config(
        "demo", _Demo, tmp_path / "demo.yaml"
    )

    assert cfg == _Demo()


def test_empty_yaml_yields_defaults(tmp_path: Path) -> None:
    path = tmp_path / "demo.yaml"

    path.write_text("", encoding="utf-8")

    assert load_module_config("demo", _Demo, path) == _Demo()


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    path = tmp_path / "demo.yaml"

    path.write_text(
        "device: 1\nmodel_path: /abs/m.onnx\n",
        encoding="utf-8",
    )

    cfg = load_module_config("demo", _Demo, path)

    assert cfg.device == 1

    assert cfg.model_path == "/abs/m.onnx"


def test_malformed_yaml_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "demo.yaml"

    path.write_text(
        "- just\n- a\n- list\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid module config"):
        load_module_config("demo", _Demo, path)


def test_resolve_path_relative_and_absolute() -> None:
    assert resolve_path("models/x.onnx") == (
        _paths.repo_root() / "models" / "x.onnx"
    )

    assert resolve_path("/abs/x.onnx") == Path("/abs/x.onnx")

    assert resolve_path("~/x.onnx").is_relative_to(
        Path.home()
    )


def test_module_config_path_is_repo_anchored() -> None:
    assert module_config_path("audio") == (
        _paths.repo_root()
        / "config"
        / "modules"
        / "audio.yaml"
    )
