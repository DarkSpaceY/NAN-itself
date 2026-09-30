"""
Development-time block list (config/dev.yaml, nan_itself.dev).

The dev block list is a debugging aid: listing a Module / Tool /
Skill name there makes the runtime treat that source as absent.
These tests pin the config contract -- a missing or broken file
degrades to an empty block list and never raises -- and the
candidate-set filtering that makes the "unload if already
loaded" behaviour fall out of the existing discovery logic.
"""

from __future__ import annotations

from pathlib import Path

from nan_itself import dev
from nan_itself.dev import load_dev_config


# ============================================================================
# load_dev_config
# ============================================================================


def test_missing_file_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    config = load_dev_config(tmp_path / "dev.yaml")

    assert config.dev.block.modules == []
    assert config.dev.block.tools == []
    assert config.dev.block.skills == []


def test_parses_the_three_block_lists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dev.yaml"

    path.write_text(
        "dev:\n"
        "  block:\n"
        "    modules: [audio, vision]\n"
        "    tools: [calculation]\n"
        "    skills: [writing-modules]\n",
        encoding="utf-8",
    )

    config = load_dev_config(path)

    assert config.dev.block.modules == [
        "audio",
        "vision",
    ]

    assert config.dev.block.tools == [
        "calculation",
    ]

    assert config.dev.block.skills == [
        "writing-modules",
    ]


def test_broken_yaml_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dev.yaml"

    # Unclosed flow sequence: yaml.safe_load raises.
    path.write_text(
        "dev: [1, 2\n",
        encoding="utf-8",
    )

    config = load_dev_config(path)

    assert config.dev.block.modules == []
    assert config.dev.block.tools == []
    assert config.dev.block.skills == []


def test_non_mapping_document_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dev.yaml"

    path.write_text(
        "- audio\n- vision\n",
        encoding="utf-8",
    )

    config = load_dev_config(path)

    assert config.dev.block.modules == []


def test_invalid_field_type_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dev.yaml"

    # modules must be a list of names, not a scalar.
    path.write_text(
        "dev:\n"
        "  block:\n"
        "    modules: 42\n",
        encoding="utf-8",
    )

    config = load_dev_config(path)

    assert config.dev.block.modules == []
    assert config.dev.block.tools == []
    assert config.dev.block.skills == []


def test_reload_picks_up_edits(
    tmp_path: Path,
) -> None:
    """The file is re-read per call: edits need no restart."""
    path = tmp_path / "dev.yaml"

    path.write_text(
        "dev:\n  block:\n    modules: [audio]\n",
        encoding="utf-8",
    )

    assert load_dev_config(
        path
    ).dev.block.modules == ["audio"]

    path.write_text(
        "dev:\n  block:\n    modules: []\n",
        encoding="utf-8",
    )

    assert load_dev_config(
        path
    ).dev.block.modules == []


# ============================================================================
# blocked()
# ============================================================================


def _point_repo_root_at(
    monkeypatch,
    root: Path,
) -> None:
    """Make the default config path resolve under `root`."""
    monkeypatch.setattr(
        dev._paths,
        "repo_root",
        lambda: root,
    )


def test_blocked_membership(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = tmp_path / "config"

    config_dir.mkdir()

    (config_dir / "dev.yaml").write_text(
        "dev:\n"
        "  block:\n"
        "    modules: [audio, vision]\n"
        "    tools: [calculation]\n"
        "    skills: [writing-modules]\n",
        encoding="utf-8",
    )

    _point_repo_root_at(monkeypatch, tmp_path)

    modules = dev.blocked("modules")

    assert isinstance(modules, frozenset)

    assert modules == frozenset(
        {"audio", "vision"}
    )

    assert "audio" in modules
    assert "voice" not in modules

    assert dev.blocked("tools") == frozenset(
        {"calculation"}
    )

    assert dev.blocked("skills") == frozenset(
        {"writing-modules"}
    )


def test_blocked_empty_without_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _point_repo_root_at(monkeypatch, tmp_path)

    assert dev.blocked("modules") == frozenset()
    assert dev.blocked("tools") == frozenset()
    assert dev.blocked("skills") == frozenset()


# ============================================================================
# Candidate-set filtering (skills registry)
# ============================================================================


def _write_skill(
    root: Path,
    name: str,
) -> None:
    directory = root / name

    directory.mkdir()

    (directory / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: a test skill\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )


def test_registry_skip_never_registers(
    tmp_path: Path,
) -> None:
    from nan_itself.skills.registry import (
        SkillRegistry,
    )

    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")

    registry = SkillRegistry()

    registry.discover_root(
        tmp_path,
        skip=frozenset({"beta"}),
    )

    assert set(registry.records) == {"alpha"}


def test_registry_skip_unregisters_loaded(
    tmp_path: Path,
) -> None:
    """A name that becomes skipped is unloaded by the next scan."""
    from nan_itself.skills.registry import (
        SkillRegistry,
    )

    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")

    registry = SkillRegistry()

    registry.discover_root(tmp_path)

    assert set(registry.records) == {
        "alpha",
        "beta",
    }

    registry.discover_root(
        tmp_path,
        skip=frozenset({"beta"}),
    )

    assert set(registry.records) == {"alpha"}
