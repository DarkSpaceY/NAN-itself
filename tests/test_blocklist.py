"""
Source block list (config/sources.yaml, nan_itself.blocklist).

Listing a Module / Tool / Skill name there makes the runtime
treat that source as absent. These tests pin the config contract
-- a missing or broken file degrades to an empty block list and
never raises -- and the candidate-set filtering that makes the
"unload if already loaded" behaviour fall out of the existing
discovery logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nan_itself import blocklist
from nan_itself.blocklist import load_blocklist


# ============================================================================
# load_blocklist
# ============================================================================


def test_missing_file_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    config = load_blocklist(tmp_path / "sources.yaml")

    assert config.block.modules == []
    assert config.block.tools == []
    assert config.block.skills == []


def test_parses_the_three_block_lists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.yaml"

    path.write_text(
        "block:\n"
        "  modules: [audio, vision]\n"
        "  tools: [calculation]\n"
        "  skills: [writing-modules]\n",
        encoding="utf-8",
    )

    config = load_blocklist(path)

    assert config.block.modules == [
        "audio",
        "vision",
    ]

    assert config.block.tools == [
        "calculation",
    ]

    assert config.block.skills == [
        "writing-modules",
    ]


def test_broken_yaml_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.yaml"

    # Unclosed flow sequence: yaml.safe_load raises.
    path.write_text(
        "block: [1, 2\n",
        encoding="utf-8",
    )

    config = load_blocklist(path)

    assert config.block.modules == []
    assert config.block.tools == []
    assert config.block.skills == []


def test_non_mapping_document_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.yaml"

    path.write_text(
        "- audio\n- vision\n",
        encoding="utf-8",
    )

    config = load_blocklist(path)

    assert config.block.modules == []


def test_invalid_field_type_yields_empty_block_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.yaml"

    # modules must be a list of names, not a scalar.
    path.write_text(
        "block:\n"
        "  modules: 42\n",
        encoding="utf-8",
    )

    config = load_blocklist(path)

    assert config.block.modules == []
    assert config.block.tools == []
    assert config.block.skills == []


def test_reload_picks_up_edits(
    tmp_path: Path,
) -> None:
    """The file is re-read per call: edits need no restart."""
    path = tmp_path / "sources.yaml"

    path.write_text(
        "block:\n  modules: [audio]\n",
        encoding="utf-8",
    )

    assert load_blocklist(
        path
    ).block.modules == ["audio"]

    path.write_text(
        "block:\n  modules: []\n",
        encoding="utf-8",
    )

    assert load_blocklist(
        path
    ).block.modules == []


# ============================================================================
# blocked()
# ============================================================================


def _point_repo_root_at(
    monkeypatch,
    root: Path,
) -> None:
    """Make the default config path resolve under `root`."""
    monkeypatch.setattr(
        blocklist._paths,
        "repo_root",
        lambda: root,
    )


def test_blocked_membership(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = tmp_path / "config"

    config_dir.mkdir()

    (config_dir / "sources.yaml").write_text(
        "block:\n"
        "  modules: [audio, vision]\n"
        "  tools: [calculation]\n"
        "  skills: [writing-modules]\n",
        encoding="utf-8",
    )

    _point_repo_root_at(monkeypatch, tmp_path)

    modules = blocklist.blocked("modules")

    assert isinstance(modules, frozenset)

    assert modules == frozenset(
        {"audio", "vision"}
    )

    assert "audio" in modules
    assert "voice" not in modules

    assert blocklist.blocked("tools") == frozenset(
        {"calculation"}
    )

    assert blocklist.blocked("skills") == frozenset(
        {"writing-modules"}
    )


def test_blocked_empty_without_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _point_repo_root_at(monkeypatch, tmp_path)

    assert blocklist.blocked("modules") == frozenset()
    assert blocklist.blocked("tools") == frozenset()
    assert blocklist.blocked("skills") == frozenset()


def test_blocked_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        blocklist.blocked("widgets")


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
