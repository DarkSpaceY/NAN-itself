"""
Source block list (config/sources.yaml, nan_itself.utils.blocklist).

Listing a Module / Tool / Skill name there makes a discovery pass
treat that source as absent. These tests pin the lookup contract
-- a missing or broken file blocks nothing and never raises -- and
the candidate-set filtering each discovery layer does with it.
"""

from __future__ import annotations

from pathlib import Path

from nan_itself.utils import blocklist
from nan_itself.utils import paths

CONFIG_DIR = "config"


def _write_config(
    root: Path,
    body: str,
) -> Path:
    config_dir = root / CONFIG_DIR

    config_dir.mkdir(exist_ok=True)

    path = config_dir / "sources.yaml"

    path.write_text(body, encoding="utf-8")

    return path


def _point_repo_root_at(
    monkeypatch,
    root: Path,
) -> None:
    """Make the default config path resolve under `root`."""
    monkeypatch.setattr(
        paths,
        "repo_root",
        lambda: root,
    )


# ============================================================================
# blocked()
# ============================================================================


def test_missing_file_blocks_nothing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _point_repo_root_at(monkeypatch, tmp_path)

    assert blocklist.blocked("modules") == set()
    assert blocklist.blocked("tools") == set()
    assert blocklist.blocked("skills") == set()


def test_parses_the_three_lists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(
        tmp_path,
        "block:\n"
        "  modules: [audio, vision]\n"
        "  tools: [calculation]\n"
        "  skills: [writing-modules]\n",
    )

    _point_repo_root_at(monkeypatch, tmp_path)

    assert blocklist.blocked("modules") == {
        "audio",
        "vision",
    }

    assert blocklist.blocked("tools") == {"calculation"}
    assert blocklist.blocked("skills") == {"writing-modules"}


def test_broken_yaml_blocks_nothing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Unclosed flow sequence: yaml.safe_load raises.
    _write_config(tmp_path, "block: [1, 2\n")

    _point_repo_root_at(monkeypatch, tmp_path)

    assert blocklist.blocked("modules") == set()


def test_unexpected_shapes_block_nothing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _point_repo_root_at(monkeypatch, tmp_path)

    # Not a mapping at all.
    config = _write_config(tmp_path, "- audio\n")
    assert blocklist.blocked("modules") == set()

    # `block` present but not a mapping.
    config.write_text("block: 42\n", encoding="utf-8")
    assert blocklist.blocked("modules") == set()

    # A list where a name should be: treated as absent, not raised.
    config.write_text(
        "block:\n  modules: 42\n",
        encoding="utf-8",
    )
    assert blocklist.blocked("modules") == set()

    # Unknown kind.
    config.write_text(
        "block:\n  modules: [audio]\n",
        encoding="utf-8",
    )
    assert blocklist.blocked("widgets") == set()


def test_reload_picks_up_edits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The file is re-read per call: edits need no restart."""
    _point_repo_root_at(monkeypatch, tmp_path)

    config = _write_config(
        tmp_path,
        "block:\n  modules: [audio]\n",
    )

    assert blocklist.blocked("modules") == {"audio"}

    config.write_text(
        "block:\n  modules: []\n",
        encoding="utf-8",
    )

    assert blocklist.blocked("modules") == set()


# ============================================================================
# Discovery filtering (skills registry)
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


def _block(
    monkeypatch,
    names: set[str],
) -> None:
    """
    Pin the lookup so registry tests do not depend on the
    repository's own config/sources.yaml.
    """
    monkeypatch.setattr(
        blocklist,
        "blocked",
        lambda kind: names if kind == "skills" else set(),
    )


def test_registry_never_registers_blocked_skill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from nan_itself.skills.registry import SkillRegistry

    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")

    _block(monkeypatch, {"beta"})

    registry = SkillRegistry()
    registry.discover_root(tmp_path)

    assert set(registry.records) == {"alpha"}


def test_registry_unregisters_a_newly_blocked_skill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A name that becomes blocked is unloaded by the next scan."""
    from nan_itself.skills.registry import SkillRegistry

    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")

    _block(monkeypatch, set())

    registry = SkillRegistry()
    registry.discover_root(tmp_path)

    assert set(registry.records) == {"alpha", "beta"}

    _block(monkeypatch, {"beta"})

    registry.discover_root(tmp_path)

    assert set(registry.records) == {"alpha"}
