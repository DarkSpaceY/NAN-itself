from __future__ import annotations

from pathlib import Path

import pytest

from src.nan_itself.skills.facade import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    SkillRuntime,
)


def write_skill(
    root: Path,
    name: str,
    description: str,
    body: str = "# Skill",
    *,
    scripts: dict[str, str] | None = None,
    references: dict[str, str] | None = None,
    assets: dict[str, str] | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
---

{body}
""",
        encoding="utf-8",
    )

    for directory, files in (
        ("scripts", scripts),
        ("references", references),
        ("assets", assets),
    ):
        if not files:
            continue

        target = skill_dir / directory
        target.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filename, content in files.items():
            path = target / filename
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            path.write_text(
                content,
                encoding="utf-8",
            )

    return skill_dir


# ============================================================================
# Discovery / catalog
# ============================================================================


def test_workspace_skill_discovery(tmp_path):
    skills = tmp_path / "workspace" / "skills"

    write_skill(
        skills,
        "python-debugging",
        "Debug Python programs.",
        body="""
Use tests and targeted inspection.

1. Run the relevant tests.
2. Inspect the failure.
3. Fix the smallest valid change.
""",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert runtime.names() == (
        "python-debugging",
    )

    metadata = runtime.get_metadata(
        "python-debugging"
    )

    assert metadata is not None
    assert metadata.name == "python-debugging"
    assert metadata.description == (
        "Debug Python programs."
    )
    assert metadata.origin == "workspace"


def test_catalog_does_not_load_skill_body(tmp_path):
    skills = tmp_path / "workspace" / "skills"

    write_skill(
        skills,
        "research",
        "Perform research tasks.",
        body="A very important full instruction body.",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    catalog = runtime.catalog()

    assert len(catalog) == 1
    assert catalog[0].name == "research"
    assert not hasattr(
        catalog[0],
        "instructions",
    )


def test_skill_activation_loads_full_instructions(tmp_path):
    skills = tmp_path / "workspace" / "skills"

    write_skill(
        skills,
        "research",
        "Perform research tasks.",
        body="""
# Research workflow

1. Find authoritative sources.
2. Compare evidence.
3. Summarize findings.
""",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    skill = runtime.activate(
        "research"
    )

    assert skill.name == "research"

    assert skill.instructions == (
        "# Research workflow\n\n"
        "1. Find authoritative sources.\n"
        "2. Compare evidence.\n"
        "3. Summarize findings."
    )


def test_skill_resources_are_progressively_loaded_as_paths(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = write_skill(
        skills,
        "python",
        "Work on Python code.",
        body="Use the bundled resources when necessary.",
        scripts={
            "run.py": "print('run')",
            "nested/check.py": "print('check')",
        },
        references={
            "patterns.md": "# Patterns",
        },
        assets={
            "template.txt": "hello",
        },
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    skill = runtime.activate(
        "python"
    )

    assert skill.scripts == (
        (
            skill_dir
            / "scripts"
            / "nested"
            / "check.py"
        ).resolve(),
        (
            skill_dir
            / "scripts"
            / "run.py"
        ).resolve(),
    )

    assert skill.references == (
        (
            skill_dir
            / "references"
            / "patterns.md"
        ).resolve(),
    )

    assert skill.assets == (
        (
            skill_dir
            / "assets"
            / "template.txt"
        ).resolve(),
    )


def test_unknown_skill_cannot_be_activated(tmp_path):
    runtime = SkillRuntime(
        workspace_skills=tmp_path / "skills",
    )

    runtime.discover()

    with pytest.raises(
        KeyError,
        match="Unknown Skill",
    ):
        runtime.activate("missing")


# ============================================================================
# Workspace validation
# ============================================================================


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Python-Debugging",
        "python_debugging",
        "python debugging",
        "-python",
        "python-",
        "python--debugging",
        "你好",
    ],
)
def test_invalid_workspace_skill_is_ignored(
    tmp_path,
    name,
):
    skills = tmp_path / "workspace" / "skills"
    skill_dir = skills / "skill"

    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Test Skill.
---

# Test
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert runtime.names() == ()

    assert (
        skill_dir
        in runtime._workspace_errors
    )


def test_workspace_skill_name_length_limit_is_ignored(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    name = "a" * (
        MAX_NAME_LENGTH + 1
    )

    skill_dir = skills / "skill"
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Test Skill.
---

# Test
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert runtime.names() == ()
    assert skill_dir in runtime._workspace_errors


def test_workspace_skill_description_length_limit_is_ignored(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    description = "a" * (
        MAX_DESCRIPTION_LENGTH + 1
    )

    skill_dir = skills / "long-description"
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        f"""---
name: long-description
description: {description}
---

# Test
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert runtime.names() == ()
    assert skill_dir in runtime._workspace_errors


def test_missing_frontmatter_workspace_skill_is_ignored(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = skills / "broken"
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        "# No frontmatter",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert runtime.names() == ()
    assert skill_dir in runtime._workspace_errors


def test_missing_name_workspace_skill_is_ignored(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = skills / "broken"
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        """---
description: Missing name.
---

# Test
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert runtime.names() == ()
    assert skill_dir in runtime._workspace_errors


def test_missing_description_workspace_skill_is_ignored(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = skills / "broken"
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        """---
name: broken
---

# Test
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert runtime.names() == ()
    assert skill_dir in runtime._workspace_errors


# ============================================================================
# Builtin Skills
# ============================================================================


def test_builtin_skill_can_be_discovered(tmp_path):
    builtin_root = (
        tmp_path
        / "builtin-skills"
    )

    write_skill(
        builtin_root,
        "core",
        "Core agent behavior.",
    )

    runtime = SkillRuntime(
        workspace_skills=tmp_path / "workspace",
        builtin_skills=(builtin_root,),
    )

    runtime.discover()

    metadata = runtime.get_metadata(
        "core"
    )

    assert metadata is not None
    assert metadata.origin == "builtin"


def test_builtin_skill_root_can_be_single_skill_directory(
    tmp_path,
):
    skill_root = (
        tmp_path
        / "core"
    )

    skill_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_root / "SKILL.md").write_text(
        """---
name: core
description: Core agent behavior.
---

# Core
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=tmp_path / "workspace",
        builtin_skills=(skill_root,),
    )

    runtime.discover()

    assert runtime.names() == (
        "core",
    )


def test_workspace_skill_cannot_override_builtin(
    tmp_path,
):
    builtin_root = (
        tmp_path
        / "builtin"
        / "skills"
    )

    workspace_root = (
        tmp_path
        / "workspace"
        / "skills"
    )

    write_skill(
        builtin_root,
        "core",
        "Builtin core behavior.",
    )

    workspace_core = write_skill(
        workspace_root,
        "core",
        "Malicious replacement.",
    )

    runtime = SkillRuntime(
        workspace_skills=workspace_root,
        builtin_skills=(builtin_root,),
    )

    runtime.discover()

    metadata = runtime.get_metadata(
        "core"
    )

    assert metadata is not None
    assert metadata.origin == "builtin"

    assert (
        workspace_core
        in runtime._workspace_errors
    )


# ============================================================================
# Hot reload / workspace changes
# ============================================================================


def test_workspace_skill_reload_increments_generation(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = write_skill(
        skills,
        "research",
        "Research tasks.",
        body="Version one.",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    first = runtime.activate(
        "research"
    )

    assert first.generation == 0
    assert first.instructions == "Version one."

    skill_file = (
        skill_dir / "SKILL.md"
    )

    skill_file.write_text(
        """---
name: research
description: Research tasks.
---

Version two.
""",
        encoding="utf-8",
    )

    runtime.refresh()

    second = runtime.activate(
        "research"
    )

    assert second.generation == 1
    assert second.instructions == "Version two."


def test_workspace_skill_can_be_removed(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = write_skill(
        skills,
        "temporary",
        "Temporary skill.",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    assert "temporary" in runtime.names()

    skill_file = (
        skill_dir / "SKILL.md"
    )

    skill_file.unlink()

    # Remove the directory after SKILL.md.
    skill_dir.rmdir()

    runtime.refresh()

    assert "temporary" not in runtime.names()


# ============================================================================
# Frontmatter compatibility
# ============================================================================


def test_frontmatter_extra_fields_are_preserved(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = skills / "research"
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        """---
name: research
description: Research tasks.
license: Apache-2.0
metadata:
  author: test
  version: "1.0"
---

# Research
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    skill = runtime.activate(
        "research"
    )

    assert skill.frontmatter["license"] == (
        "Apache-2.0"
    )

    assert skill.frontmatter["metadata"] == {
        "author": "test",
        "version": "1.0",
    }


def test_allowed_tools_frontmatter_is_preserved(
    tmp_path,
):
    skills = tmp_path / "workspace" / "skills"

    skill_dir = skills / "python"
    skill_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (skill_dir / "SKILL.md").write_text(
        """---
name: python
description: Work on Python.
allowed-tools: Bash pytest Read
---

# Python workflow
""",
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        workspace_skills=skills,
    )

    runtime.discover()

    skill = runtime.activate(
        "python"
    )

    assert skill.frontmatter["allowed-tools"] == (
        "Bash pytest Read"
    )