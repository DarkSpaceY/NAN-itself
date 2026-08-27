from __future__ import annotations

from pathlib import Path

import pytest

from src.nan_itself.skills import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    SkillRuntime,
    UnknownSkillError,
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
        UnknownSkillError,
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
        in runtime._registry.errors
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
    assert skill_dir in runtime._registry.errors


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
    assert skill_dir in runtime._registry.errors


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
    assert skill_dir in runtime._registry.errors


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
    assert skill_dir in runtime._registry.errors


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
    assert skill_dir in runtime._registry.errors


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
        in runtime._registry.errors
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

# ============================================================================
# refresh / hot reload semantics
# ============================================================================


def make_skill_dir(skills: Path, name: str, description: str = "demo", body: str = "# body"):
    skill_dir = skills / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_refresh_picks_up_new_skills(tmp_path):
    skills = tmp_path / "skills"

    runtime = SkillRuntime(workspace_skills=skills)
    runtime.discover()

    assert runtime.names() == ()

    make_skill_dir(skills, "fresh-one")

    runtime.refresh()

    assert runtime.names() == ("fresh-one",)


def test_refresh_updates_body_and_generation_on_change(tmp_path):
    skills = tmp_path / "skills"

    skill_dir = make_skill_dir(skills, " evolving".strip())

    runtime = SkillRuntime(workspace_skills=skills)
    runtime.discover()

    first = runtime.activate("evolving")

    import time

    time.sleep(0.01)

    make_skill_dir(
        skills,
        "evolving",
        body="# rewritten body v2",
    )

    runtime.refresh()

    second = runtime.activate("evolving")

    assert second.generation == first.generation + 1
    assert "rewritten body v2" in second.instructions


def test_refresh_drops_removed_skills(tmp_path):
    skills = tmp_path / "skills"

    skill_dir = make_skill_dir(skills, "gone-soon")

    runtime = SkillRuntime(workspace_skills=skills)
    runtime.discover()

    assert "gone-soon" in runtime.names()

    import shutil

    shutil.rmtree(skill_dir)

    runtime.refresh()

    assert "gone-soon" not in runtime.names()


def test_unchanged_broken_skill_is_not_retried_until_fixed(tmp_path):
    skills = tmp_path / "skills"

    skill_dir = skills / "flaky"
    skill_dir.mkdir(parents=True)

    # Missing description: registration fails, error cached.
    (skill_dir / "SKILL.md").write_text(
        "---\nname: flaky\n---\n\nbody\n",
        encoding="utf-8",
    )

    runtime = SkillRuntime(workspace_skills=skills)
    runtime.discover()

    assert "flaky" not in runtime.names()
    assert skill_dir.resolve() in runtime._registry.errors

    runtime.refresh()

    # Unchanged: still broken, still skipped.
    assert skill_dir.resolve() in runtime._registry.errors

    # Fixed on disk: next refresh recovers.
    import time

    time.sleep(0.01)

    make_skill_dir(skills, "flaky", description="now fine")

    runtime.refresh()

    assert "flaky" in runtime.names()
    assert skill_dir.resolve() not in runtime._registry.errors


def test_duplicate_name_across_workspace_dirs_is_rejected(tmp_path):
    skills = tmp_path / "skills"

    make_skill_dir(skills, "twin", description="first wins")

    runtime = SkillRuntime(workspace_skills=skills)
    runtime.discover()

    twin_dir = skills / "twin-clone"
    twin_dir.mkdir(parents=True)

    (twin_dir / "SKILL.md").write_text(
        "---\nname: twin\ndescription: second loses\n---\n\nbody\n",
        encoding="utf-8",
    )

    runtime.refresh()

    # The first source survives untouched.
    assert runtime.get_metadata("twin").description == "first wins"
    assert twin_dir.resolve() in runtime._registry.errors


def test_names_get_metadata_and_catalog_ordering(tmp_path):
    skills = tmp_path / "skills"

    make_skill_dir(skills, "zeta", description="last")
    make_skill_dir(skills, "alpha", description="first")

    runtime = SkillRuntime(workspace_skills=skills)
    runtime.discover()

    assert runtime.names() == ("alpha", "zeta")

    catalog = runtime.catalog()

    assert [m.name for m in catalog] == ["alpha", "zeta"]

    alpha = runtime.get_metadata("alpha")

    assert alpha is not None
    assert alpha.description == "first"
    assert runtime.get_metadata("missing") is None


def test_missing_builtin_root_is_silently_skipped(tmp_path):
    runtime = SkillRuntime(
        workspace_skills=tmp_path / "ws",
        builtin_skills=[tmp_path / "does-not-exist"],
    )

    runtime.discover()  # must not raise

    assert runtime.names() == ()


def test_unknown_skill_error_is_part_of_validation_family():
    from src.nan_itself.skills import (
        SkillValidationError,
        UnknownSkillError,
    )

    assert issubclass(UnknownSkillError, SkillValidationError)
    assert issubclass(SkillValidationError, ValueError)
