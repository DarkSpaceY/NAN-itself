# Add a skill

This guide adds a Skill to NAN-itself. A skill is a directory with a
`SKILL.md` plus optional bundled resources; the agent discovers it by
metadata and opens it on demand. Skills are hot-reloaded and re-scanned
every turn, so no restart is needed.

To add a skill to the shipped set, create the directory under
`builtin/skills/`; to keep it in your own territory, create it under
`workspace/skills/`. For the full format and runtime contract, see
[../reference/skills.md](../reference/skills.md).

## Create the directory

Create one directory per skill, named after the skill:

```
builtin/skills/<name>/
├── SKILL.md          # required
├── scripts/          # optional: executable helpers
├── references/       # optional: docs read on demand
└── assets/           # optional: other bundled files
```

## Write `SKILL.md`

Start the file with YAML frontmatter delimited by `---` on the very
first line:

```
---
name: my-skill
description: One or two sentences saying WHAT it does and WHEN to use it.
---

Markdown body = the instructions the agent follows when it reads
this skill. Keep the body lean; push details into references/.
```

- `name` must be kebab-case (`^[a-z0-9]+(-[a-z0-9]+)*$`) and at most 64
  characters. Give the directory the same name.
- `description` must be non-empty and at most 1024 characters. This is
  what the agent sees in the catalog — write it so correct triggering is
  likely.
- The frontmatter must start at line 1 and be terminated by the next
  `---` line, and must parse as a YAML mapping.

## Add resources

Put files in `scripts/`, `references/` and/or `assets/` as needed. These
three names are conventions, not the exhaustive surface: any other folder
or root-level file (other than `SKILL.md`) is also listed as a resource.
Hidden files and `__pycache__` are excluded.

Files under `scripts/` are **executed** when the agent invokes them, with
the interpreter chosen by suffix — `.py` → `sys.executable`, `.sh` /
`.bash` → `bash`, `.js` → `node`; any other suffix runs directly and
needs a shebang plus the executable bit. Scripts run with the skill root
as their working directory. Keep them stdlib-only where possible so they
run anywhere.

## Verify the skill

Validate the `SKILL.md` contract with the builtin skill's checker (it
does not execute scripts):

```bash
uv run python builtin/skills/writing-skills/scripts/check_skill.py builtin/skills/<name>
```

The checker exits `0` when every skill passes and `1` otherwise. It
validates the frontmatter, the `name` pattern and length, the
`description` length, and warns about a directory name that does not
match `name`, hidden files under resource folders, and `scripts/` files
whose suffix is unknown and that lack the executable bit.

Then confirm the loader sees it: the runtime re-scans every turn, so a
valid new skill appears through `list_skills` (and can be inspected with
`show_skill`) within a few seconds.

The `writing-skills` skill also ships a skeleton to copy from at
`builtin/skills/writing-skills/references/skill-template.md`.

## Related

- [../reference/skills.md](../reference/skills.md) — the full skill
  subsystem reference.
- [develop.md](develop.md) — the conventions a skill must follow.
