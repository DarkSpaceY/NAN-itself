---
name: writing-skills
description: >-
  How to write a NAN-itself Skill package: a directory with SKILL.md (YAML
  frontmatter name + description, Markdown body) plus optional scripts/,
  references/, assets/ folders. Covers the validation contract, the script
  interpreter table, progressive disclosure, and hot reload. Use when the
  user asks to create, fix, or extend a skill in builtin/skills/ or
  workspace/skills/.
---

# Writing a Skill

A Skill is **one directory** in `builtin/skills/<name>/` (shipped) or
`workspace/skills/<name>/` (user territory) containing a `SKILL.md`.
Skills are hot-reloaded: re-scanned at every turn start, registered
transactionally, and unregistered when the directory is deleted.

## SKILL.md contract

```
---
name: my-skill
description: One or two sentences saying WHAT it does and WHEN to use it.
---

Markdown body = the instructions the agent follows when it reads
this skill. Keep the body lean; push details into references/.
```

- `name`: kebab-case (`^[a-z0-9]+(-[a-z0-9]+)*$`), max 64 chars, must
  match the pattern exactly; unique across builtin + workspace roots
  (a duplicate name fails registration and the old skill stays intact).
- `description`: non-empty, max 1024 chars. This is what the agent sees
  in the metadata catalog — write it to make correct triggering likely.
- Frontmatter must start at the very first line with `---` and be
  terminated by a second `---` line.

## Resource folders (all optional)

| Folder        | Semantics |
|---|---|
| `scripts/`    | Executable. `.py` → `sys.executable`, `.sh`/`.bash` → `bash`, `.js` → `node`, others run directly (need shebang + exec bit). 300 s timeout; stdout+stderr returned, truncated to 100k chars. Scripts run with cwd = the skill root. |
| `references/` | Documentation read on demand (text, same truncation). |
| `assets/`     | Other bundled files, read as text on demand. |

Progressive disclosure: metadata lookups (`list_skills` /
`show_skill`) never load the body; invoking a script executes it,
any other resource is returned as text. Hidden files (dot-prefixed)
are excluded from resources.

Skill scripts are trusted subprocesses (inherited environment,
unrestricted). Keep them stdlib-only where possible so they run
anywhere.

## Checklist

- [ ] Directory name matches the `name` field (convention)
- [ ] Frontmatter at line 1, terminated, `name` kebab-case ≤64,
      `description` ≤1024 and non-empty
- [ ] Body gives step-by-step instructions, not prose
- [ ] Scripts are executable and exit non-zero on failure with a
      useful message on stdout/stderr

Validate before finishing:

```bash
uv run python builtin/skills/writing-skills/scripts/check_skill.py <skill-dir-or-SKILL.md>
```

Details: [references/skill-template.md](references/skill-template.md)
(skeleton to copy from).
