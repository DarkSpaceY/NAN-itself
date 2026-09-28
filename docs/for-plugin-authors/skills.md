# Skills

A skill is a capability package the agent discovers by metadata and
opens on demand. A skill is **one directory** containing a `SKILL.md`
file plus optional bundled resources. The runtime
(`backend/nan_itself/skills/`) discovers skills, validates them, exposes
a lightweight metadata catalog, and serves their resources through the
`invoke_skill` verb.

This document describes the skill format and runtime surface. See
[../how-to/add-a-skill.md](../how-to/add-a-skill.md) for how to create
one.

## Format

A skill directory contains:

- `SKILL.md` — required. YAML frontmatter followed by a Markdown body.
- Optional resource folders. The protocol convention is `scripts/`,
  `references/`, `assets/`, but these are conventions, not the
  exhaustive surface (see [Resources](#resources)).

### `SKILL.md` frontmatter

The frontmatter must start at the very first line with `---` and be
terminated by the next `---` line; it must parse as a YAML mapping.
Two fields are required and validated by `parsing.py` against the
limits in `model.py`:

| Field | Constraint |
|---|---|
| `name` | A string matching `^[a-z0-9]+(-[a-z0-9]+)*$` (kebab-case), at most 64 characters. |
| `description` | A non-empty string of at most 1024 characters. |

`name` is what the runtime keys on; `description` is what the agent sees
in the metadata catalog. The Markdown body is read only when the agent
opens the skill; it is not part of the metadata.

The live examples all follow the convention that the directory name
matches `name` (e.g. `builtin/skills/writing-skills/SKILL.md` →
`writing-skills`), but the loader keys on the frontmatter field, not the
directory name.

## Sources and discovery

Skills live in two parallel roots scanned with identical semantics:

| Root | Contents |
|---|---|
| `builtin/skills/` | Shipped skills |
| `workspace/skills/` | User territory |

Discovery rules (`registry.py`):

- A directory is a skill if it contains a `SKILL.md`; a root that is
  itself a skill directory is also accepted.
- Directories whose name starts with `_` are ignored.
- One name may be claimed by one source directory only; a second source
  claiming a live name fails registration and leaves the old skill
  intact.
- An unchanged, broken skill directory is not retried until its contents
  change.

Change detection uses a per-directory fingerprint that folds every file
in the directory (summed mtimes and sizes plus the file count), so edits
to bundled resources — and additions or removals — are detected, not
just edits to `SKILL.md`.

## Runtime surface

`SkillRuntime` (`runtime.py`) exposes:

| Method | Behaviour |
|---|---|
| `discover()` | One-shot synchronous discovery at boot (metadata files only). |
| `refresh()` | Re-scans every root; called at every turn start. |
| `catalog()` | Returns `SkillMetadata` for every skill, sorted by name. |
| `names()` | Returns the skill names. |
| `get_metadata(name)` | Returns one `SkillMetadata`, or `None`. |
| `resource_paths(name)` | Returns the skill's bundled files grouped by top-level directory. |
| `invoke(name, path, args)` | Executes a script or returns a resource as text. |

`SkillMetadata` carries `name`, `description`, `source` (the path to
`SKILL.md`) and the raw `frontmatter` mapping.

### Progressive disclosure

Metadata lookups never load skill bodies. `catalog()`, `names()` and
`get_metadata()` read only the frontmatter; the Markdown body and every
resource file are read lazily when the agent invokes them. This keeps the
per-turn `refresh()` scan cheap.

### Hot reload

`refresh()` re-scans both roots on every turn start using the same logic
as discovery: a changed directory is re-registered transactionally (the
candidate is parsed before any registry state is mutated), a removed
directory is unregistered, and a name claimed by the candidate is
validated before the old record is dropped. A `generation` counter is
maintained per skill name and survives reloads.

## Resources

`resource_paths(name)` groups a skill's bundled files by their top-level
directory:

- The protocol folders `scripts`, `references`, `assets` are listed
  first when present.
- Any other folder, and any root-level file other than `SKILL.md`, is
  listed under its own group.
- `SKILL.md` itself is never a resource.
- Hidden entries (dot-prefixed path segments) and `__pycache__` are
  excluded.

`invoke(name, path, args)` resolves `path` relative to the skill root and
rejects any path that escapes it. Dispatch depends on the top-level
folder:

- A path under `scripts/` is **executed** as a subprocess with the given
  arguments. The interpreter is chosen by suffix — `.py` →
  `sys.executable`, `.sh` / `.bash` → `bash`, `.js` → `node`; any other
  suffix is run directly, which requires a shebang and the executable
  bit. The subprocess runs with `cwd` set to the skill root, and its
  combined stdout/stderr is returned. The run is bounded by
  `skills.script_timeout` (300 s default); a non-zero exit code is
  appended to the returned text.
- Any other resource is **returned as text** (read with UTF-8,
  replacement on decode errors).

Both the executed output and the returned text are truncated to
`skills.resource_char_limit` (100 000 characters default). These two
settings live in `config/settings.yaml`; see
[configuration.md](configuration.md).

## Verb surface

Skills are reached only through the agent's verb triple
(`backend/nan_itself/agent/verbs.py`):

| Verb | Behaviour |
|---|---|
| `list_skills` | Returns `name: description` for every skill. |
| `show_skill` | Returns one skill's metadata and its resource paths, grouped. |
| `invoke_skill` | Takes a skill name, a resource path and optional args; executes a script or returns a resource as text. |

An unknown skill name surfaces as an error string built from
`UnknownSkillError`.

## Live sources

The shipped skills are directories under
[`builtin/skills/`](../../builtin/skills/): `writing-modules`,
`writing-skills` and `writing-tools`. Each documents how to author one
part of the framework and ships its own resources — a `references/` file
and a `scripts/check_*.py` validator.
