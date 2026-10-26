---
name: installing-capabilities
description: >-
  How to find an open-source project, evaluate it, and install it onto
  yourself as a usable capability. Covers search, evaluation criteria,
  the two install paths (workspace-contained vs system-wide), sandbox
  constraints, verification, and recording what you installed. Use
  when you need a capability you don't have and suspect someone has
  already built it.
---

# Installing a Capability

Your shell is unrestricted in what it can run, but the sandbox limits
where you can write: **workspace and temp only** (reads and network
are open). Everything you install must respect that, or go through
the user.

## 1. Search before building

Before writing anything yourself, assume it exists:

- GitHub search / web search: `<thing> cli`, `<thing> tui`, `awesome-<domain>`
- Homebrew: `brew search <thing>` — it may still write its cache;
  point `HOMEBREW_CACHE` at `$NAN_WORKSPACE` or temp, or go through
  the user.
- PyPI/npm: `uv tool` / `npm` registries list similar names — the
  sandbox only allows writes under `$NAN_WORKSPACE` and temp, so
  point `UV_CACHE_DIR`/`UV_TOOL_DIR` and `npm_config_cache` there
  too; their defaults live in your home and are denied.

Prefer a maintained CLI over a library you'd have to wrap, prefer a
library over a script you'd have to write.

## 2. Evaluate (don't skip this)

- **License**: MIT/Apache/BSD — fine. GPL — fine for personal use,
  note it. No license — walk away.
- **Pulse**: last commit, open issue ratio, release cadence. A tool
  that hasn't moved in two years still works — until it doesn't.
- **Weight**: one static binary < one script + runtime < one service.
  Every dependency you adopt is one you maintain.
- **Surface**: does it phone home, need keys, want root? Read the
  install section, not just the README headline.

## 3. Install — pick the path that fits the sandbox

**Path A: workspace-contained (you can do this alone).**

- Static binaries / single scripts: download into
  `workspace/bin/`, `chmod +x`, done. Prefer releases over
  building from source.
- Python tools: create the venv inside the workspace
  (`uv venv workspace/<name>/.venv`), install into it, wrap the
  invocation in a two-line script under `workspace/bin/`.
- Anything source-built: build in workspace or temp, keep artifacts
  in workspace.

**Path B: system-wide (brew, npm -g, anything outside workspace).**
The sandbox denies those writes — surface it to the user instead:
say exactly what you want installed, from which source, and why.
The user runs one command; you verify afterward.

## 4. Verify, then record

Run the tool on a real task, not `--help`. Confirm output, exit
codes, and that it works under the sandbox (no hidden writes outside
workspace/temp — those will fail at the worst time).

Then record the capability in your notes file (you keep one —
location is yours to choose): name, what it's for, where it lives,
invocation examples, gotchas discovered. An installed tool you
forget about is dead weight.

## Before finishing

- [ ] Tool runs under the sandbox on a real task (not just `--help`)
- [ ] Binary/scripts executable, location stable, no absolute paths
      into temp
- [ ] License noted; system-wide requests (if any) answered by the user
- [ ] Capability recorded in your notes with a working invocation
