# Workspace

This directory is my home on this machine. It survives turns, epochs
and restarts, so what I leave here is what the next run inherits.

## Layout

| Path | Kind | Notes |
|---|---|---|
| `AGENT.md` | index | this file; update it when the layout changes |
| `autostart.sh` | shipped | boot script, runs inside the sandbox |
| `bin/` | shipped | executables callable by bare name |
| `services/` | shipped | services and pull tools, launched at boot |
| `projects/` | mine | one directory per project, each with its own notes |
| `notes/` | mine | long-term notes that span projects |
| `home/` | regenerable | my `$HOME`: caches, configs, dotfiles |
| `logs/` | regenerable | boot and service logs |
| `.tmp/` | temporary | scratch files, safe to delete at any time |
| `.nan/` | runtime | job logs owned by the core |

## Rules

- Nothing loose at the root except this file and `autostart.sh`.
- Unsure where something belongs: `projects/<name>/`.
- One-off work goes to `.tmp/`, never the root.
- `services/` and `bin/` ship with the install: I may add files, but
  changing an existing one is a code change for the user to review.
- Safe to delete: `home/.cache/`, `logs/`, `.tmp/`, `.nan/`.
  Everything else needs a reason.

## Services

Six services live under `services/` and start with the core:

| Service | What it gives me | How I reach it |
|---|---|---|
| `system` | my body: clock, battery, focus window, host state | injected each turn; nothing to call |
| `network` | my nerves: connections, reachability, public identity | injected each turn; nothing to call |
| `mcp` | external MCP tool servers, configured in `services/mcp/config.yaml` | `mcp list`, `mcp schema`, `mcp call` |
| `segv` | durable memory: every turn archived; an epoch opens with the last three turns, everything older is pulled | `segv state`, `segv turns`, `segv turn <seq>`, `segv search` |
| `plan` | my goal tree for long tasks: outcomes, not steps | `plan`, `plan add`, `plan done`, `plan drop` |
| `read` | a separate reader: one file path plus one request in, one answer out, with no conversation attached | `reader <path> "<ask>"` |

`read` is pull-only: nothing is injected, and it keeps its own LLM
endpoint (`services/read/config.yaml`), so a long text never enters my
history. Use it only for a large text that actually has to be read —
a small file, a known line range or a `segv search` hit is cheaper
through the shell. It accepts one file of up to 128,000 characters and
refuses anything larger with `too_large` instead of truncating it, and
it only reads inside the workspace.

Each service owns its own directory, config and logs; a service I
write belongs in `services/<name>/` too.

## Projects

none yet

## Notes

- `notes/capabilities.md` — what I can do, verified 2026-11-27; includes the gaps I have not fixed.
