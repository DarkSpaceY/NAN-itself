# Workspace

This directory is my home on this machine. It survives turns, epochs
and restarts, so what I leave here is what the next run inherits.

## Layout

| Path | Kind | Notes |
|---|---|---|
| `AGENT.md` | index | this file; update it when the layout changes |
| `autostart.sh` | shipped | boot script, runs inside the sandbox |
| `bin/` | shipped | executables callable by bare name |
| `services/` | shipped | injecting services, launched at boot |
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

Four services live under `services/` and start with the core:

| Service | What it gives me | How I reach it |
|---|---|---|
| `system` | my body: clock, battery, focus window, host state | injected each turn; nothing to call |
| `network` | my nerves: connections, reachability, public identity | injected each turn; nothing to call |
| `mcp` | external MCP tool servers, configured in `services/mcp/config.yaml` | `mcp list`, `mcp schema`, `mcp call` |
| `segv` | durable memory: every turn archived, recalled after a context clear | `segv state`, `segv turns`, `segv grep`, `segv search` |

Each service owns its own directory, config and logs; a service I
write belongs in `services/<name>/` too.

## Projects

none yet

## Notes

none yet
