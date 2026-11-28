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
| `mcp` | external MCP tool servers, configured in `services/mcp/config.yaml`; non-text results are saved under `services/mcp/data/` and reported as paths | `mcp list`, `mcp schema`, `mcp call` |
| `segv` | durable memory: every turn archived; an epoch opens with the last three turns, everything older is pulled | `segv state`, `segv turns`, `segv turn <seq>`, `segv search` |
| `plan` | my goal tree for long tasks: outcomes, not steps | `plan`, `plan add`, `plan done`, `plan drop` |
| `attach` | a separate reader for text and images: one workspace file plus one request in, one answer out, with no conversation attached | `attach <path> "<ask>"` |

`attach` is pull-only: nothing is injected, and each capability keeps
its own endpoint (`services/attach/config.yaml`), so a large text or an
image never enters my history. Use it only for content that actually has
to be read — a small file, a known line range or a `segv search` hit is
cheaper through the shell. A file is routed by its bytes: png, jpeg,
webp and gif go to the image capability, everything else to text. Text
is limited to 128,000 characters and images to 8 MB; anything larger is
refused with `too_large` instead of truncated. It only reads inside the
workspace, and an unconfigured capability fails with
`capability_unavailable` rather than falling back to the other endpoint.

Each service owns its own directory, config and logs; a service I
write belongs in `services/<name>/` too.

## Projects

none yet

## Notes

none yet
