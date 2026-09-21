# Architecture

This document describes how NAN-itself is put together, section by
section, following the repository layout. It mirrors the code as of the
current development state; interfaces may still change.

## Repository layout

```
NAN-itself/
├── backend/nan_itself/   # Framework source (Python package `nan_itself`)
├── builtin/              # Shipped tools / modules / skills
├── workspace/            # User-editable tools / modules / skills + persona
├── config/               # settings.yaml, searxng.yml
├── frontend/             # Demo web UI (not covered here; likely to be rebuilt)
├── infraend/             # Planned infra backend (not covered here)
├── models/               # Downloaded model weights (runtime-managed)
├── docs/                 # This documentation
└── tests/                # Contract and integration tests
```

## backend/nan_itself/

The Python package. Flat modules at its root, one subpackage per
subsystem. Import discipline: everything anchors to the package location
(`utils/paths.py`), never to the process working directory.

### app.py — composition root

On startup it:

1. loads `config/settings.yaml` (`config.py`),
2. anchors all repo-relative directories through
   [`utils/paths.py`](../backend/nan_itself/utils/paths.py),
3. starts the tool provider runtime,
4. wires the builtin modules and skills,
5. starts the gateway (FastAPI/uvicorn) which serves the API and the web UI.

### agent/ — the turn loop

- **Turn is the sole record of a round** (`model.py`). A `Turn` holds
  `persona`, `history` (the snapshot the model saw), `messages` (what
  this round produced), `usage`, `finish_reason` and `model`. There is
  no separate persistent history.
- **Snapshot derivation** (`engine.py`). The next round's snapshot is
  derived as `last_turn.history + last_turn.messages`. When the
  character budget is exceeded the snapshot is truncated to an empty
  tuple — the model then sees only the system prompt plus the newest
  observation and the chain restarts from there.
- **Core state** (`core.py`). The core holds a single `last_turn`
  reference, replaced after each successful run and kept on failure, so
  a failed round can be inspected as-is.
- **Subagents** (`runtime.py`) run the same engine inside a `while
  True` loop with no round limit; the loop only exits when the `finish`
  tool sets the finished state. The `finish` tool is visible only at
  depth > 0 and a finish report must be non-empty.
- **Verbs** (`verbs.py`). Every interface face is reached through a
  `list_*` / `show_*` / `invoke_*` verb triple: tools, skills, and
  module channels all follow the same mental model — list enumerates,
  show inspects, invoke acts.

### tools/ — provider runtime

[`runtime.py`](../backend/nan_itself/tools/runtime.py)
(`ProviderRuntime`) reconciles sources across two parallel directory
layouts — builtin (`builtin/tools/`) and workspace
(`workspace/tools/`) — with identical hot-reload semantics:

- **One file = one provider.** An MCP YAML file declares exactly one
  stdio server (`name` / `command` / `args` / `env` / `cwd`; a relative
  `cwd` is repo-anchored, never process-cwd-anchored). A local Python
  file contains exactly one `LocalToolProvider` subclass, marked with a
  `# @tool` header in its first 20 lines.
- **Reload transaction.** Replacement workers are first started as
  unregistered candidates; only after a candidate is fully connected is
  it committed into the live tables and the old generation asked to
  stop. Old and new generations of the same provider name can therefore
  coexist during the handoff.
- **Bounded calls.** Every tool call (local or MCP) is wrapped in a
  timeout (300 s by default, `providers.tool_timeout` in settings) and
  surfaces timeouts as error results instead of hanging the agent loop.
- **Local tools** derive their JSON schema from type annotations via
  `@tool`-decorated methods; return values are JSON-serialized into the
  tool result, raised exceptions become error results.
- A dedicated supervisor task reaps dead MCP workers and reschedules
  their sources with backoff; a degraded provider never kills the
  runtime.

### modules/ — ambient state and channels

Builtin modules (`builtin/modules/`) provide ambient state the agent
can query. The machinery here (`runtime.py` = the Facade) owns:

- dependency graph, lifecycle supervision, retry and hot reload;
- DataSpace ownership: a module publishes, dependents read detached
  snapshots (`deps.py`);
- per-turn delivery: `on_turn()` observation and `query()` projection
  (`model.py` defines `Turn` and `Module`);
- private state persistence via `serialize_state()` (`persistence.py`);
- **channels** (`action.py`): modules that opt in via `ActionSurface`
  expose write-only data slots the model reaches through the
  `list_channels` / `show_channels` / `invoke_channels` verbs; see
  [design/reactive-modules.md](design/reactive-modules.md).

Module-side contracts: capture/inference work runs on daemon threads
with interval gating; the agent-facing surface is a pure `query()`
projection; provisioning failures (e.g. missing model weights) raise
out of `start()` -- the Facade marks the module DOWN with the error
and restarts it with backoff, so a module with missing weights comes
up loudly failed and revives once the weights land (all weights are
provisioned in `start()`, before any loop runs).

### skills/ — capability packages

A skill is a directory with a `SKILL.md` (YAML frontmatter) plus
optional `scripts/`, `references/`, `assets/` folders (`model.py`).
The runtime (`runtime.py`) discovers skills from the same parallel
builtin/workspace roots at boot, re-scans at every turn start (shared
hot-reload logic), and applies progressive disclosure: metadata lookups
never load bodies; invoking a script executes it, other resources
return as text.

### gateway and events

`gateway.py` serves the HTTP API on `gateway.host:gateway.port`
(default `127.0.0.1:8765`) and bridges to the web UI in `frontend/`.
`events.py` implements the in-process pub/sub bus with bounded history
and subscriber queues.

## builtin/ and workspace/

Two parallel provider layouts, scanned with identical semantics —
`builtin/` ships with the repo, `workspace/` is user territory
(deleting a source file removes its provider; edits hot-reload).

| Slot | Builtin | Workspace |
|---|---|---|
| Tools | `builtin/tools/` (`local/` Python, `mcps/` YAML) | `workspace/tools/` (same shape, starts empty) |
| Modules | `builtin/modules/` (audio, vision, network, system, inbox) | `workspace/modules/` |
| Skills | `builtin/skills/` | `workspace/skills/` |

`workspace/persona.md` holds the user-editable persona;
`workspace/desktop/` is reserved for the agent's "virtual desktop"
(planned).

## config/

`settings.yaml` is the single configuration file (loaded by
`config.py`); `searxng.yml` configures the bundled SearxNG instance
used by the search tool. See [configuration.md](configuration.md).

## tests/

Contract tests mirror the reload/consistency guarantees
(`test_reload_*.py`, `test_turn_consistency.py`, `test_layering.py`),
plus integration tests that run the real agent loop against isolated
module/tool directories.

## Security model

- The only inbound listener is the gateway, bound to the loopback
  interface by default (verified by the test suite).
- All repo-relative paths (`data/`, `models/`, `config/`, `builtin/`)
  resolve through `utils/paths.py`; the process never depends on its
  working directory and writes no state outside the repo directories.
- Trusted subprocesses (MCP servers, skill scripts) inherit the user
  environment and are unrestricted by design; review a tool config
  before adding it.
