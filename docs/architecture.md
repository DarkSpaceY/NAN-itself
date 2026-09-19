# Architecture

This document describes how NAN-itself is put together. It mirrors the code
as of the current development state; interfaces may still change.

## Composition

`src/nan_itself/app.py` is the composition root. On startup it:

1. loads `config/settings.yaml` (`config.py`),
2. anchors all repo-relative directories through
   [`src/nan_itself/utils/paths.py`](../src/nan_itself/utils/paths.py),
3. starts the tool provider runtime,
4. wires the builtin modules and skills,
5. starts the gateway (FastAPI/uvicorn) which serves the API and the web UI.

## Agent core

The agent is a turn-based loop:

- **Turn is the sole record of a round.** A `Turn` holds `persona`,
  `history` (the snapshot the model saw), `messages` (what this round
  produced), `usage`, `finish_reason` and `model`. There is no separate
  persistent history.
- **Snapshot derivation.** The next round's snapshot is derived as
  `last_turn.history + last_turn.messages`. When the character budget is
  exceeded the snapshot is truncated to an empty tuple — the model then
  sees only the system prompt plus the newest observation and the chain
  restarts from there.
- **Core state.** The core holds a single `last_turn` reference, replaced
  after each successful run and kept on failure, so a failed round can be
  inspected as-is.
- **Subagents** run the same engine inside a `while True` loop with no
  round limit; the loop only exits when the `finish` tool sets the
  finished state. The `finish` tool is visible only at depth > 0 and a
  finish report must be non-empty.

## Tool runtime

[`src/nan_itself/tools/runtime.py`](../src/nan_itself/tools/runtime.py)
(`ProviderRuntime`) reconciles sources across two parallel directory
layouts — builtin (`builtin/tools/`) and workspace
(`workspace/tools/`) — with identical hot-reload semantics:

- **One file = one provider.** An MCP YAML file declares exactly one stdio
  server (`name` / `command` / `args` / `env` / `cwd`; a relative `cwd` is
  repo-anchored, never process-cwd-anchored). A local Python file contains
  exactly one `LocalToolProvider` subclass, marked with a `# @tool` header
  in its first 20 lines.
- **Reload transaction.** Replacement workers are first started as
  unregistered candidates; only after a candidate is fully connected is it
  committed into the live tables and the old generation asked to stop. Old
  and new generations of the same provider name can therefore coexist
  during the handoff.
- **Bounded calls.** Every tool call (local or MCP) is wrapped in a
  timeout (300 s by default, `providers.tool_timeout` in settings) and
  surfaces timeouts as error results instead of hanging the agent loop.
- **Local tools** derive their JSON schema from type annotations via
  `@tool`-decorated methods; return values are JSON-serialized into the
  tool result, raised exceptions become error results.
- A dedicated supervisor task reaps dead MCP workers and reschedules
  their sources with backoff; a degraded provider never kills the runtime.

## Modules

Builtin modules (`builtin/modules/`) provide ambient state the agent can
query:

- capture/inference work runs on daemon threads with interval gating,
- the module surface exposed to the agent is a pure `query()` projection,
- missing model weights degrade a backend to an explicit `unavailable`
  state instead of failing the module,
- private state counters are persisted through `serialize_state`,
- modules that opt in (`ActionSurface`) also expose **channels**:
  write-only data slots the model reaches through the
  `list_channels` / `show_channels` / `invoke_channels` verb triple;
  see [design/reactive-modules.md](design/reactive-modules.md).

## Gateway and events

`gateway.py` serves the HTTP API on `gateway.host:gateway.port`
(default `127.0.0.1:8765`) and bridges to the web UI in `frontend/`.
`events.py` implements the in-process pub/sub bus with bounded history and
subscriber queues.

## Security model

- The only inbound listener is the gateway, bound to the loopback
  interface by default (verified by the test suite).
- All repo-relative paths (`data/`, `models/`, `config/`, `builtin/`)
  resolve through `utils/paths.py`; the process never depends on its
  working directory and writes no state outside the repo directories.
- Trusted subprocesses (MCP servers, skill scripts) inherit the user
  environment and are unrestricted by design; review a tool config before
  adding it.
