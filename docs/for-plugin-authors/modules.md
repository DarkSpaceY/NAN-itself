# Modules

A Module is a long-lived, autonomous service that senses, computes and
acts at its own rhythm while the agent runs its turn loop — a sensor
stack, an inbox, a realtime controller. This document is the reference
for the modules subsystem in
[`backend/nan_itself/modules/`](../../backend/nan_itself/modules/) and
the contract every module implements. For how modules fit into the
whole, see [architecture.md](../for-contributors/architecture.md).

## Where modules live

Modules live in two parallel roots scanned with identical semantics —
`builtin/modules/` ships with the repo, `workspace/modules/` is user
territory. **One file = one module**: a source file carries a
`# @module` header within its first 20 lines and must define exactly
one concrete `Module` subclass. Editing a file replaces the module
live; deleting the file removes it.

Each module owns `config/modules/<module_id>.yaml` and loads and
validates it itself. See [../for-users/configuration.md](../for-users/configuration.md).

## The modules package

```
runtime ──> { deps, loading, persistence, reload } ──> model
```

- `model.py` — the pure contracts: `DataSpace`, `DataSpaceReader`,
  `Turn`, `ChannelSpec`, `Module`, `ModuleState`, `ModuleRecord`. It
  imports nothing from siblings.
- `runtime.py` — the `Facade`, the runtime supervisor.
- `deps.py` (dependency graph), `loading.py` (discovery and dynamic
  import), `persistence.py` (crash-safe state files), `reload.py`
  (hot-reload handoff).

Import from the package (`from nan_itself.modules import ...`); the
runtime machinery, including `Facade`, loads lazily on first attribute
access. Loaded module files get `Module` and `Turn` injected into their
namespace; anything else (`ChannelSpec` included) is imported
explicitly.

## Module surface

A module is a class with a string `id`, an optional `requires` tuple of
dependency ids, and an instance lifecycle:

```python
# @module
"""One-line module description."""

from nan_itself.modules import Module


class MyModule(Module):
    id = "my_module"

    requires = ("other_module",)   # optional dependencies

    async def start(self) -> None:   # the service lifetime
        ...
```

The base class provides the instance handles: `self.data` (the
module-owned `DataSpace`), `self.dependencies` (read-only
`DataSpaceReader` handles onto required modules' spaces), and
`self.llm` (an LLM handle when the composition root supplies one).

The Facade owns everything else: discovery and hot reload, the
dependency graph, lifecycle supervision and retry, DataSpace
ownership, persistence, per-turn delivery, and the channel downlink
(see below).

| Face | Mechanism |
|---|---|
| Perception | `DataSpace` publish/read (`self.data`, `self.dependencies`, revisioned detached snapshots) |
| Turn coupling | `tell()` notification after each turn, `ask()` per-turn ambient projection |
| Persistence | `serialize_state()` / `restore_state()` |
| Channels | model-writable downlink endpoints (opt-in via a `channels` declaration) |

### Lifecycle and failure semantics

Module states: `NEW → STARTING → RUNNING`, with `DOWN` and `STOPPING`
on the side. `start()` represents the service lifetime — normally a
long-running coroutine; a state-only module that returns immediately is
treated as exited and retried. Failures follow let-it-crash: an
exception out of `start()`, `ask()`, `tell()` or `feed()` marks the
module `DOWN` with the recorded error, cancels its live task, and the
supervisor restarts it with backoff. The failure never propagates to
the agents — an `ask()` failure simply leaves that turn without the
module's contribution. Provisioning failures (missing model weights,
absent hardware) come up loudly failed and revive once the resource
appears.

### Performance rule

The module's one hard obligation: `ask()` stays a cheap, read-only
projection of already-computed state — it runs on the agent's execution
path, and a slow `ask()` delays every agent's start. The rest of the
discipline follows from one shared event loop:

- `tell()` runs in its own task and may *await* long operations; a
  synchronous CPU-heavy or blocking call still stalls the loop and
  freezes every module and agent — such work belongs on threads
  (`start()`'s daemon threads, `asyncio.to_thread` for short
  known-blocking calls).
- `DataSpace` carries small JSON facts only: `publish()` and
  `snapshot()` deepcopy the full state on the event loop every round,
  so frame data or large documents never go through it.
- Provisioning-length work (weight loading, device probing) runs on
  `start()`'s background threads, never synchronously inside a
  coroutine.

## DataSpace

A module's published state. The owner is the only writer;
`publish()` deep-copies the incoming mapping and bumps `revision`;
readers — dependents and the per-turn snapshot — receive completely
detached deep copies. Dependents hold a `DataSpaceReader`, which
deliberately does not expose `publish()`.

## Turn coupling

- **`ask(turn)`** is called on every RUNNING module at turn start; the
  returned strings become the turn's ambient context. It must be a pure
  projection of already-computed state.
- **`tell(turn)`** is called after the turn completes, each handler in
  its own task: a slow or failing module can never delay its peers or
  the agents.
- The turn carries a single detached `DataSpace` snapshot shared by the
  whole dispatch tree, so every ask in one turn observes the same world.

Private state is persisted as JSON under `data/modules/private/<id>.json`
(`serialize_state()` payload) alongside the published DataSpace snapshot
under `data/modules/dataspace/<id>.json`; both are restored when the
instance is (re)created. A corrupt file blocks only its own module.

## Channels — the model downlink

### Principle: delivery, not invocation

Model-to-module interaction through channels is **data flow, not
control flow**:

- Writing a channel is a one-way `feed()`: the model never blocks, and
  never knows that ticks exist.
- The module consumes the payload at its own tick; its consumption
  policy — append to a private FIFO, overwrite private state, wake a
  tick loop, preempt a goal — is entirely module-private.
- There is no acceptance protocol in the core; the only outcome of a
  write is the `written` / `rejected` result of the write itself.

### Declaring channels

Channels are declared opt-in as a `channels` mapping on the module
class; the name is the mapping key and each value is a `ChannelSpec`:

```python
class Goal(BaseModel):
    text: str
    priority: int = 1


class MyModule(Module):
    id = "my_module"

    channels = {
        "goal": ChannelSpec(Goal, description="Navigation goal"),
        "burst": ChannelSpec(description="Command burst"),  # unvalidated
    }
```

`ChannelSpec(model=None, *, description="")` declares the payload shape
with a pydantic model class — the same annotation-driven mechanism
`@tool` uses — so the JSON schema handed to the model and the validation
applied at write time derive from one source. `model=None` means the
channel accepts any JSON value unvalidated. A module that declares no
channels behaves exactly as before the mechanism existed.

### The write path

```
   model                      Facade                             Module
     │                          │                                  │
     │  invoke_channels         │ 1. module exists and RUNNING     │
     │  (module, channel,       │ 2. channel exists                │
     │   payload) ─────────────▶│ 3. ChannelSpec.validate(payload) │
     │                          │ 4. deepcopy the normalized       │
     │                          │    payload                       │
     │                          │ 5. feed(channel, payload) ──────▶│ queue / overwrite /
     │                          │                                  │ wake the tick loop
     │ ◀──────────────── 'written' / 'rejected' / error string ────│
```

Validation happens entirely at the boundary: a payload is checked
against the declared schema, normalized (`model_dump()`), deep-copied,
and only then handed to `Module.feed(channel, payload)`. A module's
`feed()` never sees malformed payloads; validation errors come back
verbatim on the `rejected` result string.

`feed()` is sync by design and returns `'written'` or `'rejected'`. The
base implementation accepts everything (`'written'`); override it to
store or act on the payload. An exception out of `feed()` crashes the
module (DOWN + supervised restart) and reaches the model only as a
failure string.

**Routing never raises.** Every failure — unknown module, module not
running, channel-free module, unknown channel, schema rejection — comes
back as a result string, never as an exception.

### The verb triple

Channels are reached through the same verb triple as every interface
face — list enumerates, show inspects, invoke acts:

- **`list_channels`** — bare `module/channel` lines for every RUNNING
  module that exposes channels;
- **`show_channels`** — a channel's description and JSON schema.
  Channels are **write-only**: the payload is never rendered back to
  the model — once fed, it belongs to the module;
- **`invoke_channels`** — feed one payload; returns `'written'`,
  `'rejected'`, or an error string.

There is **no clear verb**: to retract or supersede a not-yet-consumed
payload the model writes a new one; how a new write interacts with what
was already delivered is module-private policy. In the frontend record
stream the channel verbs render under the record kind `target`
(glyph `⌖`).

### Consumption and tick loops

A channel makes sense for modules that run a high-rate *perceive →
decide → act* loop — realtime control, streaming interaction — where
neither an LLM round nor a passive perceiver fits. The pattern:

- the tick loop is the long-running coroutine in `start()`;
- the decision step inside the loop is a swappable *decision backend*
  (a hosted API, a local model, plain rules — module-private choice);
  the LLM has no place inside the tick loop;
- fed payloads are consumed at the module's own tick; "current goal" is
  module-private state derived from them;
- there is no read receipt: downstream observers learn what happened
  through the module's own `DataSpace` and `ask()` projection;
- modules never block on the model: a tick that hits a case it cannot
  judge surfaces it through the uplink (failure / needs-guidance as
  module-rendered state), and the agent decides what to write next;
- tick frequency and budget semantics are module-private; the core does
  not see them.

Multiple channels on one module run independently — the
parallel-control case is expressed as multiple channels, not as a task
queue. Payload granularity convention: one write is one unit the module
can close the loop on within its own domain; cross-domain orchestration
stays with the agent.

### What channels deliberately are not

No `Task` abstraction: no task type, no registry, no acceptance
lifecycle in the core. An explicit goal/feedback protocol (actionlib
style) would add a core-side ledger and acceptance semantics; channels
keep the core ledger-free and push orchestration responsibility to the
agent side. If progress reporting matters, the module publishes it
through its own `DataSpace` — the perception face already covers it.

### Interaction with the rest of the architecture

- **Snapshot.** Channel summaries and one-shot feedback events are
  rendered by each module's own `ask()` from its private state (one-shot
  events are drained on render); the engine needs no changes, and the
  model does not re-send payloads it already sent.
- **Hot reload.** The core persists nothing about channels: fed payloads
  are transient and are dropped when a reload rebuilds the instance;
  whether any derived state survives is the module's own
  `serialize_state()` decision.
- **One file = one module** is untouched; channels live in the module
  file alongside the rest of the class.
