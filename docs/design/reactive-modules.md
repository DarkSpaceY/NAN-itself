# Design: Reactive Modules — Channel Downlink

Status: **implemented** (2026-09-20). ChannelSpec and the
`channels` declaration live in `modules/model.py`, facade
routing in `modules/runtime.py`, verb triple in
`agent/verbs.py`, mechanism tests in
`tests/test_reactive_modules.py`. Channel payload schemas are
pydantic models -- the same annotation-driven mechanism `@tool`
uses (`model=None` means unvalidated passthrough). Details marked
*open* are decided at implementation time.

## Motivation

Several planned capabilities — full-duplex voice conversation, high-rate
computer use, continuous 3D character animation, game bots — are all the
same shape: a high-frequency *perceive → decide → act* loop. Neither
existing loop fits:

- the **agent turn loop** is an LLM round (slow, expensive, but capable of
  planning and language);
- the **module daemon** (today) is a passive perceiver: it computes and
  projects, it never acts on goals.

The trigger for this design was the emergence of cheap calibrated decision
models ("System 1" style, e.g. Jev) that make high-frequency decision loops
economically viable. **The design binds to no specific product or backend.**
Inside a module, the decision step is a swappable *decision backend* (a
hosted API today, a local replica, distilled model, or plain rules
tomorrow). The architecture only fixes the shell around it.

Convergence: every scenario above collapses to the same action —

> declare some **channels** on a module + write a **tick loop** in
> `start()`.

## Principle: delivery, not invocation

Model-to-module interaction is **data flow, not control flow**.

- Writing a channel is a one-way `feed()`: the model never blocks, and
  never knows that ticks exist.
- The module consumes the payload at its own tick; its consumption policy
  (including preemption and goal replacement) is entirely module-private.
- Explicit acceptance / preemption protocols are *not* part of
  the architecture — only the `written` / `rejected` result of the
  write itself.

## Module surface after the change

Unchanged faces (all current mechanics stay):

| Face | Today |
|---|---|
| Perception | `DataSpace` publish/read (`self.data`, `self.dependencies`, revisioned snapshots) |
| Turn coupling | `tell()` notification + `ask()` per-turn ambient projection |
| Uplink | events / inbox |
| Persistence | `serialize_state()` / `restore_state()` |

New face, **opt-in** via a `channels` declaration on the module class:

- **Channels** — downlink endpoints the model may write to. A module holds
  zero or more; each is declared independently.

Modules that declare no channels behave exactly as today (zero regression).

## Channel contract

- **Declaration** (on the module class): `name` + `schema` (a pydantic
  model class — the same annotation-driven mechanism `@tool` uses;
  `None` means unvalidated passthrough).
- **One-way downlink** (model → module). The uplink remains
  `DataSpace` / `ask()` ambient projection + events / inbox. No new uplink
  semantics.
- **Write contract**: `invoke_channels(module, channel, payload)` →
  channel existence + schema validation → **deep copy** →
  `module.feed(channel, payload)` → returns `written` / `rejected`.
  Validation failures are rejected at the boundary; a module's `feed()`
  never sees malformed payloads.
- **Consumption**: the module consumes the payload inside its own
  `feed()` — append to a private FIFO, overwrite private state, wake a
  tick loop — and the policy is entirely module-private. "Current goal"
  is module-private state derived from fed payloads. There is no read
  receipt; downstream observers learn what happened through the module's
  own `DataSpace` state.

## Agent-facing verbs

Aligned with the existing interface-face verb triple (`list_tools` /
`show_tool` / `invoke_tool`, `list_skills` / `show_skill` /
`invoke_skill`) — the model keeps exactly one mental model for every
interface face: **list enumerates, show inspects, invoke acts.**

- `list_channels` — enumerate every RUNNING module's channels as
  `module/channel` with its schema;
- `show_channels` — channel details: description and schema. Channels
  are **write-only**: the payload is never rendered back to the model —
  once fed it belongs to the module;
- `invoke_channels` — feed the channel: schema validation → deep copy →
  `feed()` returns `written` / `rejected`.

There is **no clear verb**: to retract or supersede a not-yet-consumed
payload the model writes a new one (e.g. a standby goal); how a new write
interacts with what was already delivered is module-private policy, and
the consumption rhythm stays entirely with the module.

UI record kind `"target"` (glyph `⌖`, protocol.ts + PROTOCOL.md
synchronized) — reuse that convention from the prior implementation.

## No Task abstraction

Deliberately absent: no `Task` type, no registry, no acceptance lifecycle
in the core. If progress reporting matters, the module publishes it through
its own `DataSpace` — the perception face already covers it.

*Design note:* this is a deliberate contrast with ROS actionlib's explicit
ActionServer / Goal / Feedback / Result protocol. The explicit-task model
adds a core-side ledger and acceptance semantics; the channel model keeps
the core ledger-free and pushes orchestration responsibility to the agent
side.

## Interaction with existing architecture

- **Snapshot**: the channel registry summary and one-shot feedback
  events are rendered by each module's own `ask()` from its private
  state — the engine needs no changes, and the model does not re-send
  payloads it already sent. One-shot events are module-private state,
  drained on render.
- **Hot reload**: channel state is transient and does **not** participate
  in `serialize_state()`; it is dropped when an instance is rebuilt.
- **One file = one provider** is untouched; channels live in the module
  file alongside the rest of the class.
- **Facade routing never raises**: failures (unknown module / channel,
  module not running, schema validation) come back as result strings, not
  exceptions.
- **Modules never block on the model.** A tick loop that hits a case it
  cannot judge surfaces it through the uplink (failure /
  needs-guidance event); the agent decides what to send next. The LLM has
  no place inside the tick loop.
- **Performance rule (the module's one hard obligation)**: `ask()` stays
  a cheap, read-only projection of already-computed state -- a slow
  `ask()` delays every agent's execution start -- and the tick loop
  belongs to the long-running coroutine in `start()`. The rest of the
  discipline:
  - `tell()` runs in its own task and may *await* long operations
    (thread results, async APIs); a synchronous CPU-heavy or blocking
    call still stalls the one shared event loop and freezes every
    module and agent -- such work belongs on threads (`start()`'s
    daemon threads; `asyncio.to_thread` for short known-blocking
    calls).
  - `DataSpace` carries small JSON facts only: `publish()` and
    `snapshot()` deepcopy the full state on the event loop every
    round, so frame data or large documents never go through it.
  - Provisioning-length work (model weight loading, device probing)
    runs on `start()`'s background threads, never synchronously
    inside a coroutine.

## Prior attempt & lessons

A previous, slot-based implementation of this same design (declared in
`backend/nan_itself/modules/model.py`, facade routing in `runtime.py`, the three
verbs above in `verbs.py`, plus a temporary validation module) was built,
validated end-to-end, and then reverted; the current code is the second,
simplified take. The lessons carry over:

- **Throwaway-probe validation works**: a temporary module with two
  channels and a fake decision backend validated the full chain in isolation
  (no external deps); it was deleted after validation and must not be kept
  in the codebase.
- **Never point a test Facade at the real `builtin/modules/`** — real
  modules (audio) would start. Tests must isolate `builtin_tools_dir` /
  module dirs to tmp.
- **`asyncio.run` cannot nest** — scenario coroutines must
  `await verb.execute()` / `module.ask()` directly rather than spawning
  nested loops, or tasks starve.
- **The registry summary must be injected into the turn snapshot** —
  otherwise the model re-sends payloads it already sent. It does not live in
  `Turn` records.

## Candidate applications

All are the same move; each module declares channels and runs a tick loop
with a decision backend inside:

| Scenario | Module | Channels (illustrative) |
|---|---|---|
| Full-duplex voice | **voice** (implemented, requires `audio`) | `say` — semantic task FIFO `{intent, key_points, tone, interruptible}`; barge-in drains it |
| High-rate computer use | desktop (Touchpoint-backed) | `goal`, `interrupt` |
| 3D character | avatar | `body`, `speech`, `gaze` — independent channels give "walk while talking" for free |
| Game bot | minecraft | `locomotion`, `inventory` |

Multiple channels on one module run independently — the parallel-control
case is expressed as multiple channels, not as a task queue.

## Open implementation details

- Payload granularity convention: one write = one unit the module can
  close the loop on within its own domain; cross-domain orchestration stays
  with the agent.
- Tick frequency and budget semantics are module-private; the core does
  not see them.

(Decided during implementation: channel schemas are pydantic models;
validation errors are returned verbatim to the model on `rejected`;
the registry summary is module-rendered from module-private state.)
