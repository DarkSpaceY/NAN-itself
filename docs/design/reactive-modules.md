# Design: Reactive Modules — Channel Downlink

Status: **implemented** (2026-09-20). ChannelSpec lives in
`modules/model.py`, ActionSurface in `modules/action.py`, facade
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

- Writing a target is fire-into-slot: the model never blocks, and never
  knows that ticks exist.
- The module consumes the slot at its own tick; its consumption policy
  (including preemption and goal replacement) is entirely module-private.
- Explicit acceptance / rejection / preemption protocols are *not* part of
  the architecture — only the write result of the slot itself.

## Module surface after the change

Unchanged faces (all current mechanics stay):

| Face | Today |
|---|---|
| Perception | `DataSpace` publish/read (`self.data`, `self.dependencies`, revisioned snapshots) |
| Turn coupling | `on_turn()` observation + `query()` per-turn projection |
| Uplink | events / inbox |
| Persistence | `serialize_state()` / `restore_state()` |

New face, **opt-in** via capability declaration:

- **Channels** — downlink data slots the model may write to. A module holds
  zero or more; each is declared independently.

Modules that declare no channels behave exactly as today (zero regression).

## Channel contract

- **Declaration** (on the module class): `name` + `schema` (JSON schema
  derived from type annotations, the same mechanism `@tool` uses — *open:
  exact annotation form*) + `depth`.
- **Depth is declarative**:
  - `depth = 1` — overwrite slot; the newest target wins. Sensible default
    for goal / intent messages (no stale goals ever queue up). The write
    result is `replaced` when a previous target was present, `written`
    otherwise.
  - `depth = N` — FIFO queue. For command streams where no message may be
    dropped (e.g. key sequences). Overflow drops oldest (*open*).
- **One-way downlink** (model → module). The uplink remains
  `DataSpace` / `query()` projection + events / inbox. No new uplink
  semantics.
- **Write contract**: `update_target(module_id, channel, payload)` → schema
  validation → **deep copy** into the slot → returns `written` /
  `replaced` / `rejected`. Validation failures are rejected at the
  boundary; a module's tick never sees malformed payloads.
- **Consumption**: the module sees slot contents on its next tick and
  consumes freely. "Current goal" is module-private state derived from
  consumed slot data. There is no read receipt; downstream observers learn
  what happened through the module's own `DataSpace` state.

## Agent-facing verbs

Aligned with the existing interface-face verb triple (`list_tools` /
`show_tool` / `invoke_tool`, `list_skills` / `show_skill` /
`invoke_skill`) — the model keeps exactly one mental model for every
interface face: **list enumerates, show inspects, invoke acts.**

- `list_channels` — enumerate a module's exposed channels, rendering each
  channel's schema so the model can construct valid payloads;
- `show_channels` — channel details: schema, depth, occupancy state.
  Slots are **write-only**: the current payload is never rendered back to
  the model — once written it can only be consumed by the module or
  overwritten;
- `invoke_channels` — write the slot: schema validation → deep copy →
  returns `written` / `replaced` / `rejected`.

There is **no clear verb**: to retract a not-yet-consumed target the model
writes a new one (e.g. a standby target); overwrite semantics make an
explicit clear unnecessary, and the consumption rhythm stays entirely with
the module.

UI record kind `"target"` (glyph `⌖`, protocol.ts + PROTOCOL.md
synchronized) — reuse that convention from the prior implementation.

## No Task abstraction

Deliberately absent: no `Task` type, no registry, no acceptance lifecycle
in the core. If progress reporting matters, the module publishes it through
its own `DataSpace` — the perception face already covers it.

*Design note:* this is a deliberate contrast with ROS actionlib's explicit
ActionServer / Goal / Feedback / Result protocol. The explicit-task model
adds a core-side ledger and acceptance semantics; the data-slot model keeps
the core ledger-free and pushes orchestration responsibility to the agent
side.

## Interaction with existing architecture

- **Snapshot**: channel + current-target summaries are injected into the
  derived snapshot so the model does not re-send targets it already sent.
  Exact format and budget: *open*.
- **Hot reload**: slot residue is dropped when an instance is rebuilt.
  Channel state is transient and does **not** participate in
  `serialize_state()`.
- **One file = one provider** is untouched; channels live in the module
  file alongside the rest of the class.
- **Facade routing never raises**: failures (unknown module / channel,
  module not running, no action surface) come back as result strings, not
  exceptions.
- **Modules never block on the model.** A tick loop that hits a case it
  cannot judge surfaces it through the uplink (failure /
  needs-guidance event); the agent decides what to send next. The LLM has
  no place inside the tick loop.
- **Performance rule unchanged**: `query()` stays a cheap projection; the
  tick loop belongs to the long-running coroutine in `start()`.

## Prior attempt & lessons

A previous implementation (ActionSurface + ChannelSpec in
`src/nan_itself/modules/model.py`, Facade routing in `runtime.py`, the three
verbs above in `verbs.py`, plus a temporary validation module) was built,
validated end-to-end, and then reverted. The lessons carry over:

- **Throwaway-probe validation works**: a temporary module with two
  channels and a fake decision backend validated the full chain in isolation
  (no external deps); it was deleted after validation and must not be kept
  in the codebase.
- **Never point a test Facade at the real `builtin/modules/`** — real
  modules (audio) would start. Tests must isolate `builtin_tools_dir` /
  module dirs to tmp.
- **`asyncio.run` cannot nest** — scenario coroutines must
  `await verb.execute()` / `module.query()` directly rather than spawning
  nested loops, or tasks starve.
- **The registry summary must be injected into the turn snapshot** —
  otherwise the model re-sends targets it already sent. It does not live in
  `Turn` records.

## Candidate applications

All are the same move; each module declares channels and runs a tick loop
with a decision backend inside:

| Scenario | Module | Channels (illustrative) |
|---|---|---|
| Full-duplex voice | audio | `tts-control`, `barge-in-policy` |
| High-rate computer use | desktop (Touchpoint-backed) | `goal`, `interrupt` |
| 3D character | avatar | `body`, `speech`, `gaze` — independent channels give "walk while talking" for free |
| Game bot | minecraft | `locomotion`, `inventory` |

Multiple channels on one module run independently — the parallel-control
case is expressed as multiple channels, not as a task queue.

## Open implementation details

- Exact schema annotation form for channel declarations and where
  validation errors are surfaced to the model on `rejected`.
- Snapshot summary format and character budget for channel / target state.
- Overflow discipline for `depth > 1` queues (single-consumer FIFO; the
  leaning is drop-oldest).
- Target granularity convention: one target = one unit the module can
  close the loop on within its own domain; cross-domain orchestration stays
  with the agent.
- Tick frequency and budget semantics are module-private; the core does
  not see them.
