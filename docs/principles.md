# Core Principles

These are the load-bearing principles of NAN-itself. They constrain
design decisions across the framework; when a change violates one of
these, the change needs an explicit, argued exception — not a quiet
workaround.

## 1. Local-first

Everything runs on the user's machine. The only external calls are the
LLM endpoint(s) the user configures in `settings.yaml`. No cloud
services, no telemetry, no accounts in the core.

## 2. Cross-platform runtime

No OS-specific or hardware-specific runtime dependencies in the core.
The portability baseline is CTranslate2 / onnxruntime / torch /
transformers — the same code must run on macOS, Linux and Windows, on
any vendor's hardware. Hardware-specific accelerators (MLX, CoreML,
DirectML) are optional extras layered on top, never requirements, and
never the only code path. (Precedent: the VLM backend is
transformers/torch, MLX explicitly excluded.)

## 3. The LLM stays out of the hot path

Deliberation belongs to the agent turn loop. Perceive → decide → act
loops that need a faster rhythm than LLM round-trips run *inside*
modules, at module frequency, through channels. Ambient state is
computed continuously in the background; the model reads projections,
it is never the inner loop of a control system.

## 4. Turn snapshot

One round of agent execution is captured in a single `Turn` record:
`persona` + `history` (the snapshot the model saw) + `messages` (what
this round produced). There is no separate persistent history store —
the next round's snapshot is derived from the last turn, so anything
the model saw is reconstructible exactly.

## 5. One file = one unit; two parallel roots

One file (or directory) is exactly one provider, module, or skill —
no multi-unit files. Builtin (`builtin/`) and user territory
(`workspace/`) are two parallel roots scanned with identical hot-reload
semantics. A reload is a transaction: the replacement starts first,
unregistered; it is committed only after it is fully connected; the old
generation keeps serving until that moment. Deleting a source file
disables its unit.

## 6. Bounded by design

Every tool call is timeout-bounded. The only inbound listener is the
gateway, bound to the loopback interface. All repo-relative paths
resolve through a single path-anchoring module — the process never
depends on its working directory and writes no state outside the repo
directories. Workspace sources are code and inherit the user's trust;
review before adding.

## 7. Full implement, let it crash

Implement features completely — never ship partial implementations,
placeholder behavior, or silent simplifications. On failure, crash
loudly and surface the error: no silent fallbacks, no masking, no
"unavailable" limbo states that quietly keep running at half capacity.
Supervisors restart crashed units with backoff, and the error stays
visible until the cause is fixed. A module with missing weights comes
up failed and revives the moment the weights land.

## 8. Data flows up, decisions flow down

Modules publish facts, never conclusions — interpretation belongs to
the agent. The agent's intent reaches back down through write-only
channel slots. `query()` is a cheap projection (it runs on the agent's
critical path every turn); heavy work lives in module loops. Neither
side blocks the other.
