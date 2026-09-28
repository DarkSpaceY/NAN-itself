# NAN WS Event Protocol v2

Status: **target specification**. This document describes the protocol both sides must
converge on. It is *not* implemented yet: the current code still speaks v1, and the gap
is enumerated under **Migration**. Where this document and the code disagree today, the
code is the old behaviour — not a reason to keep it.

The one principle that drives v2:

> **The backend sends structured, typed data; the frontend owns all rendering.**

Everything below follows from that sentence. Concretely: the backend never chooses a
glyph, a colour, a label or a sentence. It publishes *categories*, *typed payloads* and
*typed entries*; the mapping from category to visual vocabulary (glyph, colour, dimming,
folding, the human label) lives entirely in the frontend. This spec therefore pins **no
glyph at all** — v1 carried a glyph table that disagreed with both `DESIGN.md` and
`protocol.ts`, which is exactly the drift the ownership rule removes.

## 1. Transport

- A **same-origin WebSocket** at `/ws`. The client connects to
  `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`.
- In production the gateway serves the static frontend and the `/ws` endpoint from the
  same origin (`127.0.0.1:8765`); in development Vite proxies `/ws` to that port.
- One JSON object per WebSocket message (`send_text`), no line framing.
- Field names are lowercase.

## 2. client → server

| `t` | fields | meaning |
|---|---|---|
| `input` | `text`, `mid?` | One line of user input (equivalent to a line on stdin). `mid` is the client message id: the server dedups by it, so a `mid` re-sent after reconnect is delivered exactly once. The `user_input` echo carries the same `mid` back. |
| `ping` | — | Heartbeat. Answered with `{"t":"pong","content":{}}`. |

## 3. Connection layer (heartbeat, reconnect, dedup)

These rules are client- and gateway-owned and are unchanged in spirit from v1:

1. **Liveness is the heartbeat's job.** The client sends `ping` every 15 s; if no `pong`
   arrives within 4 s it treats the socket as dead and closes it (which triggers a
   reconnect).
2. **The acknowledgement watchdog never drops the line.** After sending `input` the
   client arms a 5 s timer that only clears the local pending state. A slow echo (a busy
   turn) is not a lost message; kicking the socket and re-sending would duplicate it.
3. **Exactly-once on reconnect.** On a real close (`onclose`) with an input still
   pending, the client reconnects (~1.2 s later) and re-sends that input with the **same
   `mid`**. The gateway dedups by `mid` against a bounded cache (`dedup_cache_size`), so
   the message is delivered once. A `mid`-less input (the stdin entry) is not deduped.
4. **`seq` dedup.** Any event whose `seq` is `<=` the highest seen is dropped as replay.
   `hello` is exempt: it is always let through.
5. **`hello` starts the session.** On connect the gateway sends `hello` first, then
   replays the bus **history ring** (up to `HISTORY_LIMIT` = 500 raw events) and the
   client folds it. If `hello.content.boot` differs from the previous boot, the client
   resets its `seq` baseline and clears the stream (the process was replaced).

> v1 claimed a 200-event *folded projection*. The code replays the raw 500-event ring
> (`HISTORY_LIMIT` in `backend/nan_itself/events.py`) and the client folds it; v2 states
> the real behaviour.

## 4. server → client

Every event is the **envelope** (top-level, stamped by the transport / `sink`) plus `t`
(the type discriminant) and `content` (the business payload):

| field | filled by | meaning |
|---|---|---|
| `seq` | `EventBus` | Monotonic counter. The client dedups history replay by it. |
| `ts` | `EventBus` | Wall-clock epoch **seconds, numeric** (`time.time()`). The frontend turns it into a date or time only at the render/fold edge. |
| `t` | caller | The **single** event-type discriminant (a flat string). |
| `id` | `sink` | Event id — the UI row key. A caller may pass one to reuse an existing row; otherwise the sink generates it. |
| `boot_id` | `sink` | Process-unique id recorded at `attach`. Present on every event once set; the key is absent otherwise. |

`content` rules:

- It is **always an object**. An event with no payload still carries `{}` — never omitted.
- It carries **all** of the `t`'s business fields (no kwargs scattered at the top level).
- The identity keys (§5) appear only when the event has agent context.

### Events by `t`

| `t` | `content` | notes |
|---|---|---|
| `hello` | `{ boot, model, base_url, status: { state } }` | Handshake. `boot` is the process-unique id; a client seeing it change clears its stream and resets the `seq` baseline. **No `id`** (it does not go through the sink); `seq` is mandatory. |
| `status` | `{ state: "idle" \| "working" \| "error" }` | Loop state-machine change. |
| `user_input` | `{ text, mid? }` | Echo of accepted input (emitted only once it is in the Inbox). `mid` is returned verbatim. |
| `output_started` | `{}` | A NAN text stream begins. |
| `output_delta` | `{ text }` | Streaming increment (concatenate directly). |
| `output_done` | `{ duration_s }` | Typing stopped. `duration_s` is a **number of seconds** (§8). |
| `output_cancelled` | `{}` | The text stream turned out to be tool calls; retract that segment. |
| `record_started` | `{ category, payload }` | A machine process begins (spinner). See §6. |
| `record_detail` | `{ entries }` | Typed detail entries appended. See §7. |
| `record_done` | `{ result?, duration_s? }` | Auto-fold. `duration_s` is a **number of seconds**. |
| `record_failed` | `{ error: { type, message } }` | Stays expanded. `message` is prose (§9). |
| `record_void` | `{}` | The record produced nothing user-visible; retract its row. |

`pong` is the heartbeat reply, `{"t":"pong","content":{}}`; it takes no part in stream
folding.

### `id` association

`record_started` returns a fresh (or caller-supplied) `id`; the same record's subsequent
`record_detail` / `record_done` / `record_failed` / `record_void` reuse it.
`output_started` works the same way for `output_delta` / `output_done` / `output_cancelled`.
`record_void` and `output_cancelled` retract only the **most recent** same-`id` row: a
same-`id` row from an older turn is untouched.

**Dividers are not protocol events.** The frontend derives date separators itself from
each event's numeric `ts` (inserting one when the local date changes).

## 5. Identity keys (inside `content`)

The three identity keys are placed in `content` by whichever side emits the event. When
absent there is **no** agent context (module lifecycle, the gateway's `user_input` echo);
the keys do not exist — they are not `null` (except `parent_hash`, which is explicitly
`null` for the root).

- `agent_hash` — the emitting agent's stable id.
- `parent_hash` — the parent agent's `agent_hash`; `null` for the root.
- `depth` — depth in the agent tree; `0` for the root.

## 6. Record phases and categories

A record is a four/five-event family keyed by `id`:

```
record_started  { category, payload: { ... } }
record_detail   { entries: [ ... ] }
record_done     { result?: {...}, duration_s?: number }
record_failed   { error: { type, message } }
record_void     { }
```

`category` is the semantic type of the record; `payload` is a **union discriminated on
`category`**, so each category has its own typed fields (there is no shared `name` /
`summary` display string any more).

| old kind (v1) | category (v2) | emission site | `payload` fields |
|---|---|---|---|
| `tool` | `tool_call` | `agent/core.py` (tool verbs) | `provider`, `tool`, `arguments` — the **raw argument object**, not pretty-printed text |
| `skill` | `skill_invoke` | `agent/core.py` (skill verbs) | `skill`, optional `resource` |
| `target` | `channel_write` | `agent/core.py` (channel verbs) | `module`, `channel`, optional `payload` |
| `spawn`, and `agent` from `agent/verbs.py` | `subagent_spawn` | `agent/verbs.py` | `agent_id`, `depth`, `task` (prose) |
| `agent` from `agent/core.py` | `subagent_report` | `agent/core.py` (report harvest) | `agent_id`, `task`, `status`, `body` (prose) |
| `module` | `module_query` | `agent/core.py` (ambient module query) | — (no category-specific fields) |
| `sleep` | `sleep` | `agent/core.py` (from the `sleep` verb) | `seconds` — a **number**, not a formatted string |
| `finish` | `finish` | `agent/core.py` (from the `finish` verb) | — (no category-specific fields) |
| anything else (v1 fallback `verb`) | `unknown` | `agent/core.py` (verb dispatch fallback) | `verb` — the original verb tool-name |

> The old `agent` kind was overloaded: `agent/verbs.py` used it for a **subagent spawn**
> while `agent/core.py` used it for a **harvested subagent report**. v2 splits them into
> `subagent_spawn` and `subagent_report`.

## 7. Detail entries

`record_detail.content.entries` is an array of typed entries. The four kinds replace the
v1 free-text `line`:

| entry | shape | replaces (v1) |
|---|---|---|
| text | `{ kind: "text", text }` | a plain line |
| item | `{ kind: "item", text }` | the `· `-prefix convention: the frontend stripped the prefix (`frontend/app/src/design/RecordItem.tsx`) and dimmed the remainder. No backend site ever emitted that prefix, so the marker was dead defensive code; the typed entry retires it. |
| field | `{ kind: "field", label, value }` | the `"id: x"` / `"task: y"` / `"status: z"` strings |
| code | `{ kind: "code", text }` | the multi-line pretty-printed JSON block (today `json.dumps(..., indent=2)`) |

The frontend renders by entry `kind`; it never sniffs the text for `· `, `✓` or `✗`.

## 8. Numbers are numbers

Every duration on the wire is a **number of seconds**, never a pre-formatted string:

- `output_done.content.duration_s`
- `record_done.content.duration_s`
- `sleep.payload.seconds`

The frontend formats them (units, decimals) at the render edge. The v1 formatted strings
— `note` (`f"{…:.1f}s"` in `agent/core.py`) and `duration` (`f"{…:.1f}s"` in
`agent/engine.py`) — are gone.

## 9. The prose boundary — what stays a string

Strings remain legitimate only for genuine prose: model output, user language, error
messages, delegation instructions and report bodies. Concretely, the following are
strings and **must not** be turned into display-formatted structures by the backend:

- `output_delta.content.text` — model output.
- `user_input.content.text` — user language.
- `subagent_spawn.payload.task` — the model's delegation instruction.
- `subagent_report.payload.body` — the child agent's report body.
- `record_failed.content.error.message` — an error message (`error.type` is a machine
  token, e.g. an exception class name; `message` is prose).
- tool / skill result payloads — the callee's own text or data.

Outside this list, if the backend finds itself choosing a label, a separator or a
sentence, that is a design error: it belongs to the frontend.

## 10. TypeScript contract

The frontend-side mirror of this protocol (documentation only — the real file is
`frontend/app/src/protocol.ts`):

```ts
export type StatusState = 'idle' | 'working' | 'error';

export interface Status {
  state: StatusState;
}

// Envelope: transport-layer fields stamped by the EventBus / sink.
export interface Envelope {
  seq?: number;
  ts?: number;
  id?: string;
  boot_id?: string;
}

// Identity keys: carried inside `content` by the emitting side; absent when the
// event has no agent context (the keys do not exist, they are not null).
export interface Identity {
  agent_hash?: string;
  parent_hash?: string | null;
  depth?: number;
}

interface Payload<T extends string, C> extends Envelope {
  t: T;
  id: string;
  content: C & Identity;
}

// --- record categories -----------------------------------------------------

export type RecordCategory =
  | 'tool_call'
  | 'skill_invoke'
  | 'channel_write'
  | 'subagent_spawn'
  | 'subagent_report'
  | 'module_query'
  | 'sleep'
  | 'finish'
  | 'unknown';

// The `payload` union is discriminated on `category`: narrowing `category`
// narrows `payload` with it.
export type RecordStarted =
  | { category: 'tool_call';       payload: { provider: string; tool: string; arguments: Record<string, unknown> } }
  | { category: 'skill_invoke';    payload: { skill: string; resource?: string } }
  | { category: 'channel_write';   payload: { module: string; channel: string; payload?: unknown } }
  | { category: 'subagent_spawn';  payload: { agent_id: string; depth: number; task: string } }
  | { category: 'subagent_report'; payload: { agent_id: string; task: string; status: string; body: string } }
  | { category: 'module_query';    payload: Record<string, never> }
  | { category: 'sleep';           payload: { seconds: number } }
  | { category: 'finish';          payload: Record<string, never> }
  | { category: 'unknown';         payload: { verb: string } };

// --- detail entries --------------------------------------------------------

export type RecordEntry =
  | { kind: 'text';  text: string }
  | { kind: 'item';  text: string }
  | { kind: 'field'; label: string; value: string }
  | { kind: 'code';  text: string };

// --- events ----------------------------------------------------------------

// hello is the baseline handshake: it does not go through the sink (no `id`),
// and `seq` is mandatory.
export interface HelloEvent extends Envelope {
  t: 'hello';
  seq: number;
  content: {
    boot: string;
    model: string;
    base_url: string;
    status: Status;
  } & Identity;
}

export type ServerEvent =
  | HelloEvent
  | Payload<'status', { state: StatusState }>
  | Payload<'user_input', { text: string; mid?: string }>
  | Payload<'record_started', RecordStarted>
  | Payload<'record_detail', { entries: RecordEntry[] }>
  | Payload<'record_done', { result?: Record<string, unknown>; duration_s?: number }>
  | Payload<'record_failed', { error: { type: string; message: string } }>
  | Payload<'record_void', Identity>
  | Payload<'output_started', Identity>
  | Payload<'output_delta', { text: string }>
  | Payload<'output_done', { duration_s: number }>
  | Payload<'output_cancelled', Identity>;

// mid: the client message id, used for ack matching and server-side dedup
// (exactly-once under reconnect re-send).
export type ClientEvent = { t: 'input'; text: string; mid?: string } | { t: 'ping' };
```

## 11. Migration

What changes when this spec is implemented (no code is changed by this document):

**Backend — keeps emitting, stops formatting.** The emission API and envelope are
unchanged (`Sink.emit(t, content=…)` in `backend/nan_itself/events.py`); only the
`content` shape changes:

- `kind` + `name` + `summary` → `category` + typed `payload` (§6). `_RECORD_KINDS` in
  `agent/core.py` becomes a verb → category table; the fallback becomes `unknown` with
  `verb`.
- `record_detail.line` (a string) → `entries[]` typed entries (§7). The `_pretty_args`
  JSON block becomes a `code` entry; harvested `id: …` / `task: …` / `status: …` lines
  become `field` entries.
- Formatted durations (`note` in `agent/core.py`, `duration` in `agent/engine.py`) →
  numbers (§8).
- `_compact_result` — which today rewrites an MCP tool result into a display sentence —
  stops reformatting; the raw result travels as prose/`code`.

**Frontend — drops its string parsing.**

- `frontend/app/src/design/RecordItem.tsx` loses its `line.startsWith('· ')` prefix
  stripping and its numeric-highlight regex; both become unnecessary once entries are
  typed and durations are numbers. Rendering switches to `entry.kind` + `category`.
- The record `Item` in `frontend/app/src/state/store.ts` carries `category`/typed
  entries instead of a `kind` string and a `detail: string[]`; `duration` changes type
  from `string` to `number`.
- `frontend/app/src/protocol.ts` adopts §10 and drops the v1 `error` kind (the backend
  never emitted it) and the `verb` glyph fallback.

**v1 corrections folded in above.** The rewritten spec does not carry forward: v1's
`verb` kind with glyph `•` (not implemented), v1's 200-event folded replay claim (the
code replays the raw 500-event ring), or any glyph table (glyphs/colours are
frontend-owned). The `frontend/app/README.md` mock-replay description is out of scope
for this document and is left for a separate fix.
