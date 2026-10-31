# Frontend protocol

The contract between the core process and the web frontend:
one WebSocket, one event stream. The frontend is a renderer of
this stream; it holds no agent state of its own.

## Transport envelope

Events emitted through the sink share this envelope:

```json
{
  "t": "turn_done",
  "seq": 4127,
  "ts": 1761192000.123,
  "id": "e3f1...",
  "boot_id": "b7c2...",
  "content": { "agent_id": "root", "turn_index": 3, "...": "..." }
}
```

- `t` -- event type; the frontend dispatches on it
- `seq` -- process-wide monotonic counter (gap = missed events,
  reconnect)
- `ts` -- wall-clock emission time, unix seconds (float)
- `id` -- event id; the `output_*` events of one stream share
  the same id
- `boot_id` -- core process instance, stamped when the sink
  knows it; `hello.content.boot` carries the same value
- `content` -- the payload; the protocol lives HERE, per event
  type, decoupled from the envelope

The gateway builds `hello` and the `surface_*` events itself.
Those carry `t` and `content` (plus `seq` on `hello`) and are
described in their own sections below. The envelope is transport;
payload schemas are the contract.

## Hello snapshot

On connect the gateway sends `hello` with the current `seq` and
gateway-level state, then replays the history ring; events after
that are live. The cursor lives on the events: the frontend drops
anything at or below the last applied `seq`, so a replay can
overlap what it already has.

```json
{ "t": "hello", "seq": 1,
  "content": { "boot": "<id>", "model": "<name>", "paused": false } }
```

`boot` changes when the core restarts: the frontend drops history
and starts over. `paused` is the root agent's current pause state,
falling back to the newest `agent_paused`/`agent_resumed` in
history when the core does not supply it, so a reconnect while
paused still shows the right button.

## Surfaces

A service declares its own panel to the gateway

```
POST   /surfaces          {service_id, url, title, icon?}
DELETE /surfaces/{id}
```

`POST /surfaces` is a declaration and the heartbeat in one call:
the body carries the full spec, the call is an upsert keyed by
`service_id`, and repeating it both refreshes the entry and
re-states what the panel is. A service that restarts on another
port is re-declared by the loop it already runs, so liveness and
spec stay in step.

```json
{ "t": "surface_added",
  "content": { "service_id": "record",
               "url": "http://127.0.0.1:9002/ui",
               "title": "record",
               "icon": "http://127.0.0.1:9002/icon.svg" } }

{ "t": "surface_removed", "content": { "service_id": "record" } }
```

- `surface_added` is an upsert: a changed spec (new port, title,
  icon) updates the existing window in place under the same
  `service_id`, with no `surface_removed` in between. The desktop
  keeps the window's rail position and updates the spec under the
  same id; the iframe reloads only when its `url` changes.
- Re-post every 10-15 s. An entry without a POST for
  `gateway.surface_ttl` (30 s) is dropped and `surface_removed` is
  broadcast.
- Both registry calls answer `{"status": "ok"}`; a non-loopback
  panel `url` is rejected with 400, and a DELETE of an unknown
  `service_id` answers 404.
- These messages are gateway-built live state: they carry no
  `seq` and no `id`. The transport handles them before the hub
  and re-derives windows from them after a reconnect -- the
  gateway replays the current table right after `hello`, so
  every connection ends up with the same set.
- `surface_removed` names the service: the desktop drops the
  window and its queue slot. An expiry and a DELETE both produce
  it, and a later declaration with the same `service_id` adds the
  window again.
- The panel is a plain loopback page with no desktop bridge: it
  talks to its own backend directly (fetch / SSE / WebSocket)
  and the backend allows that opaque origin. What the desktop
  hands it is a URL, a sandbox, and the theme.

## Input

The frontend sends:

```json
{ "t": "input", "mid": "<client-generated id>", "text": "..." }
```

`mid` comes back on the `user_input` event; the gateway drops
repeated `mid` values.

Pause and resume travel over the socket as client messages:

```json
{ "t": "pause" }
{ "t": "resume" }
```

They take effect asynchronously; the confirmation is the
`agent_paused` / `agent_resumed` event on the same socket.

## Payload schemas

Conventions:

- every payload that belongs to an agent carries `agent_id`,
  filled at the emission point; `user_input` is the exception -- it
  carries no `agent_id`, and it marks root input from the user
- optional fields are omitted; fields documented as always present
  carry null instead
- `verb_result.text` is exactly the text the model received;
  the only truncation is `shell_output_limit`, applied by the
  shell verb itself (head + tail kept); `output_delta.text` is
  one untruncated stream chunk

### Lifecycle

| t | content |
|---|---|
| `user_input` | `{text, mid?}` |
| `agent_started` | `{agent_id, depth, parent_id, pacing}` -- `parent_id` is always present, null for the root; pacing ∈ `event-driven` \| `interval` |
| `agent_paused` | `{agent_id}` |
| `agent_resumed` | `{agent_id}` |
| `agent_exited` | `{agent_id, reason, report}` -- reason ∈ `finished` \| `killed by holder <id>` \| `cancelled` \| `shutdown`; `report` is always present, null unless finished |

### Turn

| t | content |
|---|---|
| `turn_started` | `{agent_id, turn_index, inbox_count}` -- turn_index = cycles+1, unique per agent |
| `injections_collected` | `{agent_id, services: [{service_id, ok, duration_s, text_len}]}` -- one event per turn, aggregate over services |
| `turn_done` | `{agent_id, turn_index, usage?, latency_s, retries, calls, history_chars}` -- usage = `{prompt_tokens, completion_tokens}` |
| `turn_failed` | `{agent_id, error, backoff_s, retry_index}` |
| `turn_cancelled` | `{agent_id, requeued_count}` -- pause or death mid-turn; the turn is discarded whole |

### Model stream (correlated by envelope `id`)

| t | content |
|---|---|
| `thinking_started` | `{agent_id, depth}` |
| `thinking_delta` | `{agent_id, depth, text}` |
| `thinking_done` | `{agent_id, depth}` |
| `output_started` | `{agent_id, depth}` |
| `output_delta` | `{agent_id, depth, text}` |
| `output_done` | `{agent_id, depth}` |
| `output_cancelled` | `{agent_id, depth}` |

Each family carries its own envelope `id` (stream_id) and follows
started -> delta* -> done | cancelled. `thinking_*` is reasoning
streamed before the answer; it opens lazily on the first thinking
chunk, and `output_*` opens lazily on the first text chunk, so the
two render in arrival order. `output_cancelled` closes the text
stream when the model returned tool calls, and the turn continues
into the verbs; turn-level cancellation is the separate
`turn_cancelled` event. Empty assistant answers still get an
`output_started` + close pair at the end of the turn.

### Verbs (linear per call, no correlation id)

| t | content |
|---|---|
| `verb_started` | `{agent_id, verb, arguments}` |
| `verb_failed` | `{agent_id, verb, error: {type, message}}` |
| `verb_result` | `{agent_id, verb, text, detail?}` |
| `verb_done` | `{agent_id, verb, duration_s}` |

A call is started -> (result, done) | failed. `arguments` is the
raw model-supplied object; `detail` carries structured domain
facts and is omitted when a verb has none (validation
failures, for instance, are a plain-text result the model
reads).

| verb | `arguments` | `detail` | renders as |
|---|---|---|---|
| `shell` | `{command}` | `{exit_code, timed_out}`; on a deny-list hit `{refused, reason}`, on a timeout only `{timed_out: true}` | terminal block |
| `spawn` | `{task}` | `{child_id, depth}` | child-agent card linked to `child_id` |
| `send` | `{direction, handle_id?, text}` | `{direction, to?}` -- `to` only for `down` | message bubble, up/down |
| `kill` | `{handle_id}` | `{target_id}` | destructive action marker |
| `sleep` | `{seconds}` | `{seconds}` | timer |
| `finish` | `{report}` | -- (the report travels in `agent_exited`) | report card |

The event family is shared; the verb name rides in `verb`. Dispatch
on it, and render an unknown name as raw JSON. Spawn also triggers
the child's own `agent_started`; the spawn call itself still emits
`verb_result` / `verb_done` like any other.

## Timeline reconstruction

The stream is sufficient to rebuild the full UI state:

- agents: from `agent_started` / `agent_exited`
- per-agent turn list: from `turn_started` / `turn_done` /
  `turn_failed` / `turn_cancelled`, keyed by
  `(agent_id, turn_index)`
- live output: from `output_*` keyed by envelope `id`
- activity trace: from `verb_*`
- system health: from `injections_collected`

These schemas are all the frontend needs to know about agent
semantics; depth and parent_id arrive as data.
