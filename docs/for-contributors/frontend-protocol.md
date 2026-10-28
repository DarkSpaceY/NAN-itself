# Frontend protocol

The contract between the core process and the web frontend:
one WebSocket, one event stream. The frontend is a renderer of
this stream; it holds no agent state of its own.

## Transport envelope

The sink layer stamps every event before the gateway forwards
it:

```json
{
  "t": "turn_done",
  "seq": 4127,
  "ts": 1761192000.123,
  "id": "e3f1...",
  "content": { "agent_id": "root", "turn_index": 3, "...": "..." }
}
```

- `t` -- event type; the frontend dispatches on it
- `seq` -- process-wide monotonic counter (gap = missed events,
  reconnect)
- `ts` -- wall-clock emission time, unix seconds (float)
- `id` -- event id: every event carries one; the `output_*`
  events of one stream share the same id
- `content` -- the payload; the protocol lives HERE, per event
  type, decoupled from the envelope

The envelope is transport; it is not normalized or re-shaped.
Payload schemas are the contract.

## Hello snapshot

On connect the gateway sends `hello` with the current `seq` and
gateway-level state, then replays the history ring; events after
that are live. `seq` is used for dedup and gap detection, not
for skipping the replay.

```json
{ "t": "hello", "seq": 1,
  "content": { "boot": "<id>", "model": "<name>", "paused": false } }
```

`boot` changes when the core restarts: the frontend drops history
and starts over. `paused` is a snapshot derived from the newest
`agent_paused`/`agent_resumed` in history, so a reconnect while
paused still shows the right button.

## Surfaces

A service declares its own panel to the gateway

```
POST   /surfaces          {service_id, url, title, icon?}
DELETE /surfaces/{id}
```

`POST /surfaces` is a declaration, not a registration followed by
a separate heartbeat: the body carries the full spec, the call is
an upsert keyed by `service_id`, and repeating it both refreshes
the entry and re-states what the panel is. A service that restarts
on another port is re-declared by the loop it already runs, so
liveness and spec do not drift apart.

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
  keeps the window's rail position and only swaps its spec, so the
  panel is not torn down and rebuilt.
- Re-post every 10-15 s. An entry without a POST for
  `gateway.surface_ttl` (30 s) is dropped and `surface_removed` is
  broadcast.
- Both registry calls answer `{"status": "ok"}`; a panel `url`
  that is not loopback is rejected with 400.
- These messages carry no `seq`: they are live state, not
  history. The transport handles them before the hub and
  re-derives windows from them after a reconnect -- the gateway
  replays the current table right after `hello`, so every
  connection ends up with the same set.
- `surface_removed` names only the service: the desktop drops
  the window and its queue slot. A service that stops posting is
  dropped after `gateway.surface_ttl`, and `surface_removed` is
  broadcast; a later declaration with the same `service_id` adds
  it again.
- The panel is a plain loopback page with no desktop bridge: it
  talks to its own backend directly (fetch / SSE / WebSocket)
  and the backend allows that opaque origin. The desktop only
  hands it a URL and a sandbox.

## Input

The frontend sends:

```json
{ "t": "input", "mid": "<client-generated id>", "text": "..." }
```

`mid` comes back on the `user_input` event; the gateway drops
repeated `mid` values.

Pause and resume are socket messages too, not HTTP endpoints:

```json
{ "t": "pause" }
{ "t": "resume" }
```

They take effect asynchronously; the confirmation is the
`agent_paused` / `agent_resumed` event on the same socket.

## Payload schemas

Conventions:

- every payload that belongs to an agent carries `agent_id`
  (filled at the emission point, not by the envelope);
  `user_input` is the exception -- it is root input from the
  user, not something an agent emitted
- optional fields are omitted, not null, unless stated
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
| `output_started` | `{agent_id, depth}` |
| `output_delta` | `{agent_id, depth, text}` |
| `output_done` | `{agent_id, depth}` |
| `output_cancelled` | `{agent_id, depth}` |

All four carry the same envelope `id` (stream_id); a stream is
exactly started -> delta* -> done | cancelled.

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

| verb | `arguments` | `detail` on success | renders as |
|---|---|---|---|
| `shell` | `{command}` | `{exit_code, timed_out}` | terminal block |
| `spawn` | `{task}` | `{child_id, depth}` | child-agent card linked to `child_id` |
| `send` | `{direction, handle_id?, text}` | `{direction, to?}` -- `to` only for `down` | message bubble, up/down |
| `kill` | `{handle_id}` | `{target_id}` | destructive action marker |
| `sleep` | `{seconds}` | `{seconds}` | timer |
| `finish` | `{report}` | -- (the report travels in `agent_exited`) | report card |

Verbs are verbs, not event types: dispatch on `verb`, and fall
back to raw JSON for an unknown name. Spawn also triggers the
child's own `agent_started`; the spawn call itself still emits
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

No event requires the frontend to know agent semantics beyond
these schemas; depth and parent_id are data, not behavior.
