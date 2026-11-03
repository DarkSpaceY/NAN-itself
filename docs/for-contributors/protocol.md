# Protocol

The core is one process with two HTTP servers, one shared event bus,
and files that outlive it. This page states the contracts that make
up its backend surface: transport, services, panels, the model,
snapshots, and boot scripts.

[frontend-protocol.md](frontend-protocol.md) stays the renderer's
contract: envelope payload schemas, stream correlation, and timeline
reconstruction. This page covers the transport from the backend side,
then the protocols that file does not cover: service registration and
delivery, surface registration, what the model sees, snapshots, and
the autostart environment.

| Section | Applies to |
|---------|------------|
| Frontend / event protocol | desktop, any event consumer |
| Service registration | injection and archive services |
| Service delivery | injection and archive services |
| Surface registration | services that declare a panel |
| Model-visible protocol | prompt assembly and the model's wire shape |
| Snapshot protocol | core restart |
| Autostart environment | boot scripts and the services they launch |

## Frontend / event protocol

The sink is the process-wide event bus. Every event emitted through
it carries the same envelope:

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

- `t` -- event type
- `seq` -- process-wide monotonic counter
- `ts` -- emission time, unix seconds
- `id` -- event id; one model stream shares its id across chunks
- `boot_id` -- core process instance
- `content` -- the payload

The stream is lossy. A consumer drops events at or below its last
applied `seq` and treats a gap as a reason to reconnect. The history
ring holds the last 4000 events; streamed deltas are coalesced there,
so the ring holds one entry per stream.

The gateway builds `hello` and `surface_*` itself. `surface_*` events
carry no `seq` and no `id`; `hello` carries `seq` because it also
reports the cursor.

### Connection

```
ws://127.0.0.1:8765/ws
```

Opening sequence, in order:

1. `hello` -- `{"t": "hello", "seq": <current>, "content": {"boot": "<boot_id>", "model": "<name>", "paused": <bool>}}`
2. one `surface_added` per registered panel
3. the history ring
4. live events

A changed `boot` means the core restarted: drop history and start
over.

Client messages:

| `t` | Fields | Effect |
|-----|--------|--------|
| `input` | `text`, `mid`? | one input line to the root agent |
| `pause` | -- | pause the root agent |
| `resume` | -- | resume the root agent |
| `ping` | -- | reply `{"t": "pong", "content": {}}` |

`text` is stripped; empty input is ignored. `mid` is an optional dedup
key: a repeated `mid` is dropped, and the cache holds
`gateway.dedup_cache_size` entries (default `256`), evicting the
oldest half on overflow. A non-string `mid` counts as absent. Invalid
JSON, non-object bodies, and unknown `t` values are ignored without a
reply.

`pause` and `resume` have no acknowledgement; their effect arrives as
`agent_paused` / `agent_resumed` on the same socket.

### Event families

| Family | `t` values |
|--------|-----------|
| Lifecycle | `user_input`, `agent_started`, `agent_paused`, `agent_resumed`, `agent_exited` |
| Turn | `turn_started`, `injections_collected`, `turn_done`, `turn_failed`, `turn_interrupted`, `snapshot_restored` |
| Model stream | `thinking_started` / `thinking_delta` / `thinking_done`, `output_started` / `output_delta` / `output_done` / `output_cancelled` |
| Verbs | `verb_started`, `verb_failed`, `verb_result`, `verb_done` |
| Gateway-built | `hello`, `surface_added`, `surface_removed` |

Payload schemas and reconstruction rules live in
[frontend-protocol.md](frontend-protocol.md).

## Service registration

Services register with the injection hub. Registration is plain HTTP
on the hub's own port (`injection.port`, default `8901`).

```
POST /register     {service_id, endpoint, mode?}
POST /heartbeat    {service_id}
POST /unregister   {service_id}
GET  /services
```

All four answer `200`; the outcome is in `status`. A loopback violation
is `403`.

`POST /register`

| Field | Rule |
|-------|------|
| `service_id` | required, non-empty |
| `endpoint` | required, absolute `http(s)` loopback URL |
| `mode` | optional, `inject` (default) or `archive` |

| Outcome | Body |
|---------|------|
| accepted | `{"status": "registered"}` |
| missing id or endpoint | `{"status": "rejected", "reason": "service_id and endpoint required"}` |
| unknown mode | `{"status": "rejected", "reason": "mode must be inject or archive"}` |
| non-loopback endpoint | `{"status": "rejected", "reason": "endpoint must be loopback"}` |

Re-registering an existing id replaces its endpoint, mode, and
freshness in one step.

`POST /heartbeat` refreshes the entry and returns `{"status": "ok"}`,
or `{"status": "unknown"}` when the id is not registered (it does not
implicitly register). `POST /unregister` removes the entry, is
idempotent, and always returns `{"status": "ok"}`.

`GET /services` returns every entry currently held:

```json
{ "services": [
  { "service_id": "record", "endpoint": "http://127.0.0.1:9002/turn",
    "mode": "archive", "last_seen": 1761192000.123 } ] }
```

Freshness is not applied on read: an entry stays listed until a
delivery pass evicts it. An entry is live when `last_seen` is within
`injection.heartbeat_ttl` (default `15` seconds).

## Service delivery

The core pushes one turn per request to each live service endpoint,
as a flat JSON body. The mode decides the shape:

| Mode | When | Body | Reply |
|------|------|------|-------|
| `inject` | before each prompt assembly | fresh turn | `{"text": "<prompt text>"}` |
| `archive` | after the turn ends and commits | complete turn | any `2xx`; body ignored |

The fresh shape has `agent_id`, `parent_id`, `depth`, `epoch`,
`origin`, `persona`, `inbox`, `injections`, `delta`, `started_at`.

The complete shape adds `phase`, `reply`, `thinking`, `calls`,
`results`, `llm`, `death`, `rendered`, `ended_at`. Field meanings and
examples live in
[injection-service.md](../for-service-authors/injection-service.md#the-turn-json).

Rules:

- Every live service of the matching mode is called concurrently. One
  slow service does not serialize the rest; the turn waits for all.
- Both modes share `injection.timeout` (default `2` seconds).
- An `inject` reply must be a `2xx` JSON object whose `text` is a
  string. Anything else is a failed delivery recorded as
  `injections[].ok == false`, with `text` set to the exception. A
  successful empty `text` is not recorded.
- An `archive` failure is logged and dropped; the committed turn never
  changes.
- A service that stops heartbeating is excluded after `heartbeat_ttl`,
  without an unregister call.
- The archive sees every turn that commits, in the order it commits.
  A turn that never commits (kill, pause, shutdown, unrecoverable
  failure) never reaches it; a finished turn is archived with its death
  record.

### Service delta

Each agent learns about service transitions at its own turn boundary.
The hub diffs the live set against the set this agent saw last and
appends `DeltaEvent("service", line)` entries:

```
+record online
-weather offline
```

Both modes appear here. The diff and the live set come from one
snapshot: an `inject` service reported online received the fresh turn;
an `archive` service reported online receives that turn only if it
commits.

## Surface registration

A service declares its panel to the gateway (default port `8765`). The
declaration is also its heartbeat.

```
POST   /surfaces              {service_id, url, title, icon?}
DELETE /surfaces/{service_id}
```

| Field | Rule |
|-------|------|
| `service_id` | required, non-empty; the upsert key |
| `url` | required, absolute `http(s)` loopback URL |
| `title` | required, non-empty |
| `icon` | optional; a local path (`/...`) or a loopback URL |

| Outcome | Status | Body |
|---------|--------|------|
| accepted, unchanged spec | `200` | `{"status": "ok"}` |
| accepted, new or changed spec | `200` | `{"status": "ok"}` |
| invalid spec | `400` | `{"status": "rejected", "reason": "..."}` |
| `DELETE` for an absent id | `404` | `{"status": "unknown"}` |
| `DELETE` for a present id | `200` | `{"status": "ok"}` |

`POST` refreshes the entry and re-states the spec. A changed spec
replaces the panel in place under the same `service_id`; the desktop
receives `surface_added` again and keeps the window's position. An
unchanged spec only refreshes the heartbeat and emits nothing.

An entry with no POST for `gateway.surface_ttl` (default `30` seconds)
is dropped and `surface_removed` is broadcast. A `DELETE` withdraws it
immediately with the same event. Surface state is not part of event
history: the gateway replays the current table to each new connection.

## Model-visible protocol

The model sees three layers: the system prompt, the committed
history, and the open turn's observation.

### Prompt assembly

- **System prompt** -- `paths.persona`, read at boot and re-read every
  turn. A content change applies from that turn on; the change is
  recorded on the turn as `persona`, but the system message always
  carries the current file.
- **History** -- one message run per committed turn of the current
  context epoch, rendered once and frozen. The open turn is not part of
  history.
- **Context epoch** -- when rendered history exceeds
  `agent.history_char_limit`, the turn list is cleared when the next
  turn opens. That turn gets `epoch + 1` and `origin =
  "context_cleared"`; only the new epoch's turns feed history.
- **Call** -- `messages = [system, *history, *render_turn(turn)]`,
  plus the verb definitions the agent's role allows.

### The observation message

`render_turn` turns one turn into messages. A turn before the model
answered renders as the observation alone. After the answer it renders
the observation, the assistant message (reply, thinking, verb calls),
and one tool message per call, paired by call id. A call with no result
renders `INTERRUPTED`:

```
interrupted: agent stopped during execution
```

The observation is one user message with up to three sections, in this
order, each omitted when empty:

```
<delta>
<service>
+record online
</service>
</delta>

<injection>
<service id="weather">
forecast text
</service>
</injection>

<inbox>
<user>
hello
</user>
</inbox>
```

When all three are empty the message is `(no input this turn)`.

`<delta>` groups `DeltaEvent`s by section in first-seen order. Sections
are `service`, `subagent` (spawned, died), and `agent` (paused, retry
exhausted).

`<injection>` holds one block per recorded service, keyed by
`service_id`. A failed delivery keeps its text prefixed with
`[service error] `.

`<inbox>` tags each message by the sender's role:

| `kind` | Rendered tag | Attributes |
|--------|--------------|------------|
| `user` | `<user>` | -- |
| `parentagent` | `<parentagent>` | -- |
| `subagent` | `<subagent>` | `id="<source>"`; `event="<event>"` when not `send` |

The spawn task arrives as a `parentagent` message; a child's finish
report arrives as a `subagent` message with `event="finish"`, and its
text is the report.

### Verbs

The verb definitions handed to the model are exactly the role's set:
a root runs `ROOT_VERBS` (no `finish`, because it stays resident); a
child runs all six. Each definition carries a name, a description, and
a JSON Schema with `additionalProperties: false`.

| Verb | Arguments | Available to | Effect |
|------|-----------|--------------|--------|
| `spawn` | `task` | root, child | create a child; its id returns in the result |
| `send` | `direction` (`up` \| `down`), `child_id` (down only), `text` | root, child | deliver text to the parent or to a spawned child |
| `shell` | `command`, `cwd`?, `env`?, `stdin`?, `background`? | root, child | run bash in the workspace; the timeout is fixed by the user, and `background` starts a job whose log later calls poll |
| `sleep` | `seconds` | root, child | wait before the next step |
| `kill` | `child_id` | root, child | kill a child; its subtree dies with it |
| `finish` | `report` | child only | submit the report upstream and die |

A malformed tool call is not fatal: its parse error becomes that
call's result, the same channel as a failed verb. Every result text is
what the model reads on the next call.

## Snapshot protocol

Snapshots are the restore contract between core runs. One file per
agent, `<agent_id>.json` under `paths.snapshots` (`snapshots/`).

```json
{
  "agent_id": "root",
  "parent_id": null,
  "depth": 0,
  "turns": [ { "...": "Turn.to_json" } ],
  "pending": [ { "...": "InboxMessage.to_json" } ]
}
```

- A write is whole-state: a temporary file is fsynced and renamed over
  the target, so a reader never sees a partial snapshot.
- The tree is not stored separately: `parent_id` and `depth` rebuild
  it. `parent_id: null` marks the root; empty `turns` means a spawned
  agent that has not observed yet and is still live.
- `pending` holds inbox messages no turn has observed yet.
- On restore, a snapshot carrying a `death` record is discarded, and so
  is every subtree whose parent is unreachable.
- The last open turn (phase before `committed`) decides what happens:
  `observed` means nothing was seen, so the turn is dropped and its
  inbox rides into the next turn; any later phase keeps the turn and
  the loop continues from that phase.
- An unreadable file is logged and skipped; a snapshot whose
  `agent_id` does not match its filename is unreadable.

## Autostart environment

The core launches two boot scripts once at startup, after the
injection hub is serving and before the gateway starts:

| Script | Rights |
|--------|--------|
| `paths.autostart` (`builtin/autostart.sh`) | user-owned, runs outside the sandbox with the user's full rights |
| `paths.agent_autostart` (`workspace/autostart.sh`) | agent-writable, runs inside the execution sandbox |

`autostart.enabled` (default `true`) turns both on. A script that does
not exist is skipped.

The core exports these variables to both scripts, on top of the full
inherited environment:

| Variable | Value |
|----------|-------|
| `NAN_CORE_HOME` | `paths.package`, the `nan` package directory |
| `NAN_WORKSPACE` | `paths.workspace`, the agent-writable workspace |
| `NAN_INJECTION_PORT` | `injection.port`, the live hub port |
| `NAN_CORE_PID` | the core process id |

`NAN_INJECTION_PORT` is the only supported way for a boot script to
find the registry; the port is a setting and must not be assumed to be
`8901`.

Each script starts in its own session, so everything it launches shares
one process group. The shell then stays alive as the group anchor, and
the core keeps the group id. At shutdown the core stops the group
gently (`SIGTERM`), waits up to 5 seconds, then firmly (`SIGKILL`),
while the hub and gateway are still up so services can unregister and
flush. Nothing started by autostart should detach; a leftover listener
would collide with the next boot.
