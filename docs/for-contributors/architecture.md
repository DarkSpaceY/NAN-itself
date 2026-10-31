# Architecture

The core runtime lives in `nan/`; service processes share
`lib/nan-service-kit`. The tree below shows the core modules and
follows the actual layout.

    nan/
    ├── app.py            composition root
    ├── events.py         process-wide event sink (singleton)
    ├── gateway.py        web UI: HTTP + WebSocket
    ├── agent/
    │   ├── core.py       the Agent loop
    │   ├── engine.py     the turn engine (one model call)
    │   ├── handle.py     Handle + Book
    │   ├── sandbox.py    OS-level execution sandbox
    │   └── verbs/        spawn / sleep / shell / send / kill / finish
    ├── turn/
    │   ├── model.py      the Turn record
    │   └── render.py     frozen prompt rendering
    ├── llm/
    │   ├── provider.py   the model API transport
    │   └── types.py      protocol vocabulary (DTOs)
    ├── injection/
    │   ├── hub.py        registry + per-turn collection
    │   └── client.py     the concurrent collector
    ├── utils/
    │   ├── loopback.py   entry guard primitives
    │   └── autostart.py  boot scripts and process groups
    └── config/           pydantic-settings, YAML source

Any domain may import the DTOs in `nan/llm/types.py`. The
composition root constructs one `LLMProvider` (the transport edge),
and the turn engine is the module that calls it. The agent runtime
stays independent of which provider is in use.

## One agent, one task

An Agent is a single asyncio task running turns in a loop. The
loop paces itself with `autonomous_interval`: above zero, each turn
is followed by a sleep of that length; at zero, the loop waits for
inbox input, and a turn that called verbs runs again so the model can
read the results. Messages queue in the inbox and drain at the next
turn boundary.

## The turn

```mermaid
flowchart LR
    A["drain inbox"] --> B["collect injections<br/>(live inject services, concurrent)"]
    B --> C["one model call"]
    C --> D["run verbs positionally"]
    D --> E["freeze rendered messages"]
    E -->|"next turn"| A
```

A turn is the unit of record. `Turn` holds the inbox drained
into it, the injections collected, the delta lines, and the
model's reply -- the complete runtime log. The event stream
exposes the same facts live, and the turn is where they persist.

The rendered prompt is frozen on first render and later turns
append to it. History is add-only, so unchanged turns stay
byte-stable and cache-friendly for the model's prompt prefix.
Assistant messages carry the model's reasoning (thinking blocks,
signatures included) so the adapter can replay it verbatim; the
prefix cache stays valid and Anthropic/OpenRouter keep the
signed reasoning across turns. When the history exceeds
`history_char_limit`, the whole history is cleared and a new
context epoch begins.

```mermaid
flowchart LR
    T1["turn 1<br/>(frozen)"] --> T2["turn 2<br/>(frozen)"] --> T3["turn 3<br/>(frozen)"]
    T3 -->|"history_char_limit exceeded"| R["history cleared whole<br/>(new context epoch)"]
    R --> T4["turn 4<br/>(frozen)"]
```

## Handles: the multi-agent graph

```mermaid
flowchart TD
    R["root"] -->|"Handle a1<br/>send / kill"| A["a1"]
    A -->|"Handle a1.1<br/>send / kill"| C["a1.1"]
    C -.->|"self.parent<br/>send up / finish"| A
    A -.->|"self.parent<br/>send up / finish"| R
```

Solid edges are handles (downward, full authority); dotted
edges are the parent reference a child uses to talk upstream,
which carries no control right over the parent.

A Handle is a cross-agent channel plus the control right over
one agent:

- bound to exactly one target and one holder, the spawner. The
  composition root holds the root agent directly and feeds it
  through `receive_up`; every child is reached through a handle
- owned by its spawner, so the cascade-death graph follows the
  spawn tree
- through it the holder may `send()` into the target's inbox
  and `kill()` the target; the target replies upstream through
  its parent reference

The `Book` is the process-wide id -> agent registry; routing
goes through handles, and whole-tree shutdown starts at
`root.request_death` and follows the cascade. When an agent
dies it tears down everything it holds (cascade), then a finish
report goes to its parent's inbox; a bare death becomes a
`parent.note_delta` line in `<delta>`.

```mermaid
flowchart TD
    D["a1 dies"] --> D1["a1.1 killed (held handle)"]
    D --> D2["a1.2 killed (held handle)"]
    D1 --> D3["a1.1.1 killed"]
    D -.->|"death notice / finish report"| P["parent inbox"]
```


## Verbs

The six cognitive actions: `spawn`, `sleep`, `shell`, `send`,
`kill`, `finish` (root has no finish). Tools and skills are plain
files the agent reaches through `shell` inside the sandbox.

Before a command is wrapped, it passes the `sandbox.deny` rules: a
regex hit returns the rule's `reason` to the model as the shell
result, and the command does not run.

## Observations

Each turn the agent sees one structured observation:

    <delta>      state changes (services online/offline,
                 subagent spawns, ...), grouped by first
                 appearance
    <injection>  contributions from external services, labelled
                 by service_id
    <inbox>      parent/user messages and subagent finish notices

```mermaid
flowchart TB
    O["observation<br/>(one per turn)"]
    O --> D["delta<br/>state changes"]
    O --> I["injection<br/>service contributions"]
    O --> X["inbox<br/>contentful messages"]
    D --> D1["service online / offline"]
    D --> D2["subagent spawned"]
    D --> D3["subagent/child death notice"]
    I --> I1["per service_id: contributed text<br/>failures marked ok=false"]
    X --> X1["parent / user message"]
    X --> X2["subagent finish report"]
```

The boundary: state changes ride `<delta>`, contentful messages
ride `<inbox>`.

## Injection

```mermaid
sequenceDiagram
    participant S as Service process
    participant H as Injection hub
    participant A as Agent turn

    S->>H: POST /register {service_id, endpoint, mode}
    loop heartbeat interval
        S->>H: POST /heartbeat {service_id}
    end
    A->>H: inject(turn)
    par concurrent, per live inject service
        H->>S: POST endpoint <fresh turn>
        S-->>H: {"text": "<str>"} (plain text => ok=false)
    end
    H-->>A: one result per service<br/>(ok, timing, text length)
    note over A: contributions and failures<br/>both land on the turn record
    A->>H: archive(turn), when the turn ends
    par concurrent, per live archive service
        H->>S: POST endpoint <complete turn>
        S-->>H: 2xx, acknowledgement only
    end
```

Services are independent processes registered with the hub
(see for-service-authors/injection-service.md). Each entry
declares a mode: `inject` entries receive the fresh Turn before
every prompt assembly and their text lands on the turn record;
`archive` entries receive the complete Turn once it ends and
their reply is an acknowledgement. Both modes feed the service
online/offline delta.

## Events

```mermaid
flowchart LR
    AG["agent code"] -->|"sink.emit(...)"| S["sink<br/>(process-wide singleton)"]
    S -->|"live copy"| GW["gateway subscribers"]
    S -->|"every event"| HR["bounded history ring<br/>(reconnect snapshots)"]
```

`nan.events.sink` is a process-wide singleton (loguru-style):
agent code emits through `sink.emit`, and the gateway subscribes
on its own. Every event also enters a bounded history ring, which
supplies the snapshot a client receives on reconnect. Modules
receive the sink as an argument or import it directly.

## Gateway

The gateway formats the event stream for the web UI and carries
user input back to the root agent's inbox. It owns the surface table
and the input dedup cache; agent semantics stay with the runtime.
It binds loopback and checks Host/Origin on entry (see
[SECURITY.md](../../SECURITY.md)).
