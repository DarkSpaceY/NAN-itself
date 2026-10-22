# Architecture

One runtime package, `nan/`. Every module below is a real file
in it; this document follows the actual layout.

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
    │   ├── provider.py   the only module that talks to model APIs
    │   └── types.py      protocol vocabulary (DTOs)
    ├── injection/
    │   ├── hub.py        registry + per-turn collection
    │   └── client.py     the concurrent collector
    └── config/           pydantic-settings, YAML source

## One agent, one task

An Agent is a single asyncio task running turns in a loop. The
loop paces itself with `autonomous_interval` (or waits for input
when it is 0). Messages never interrupt an in-flight turn: they
queue in the inbox and drain at the next turn boundary.

## The turn

```mermaid
flowchart LR
    A["drain inbox"] --> B["collect injections<br/>(all live services, concurrent)"]
    B --> C["one model call"]
    C --> D["run verbs positionally"]
    D --> E["freeze rendered messages"]
    E -->|"next turn"| A
```

A turn is the unit of record. `Turn` holds the inbox drained
into it, the injections collected, the delta lines, and the
model's reply -- a complete runtime log with no separate trace
facility. The rendered prompt is frozen on first render and
never recomputed: history is add-only, so unchanged turns are
byte-stable and cache-friendly for the model's prompt prefix.
When the history exceeds `history_char_limit` it is cleared
whole -- a reset, not a rewrite.

```mermaid
flowchart LR
    T1["turn 1<br/>(frozen)"] --> T2["turn 2<br/>(frozen)"] --> T3["turn 3<br/>(frozen)"]
    T3 -->|"history_char_limit exceeded"| R["history cleared whole<br/>(new base, not a rewrite)"]
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
edges are the parent reference a child uses to talk upstream
(no handle, so a child cannot kill its parent).

A Handle is a cross-agent channel plus the control right over
one agent:

- bound to exactly one target and one implicit holder (the
  spawner; the root's handle is held by the composition root)
- not transferable: the cascade-death graph is exactly the
  spawn tree
- through it the holder may `send()` into the target's inbox
  and `kill()` the target; the target replies upstream through
  its parent reference

The `Book` is the process-wide id -> agent registry used for
routing, status snapshots and whole-tree shutdown. When an agent
dies it tears down everything it holds (cascade), then its
parent's inbox receives a finish report or a death notice.

```mermaid
flowchart TD
    D["a1 dies"] --> D1["a1.1 killed (held handle)"]
    D --> D2["a1.2 killed (held handle)"]
    D1 --> D3["a1.1.1 killed"]
    D -.->|"death notice / finish report"| P["parent inbox"]
```


## Verbs

The six cognitive actions: `spawn`, `sleep`, `shell`, `send`,
`kill`, `finish` (root has no finish). Tools and skills are not
verbs -- they are plain files the agent reaches through `shell`
inside the sandbox.

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
    I --> I1["per service_id: contributed text<br/>failures marked ok=false"]
    X --> X1["parent / user message"]
    X --> X2["subagent finish or death notice"]
```

The boundary: state changes ride `<delta>`, contentful messages
ride `<inbox>`.

## Injection

```mermaid
sequenceDiagram
    participant S as Service process
    participant H as Injection hub
    participant A as Agent turn

    S->>H: POST /register {service_id, endpoint}
    loop heartbeat interval
        S->>H: POST /heartbeat {service_id}
    end
    A->>H: collect(turn)
    par concurrent, per live service
        H->>S: POST endpoint {turn: {...}}
        S-->>H: text or {"text": "..."}
    end
    H-->>A: Injection(ok, duration_s)
    note over A: contributions and failures<br/>both land on the turn record
```

Services are independent processes registered with the hub
(see for-service-authors/injection-service.md). Before every
prompt assembly the hub POSTs the new Turn to every live service
concurrently; contributions, failures and timeouts all land on
the turn record.

## Events

```mermaid
flowchart LR
    AG["agent code"] -->|"sink.emit(...)"| S["sink<br/>(process-wide singleton)"]
    S -->|"fan-out"| GW["gateway subscribers"]
    S -->|"no subscribers"| HR["bounded history ring<br/>(reconnect snapshots)"]
```

`nan.events.sink` is a process-wide singleton (loguru-style):
agent code only *emits* (`sink.emit`), the gateway subscribes on
its own; with no subscribers events simply land in a bounded
history ring for reconnect snapshots. Nothing is wired by
reference.

## Gateway

The gateway formats the event stream for the web UI and carries
user input back to the root agent's inbox. It owns nothing about
agent semantics. Loopback-only with a Host/Origin entry guard
(see for-users/security.md).
