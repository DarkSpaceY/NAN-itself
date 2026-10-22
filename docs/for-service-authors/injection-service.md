# Writing an injection service

An injection service is an independent local process that
contributes text to every prompt assembly. It is the replacement
for the old in-process module system: no registry to import, no
hot-reload contract -- just HTTP on loopback.

## Lifecycle

```mermaid
sequenceDiagram
    participant S as Your service
    participant H as Injection hub
    participant A as Agent turn

    S->>H: POST /register {service_id, endpoint}
    H-->>S: {"status": "registered"}

    loop every few seconds (well under heartbeat_ttl)
        S->>H: POST /heartbeat {service_id}
        H-->>S: {"status": "ok"}
    end

    A->>H: collect(turn)
    par all live services, concurrently
        H->>S: POST your endpoint {turn: {...}}
        S-->>H: "text" or {"text": "..."}
    end
    H-->>A: Injection(ok=true/false, duration_s)

    opt clean shutdown
        S->>H: POST /unregister {service_id}
    end
```

1. **Register** -- on startup, POST to the hub:

       POST http://127.0.0.1:8901/register
       {"service_id": "weather", "endpoint": "http://127.0.0.1:9001/turn"}

   `service_id` is your identity in the observation stream;
   `endpoint` is the URL the hub will call every turn.

2. **Heartbeat** -- POST periodically (keep the interval well
   under `injection.heartbeat_ttl`, default 15 s):

       POST /heartbeat
       {"service_id": "weather"}

   A service silent longer than the TTL is evicted; re-register
   to come back.

3. **Serve turns** -- before every prompt assembly the hub POSTs
   the turn payload to your endpoint, concurrently across all
   live services. Reply with the text you want injected, either
   plain text or JSON `{"text": "..."}`.

4. **Unregister** (optional) -- `POST /unregister` with your
   `service_id` for a clean shutdown; otherwise the TTL evicts
   you.

Introspection: `GET /services` lists live entries
(`service_id`, `endpoint`, `last_seen`).

## The turn payload

The request body is `{"turn": {...}}` -- the full new Turn as
JSON, minus the rendered messages (they are large and you do not
need them to decide what to contribute):

    {
      "agent_id":   "root",        # who is assembling a prompt
      "parent_id":  null,          # null for the root agent
      "depth":      0,
      "inbox":      [              # messages drained since last turn
        {"kind": "user", "source": "user", "text": "...",
         "event": null, "received_at": 1760000000.0}
      ],
      "injections": [              # what services contributed last
        {"service_id": "...", "text": "...", "ok": true,
         "duration_s": 0.02}       # ok=false: their error text
      ],
      "delta":      [              # state changes since last turn
        {"section": "service", "line": "+weather online"},
        {"section": "subagent", "line": "spawned a1: researching"}
      ],
      "reply":      null           # this turn's model output; null
    }                              # until the turn completes

`injections` includes your own previous reply -- use it to keep
state across turns (you are stateless to the core; persistence is
your business, typically a file under your own directory).

```mermaid
flowchart LR
    subgraph CORE["core"]
        T["Turn n<br/>inbox / injections / delta"]
    end
    T -->|"POST on every turn"| S["your service"]
    S -->|"previous reply comes back<br/>in turn.injections"| T
    S -->|"on turn request"| F["your own state file"]
    F -->|"reload before answering"| S
    style CORE fill:transparent,stroke:#2563eb,stroke-width:2px
```

## Failure semantics

```mermaid
flowchart TD
    R["turn collects services"] --> OK{"service replies<br/>within timeout?"}
    OK -->|"yes, non-empty"| C["Injection(ok=true)<br/>text rides the prompt"]
    OK -->|"yes, empty"| N["dropped from the record"]
    OK -->|"no / error / timeout"| E["Injection(ok=false)<br/>error text on the record"]
    E --> NEXT["turn proceeds<br/>no retry, no block"]
    C --> NEXT
    N --> NEXT
```

- The hub allows `injection.timeout` (default 2 s) per service
  per turn. There is no retry.
- A failure or timeout becomes an `Injection` with `ok: false`
  and the error text, recorded in the turn -- the agent sees it.
- An empty reply contributes nothing.
- Services are never in the critical path of the turn: a dead
  service costs one timeout, then eviction.

## Ground rules

- Loopback only; the core binds `127.0.0.1` and does not
  authenticate services (trusted-machine model).
- Return promptly. If you need longer than the timeout to compute
  a contribution, compute it eagerly (e.g. on heartbeat) and
  serve the ready text on turn requests.
- Everything you contribute is prompt text for the model: keep it
  compact, factual and stable across turns -- it rides the
  context of every turn it appears in.
