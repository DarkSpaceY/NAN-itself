# Principles

The design rules the codebase is written against. When a change
feels ambiguous, these decide.

## Shell-first

The model already knows the shell. Tools, skills, data
processing, service management -- all of it is reachable through
`shell` as plain files. There is no tool registration, no plugin
loader, no hot-reload machinery to rebuild: the filesystem is
the ecosystem. New capabilities are added by writing files into
`builtin/` (read-only resources) or `workspace/` (agent-writable).

## One loop per agent

No thread pools of agents, no scheduler. An agent is one asyncio
task with one inbox. Concurrency comes from spawning subagents;
coordination comes from handles. This keeps death, pause and
cascade semantics exact.

## The turn is the record

Everything that happened in a turn -- inbox drained, injections
collected, deltas observed, the reply -- lives on the Turn. No
separate trace, no side-channel logging to reconstruct behavior.
LLM calls are metered on the record (`LLMCall`: model, usage,
latency, retries).

## Add-only context

```mermaid
flowchart LR
    P["system prompt<br/>(persona; changes invalidate the prefix)"]
    P --> T1["turn 1"]
    T1 --> T2["turn 2"]
    T2 --> T3["turn 3"]
    T3 -.->|"append only:<br/>existing bytes unchanged"| SOLID["stable prompt prefix<br/>= cache hit"]
```

Rendered turns freeze on first render and never change. History
grows by appending; when the char limit trips, it clears whole.
Unchanged prefixes stay byte-stable, which is what makes prompt
caching work. There is no in-place rewriting of history.

## Core files vs agent territory

```mermaid
flowchart TB
    U["user"] -->|"writes"| CORE["nan/<br/>code, persona, config"]
    U -->|"writes"| BI["builtin/<br/>resources"]
    U -->|"writes"| AU["workspace/autostart.sh<br/>(user-owned)"]
    A["agent"] -->|"shell: read-only"| CORE
    A -->|"shell: read-only"| BI
    A -->|"shell: write"| WS["workspace/<br/>(except autostart.sh)"]
    CORE -.->|"boots and governs"| A
```

Three zones: the nan package (core code, persona -- the agent
cannot write these), `builtin/` (read-only resources), and
`workspace/` (the agent's writable territory). One exception
inside the workspace: `autostart.sh` is user-owned -- it boots
the machine-level components, so the agent must never write it.

## One path for everything important

Death has one path (finish or kill, both through the same
teardown). Configuration has one source (the YAML). Event flow
has one singleton (sink). Duplicated paths drift; when two
mechanisms could do a job, the codebase picks one and deletes
the other.

## Loud failure

A missing persona fails startup. A failed turn is retried on
the backoff ladder and every failure is recorded. A sandbox
missing its tooling warns loudly. Silence hides decay; the
system is designed so that breakage is visible at the point it
happens.
