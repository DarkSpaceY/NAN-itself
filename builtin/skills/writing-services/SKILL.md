---
name: writing-services
description: >-
  How to write an injection service for NAN-itself: an independent
  local process that registers with the core and contributes prompt
  text at every turn boundary. Covers the HTTP contract, the project
  layout, the facts-only discipline, graceful degradation, tests, and
  wiring into autostart. Use when creating, fixing, or extending a
  service under workspace/services/.
---

# Writing an Injection Service

A service is **one uv project** in `workspace/services/<name>/`. It is
an ordinary local process — NOT imported by the core, NOT in-process.
The core never links service code; the only contract is HTTP.

```
<name>/
├── main.py          # entry: probe + FastAPI app (required)
├── config.yaml      # tunables, loaded once at startup (optional)
├── pyproject.toml   # its own dependencies + [dependency-groups] dev = ["pytest"]
├── uv.lock          # committed; CI runs with --frozen
└── tests/           # its own tests; the core repo never tests service internals
```

## The protocol (nan/injection/hub.py is the source of truth)

1. On startup, POST to the hub (default `http://127.0.0.1:8901`):

   ```json
   POST /register  {"service_id": "<name>", "endpoint": "http://127.0.0.1:<port>/turn",
                    "mode": "inject"}
   ```

   `mode` is `inject` (default) or `archive` (see step 3).

2. Heartbeat `POST /heartbeat {"service_id": "<name>"}` more often
   than every 15 s. After `injection.heartbeat_ttl` (15 s) a stale
   service stops receiving turns; the entry itself is not deleted.
   A dead heartbeat simply means the next turn runs without you —
   never blocks anything.
3. Serve the shape your `mode` asks for. An `inject` service
   receives the fresh turn — a flat JSON object — before every
   prompt assembly and has **2 seconds** to reply
   `{"text": "<str>"}`; that text rides the turn as an Injection,
   and plain text or any other shape is a failed delivery
   (`ok=False`). An `archive` service receives the complete turn
   (adds `rendered`, `reply`, `calls`, `results`, `llm`, `death`,
   `ended_at`) once the turn ends; any 2xx is enough. A non-2xx,
   timeout, or crash is recorded on the Turn for `inject` and
   logged then dropped for `archive` — never fatal.
4. On shutdown `POST /unregister`. Use a FastAPI lifespan to do
   register → run → unregister so the sequence always holds.

## Discipline

- **Facts only.** Every line you emit must be a directly verifiable
  measurement or a deterministic event. No heuristics, no
  conclusions — the agent judges, the service measures.
- **Degrade, never die.** Every platform probe returns None on
  failure; a missing binary or denied syscall silences one line, not
  the service.
- **Sample off-thread, reply from cache.** Probes run on a background
  cadence; `/turn` returns the cached rendering immediately. Sampling
  inside `/turn` risks the 2 s timeout.
- **Diff, don't spam.** Detect events (state changes) by comparing
  against the previous sample. First observation of a state emits
  nothing; only transitions are events.
- **Render as stable lines.** No header tag (the core labels the
  source), fixed line order, every line present only when it has
  data. Values change; the shape does not.

## Testing

Inject fakes — clock, probe functions, subprocess shims — so tests
never touch real hardware or the real network. Cover the differential
logic (each event fires exactly on transition) and the `/turn`
contract (status code, JSON `{"text": ...}` body, empty-ok on no
data). Run from the
service directory:

```bash
cd workspace/services/<name> && uv run --frozen pytest
```

## Wiring into autostart

`workspace/autostart.sh` is your boot script (you may write it; it
runs inside the sandbox) — add a `launch_service` line for your
service. The user's own script, `builtin/autostart.sh`, is
off-limits (no write access); if present it runs outside the
sandbox (the repo does not ship one). The core exports
`NAN_WORKSPACE`, `NAN_CORE_HOME`, `NAN_INJECTION_PORT` and
`NAN_CORE_PID` to launched services; logs go to
`$NAN_WORKSPACE/<name>.log`.

## Before finishing

- [ ] `uv sync` passes; `uv.lock` committed; `uv run --frozen pytest` green
- [ ] Kill it mid-run: next turn proceeds without you, no retries, no noise
- [ ] Every config key has a default; `config.yaml` documents each one
- [ ] Re-read `nan/injection/hub.py` — if your assumptions differ from the
      code, the code wins

A service may also provide only a panel: register it by heartbeating
`POST /surfaces` (see `docs/for-service-authors/injection-service.md`).
