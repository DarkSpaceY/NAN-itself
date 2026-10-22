# UI redesign: Conversation + Ledger

Design for the frontend rewrite. Direction chosen from four
wireframes: **A (conversation + inspector) as the shell, C
(event ledger) as its second view**. Reference wireframes:
`/tmp/nan-ui-companion` (scratch, not part of the repo).

## Goals

- stay readable as a conversation for the root agent
- expose every event as a filterable row for monitoring and
  debugging (the ledger view)
- make the agent tree navigable without leaving the main view
- keep working when the agent runs indefinitely

Non-goals: artifact canvas / diff review, kanban or agent-card
mission control, editing history, multi-user sessions.

## Shell

```
┌──────────────────────────────────────────────────────────────┐
│ status strip   model · boot · paused · injection health       │
├───────────┬──────────────────────────────────┬───────────────┤
│ left rail │ center: selected agent            │ inspector     │
│ sessions  │ [ Conversation | Ledger ]         │ Turn/Agent/   │
│ agent tree│                                   │ Services      │
├───────────┴──────────────────────────────────┴───────────────┤
│ composer                                        pause button │
└──────────────────────────────────────────────────────────────┘
```

The left rail lists past sessions and the live agent tree
(indented by depth, status dot, child count). The center always
shows exactly one agent. The inspector describes the selected
turn or agent; it never duplicates the center.

## Views of the center

### Conversation

The selected agent's turns as a chat thread.

- turn separator: `#13 · 12:41:07 · 3.1s · 812 / 388 tok`,
  clickable (selects the turn in the inspector)
- inbox messages of the turn: user input on the right, parent
  messages and subagent finish/death notices on the left with a
  source label
- assistant output: markdown, streamed; a caret while the
  stream is open
- verbs: collapsed inline cards (`shell · rg -n ... · 0.4s`);
  click expands arguments and the result, click again collapses
- spawn: a card with the child id and task; clicking it selects
  the child (the thread switches, a breadcrumb returns)
- failures: `turn_failed` / `turn_cancelled` render as inline
  error blocks with backoff and retry index
- when idle: one muted line `idle · 2 queued` -- never a fake
  "thinking" state

### Ledger

Every event of the selected agent (or all agents) as rows,
grouped under turn headers, ordered by `seq`.

- row: time gutter, actor, event kind, one-line summary,
  right-aligned duration/cost; expand for the full payload
- filters: all / verbs / injections / errors / lifecycle
- "follow tail" toggle pins to the newest event
- rows for other agents are indented by depth

The ledger is a direct projection of the event stream: no
derived semantics beyond grouping by turn and agent.

## What each event becomes

| event | conversation | ledger row |
|---|---|---|
| `user_input` | user bubble | input |
| `agent_started` | muted header line | lifecycle |
| `agent_paused`/`resumed` | status strip + system line | lifecycle |
| `agent_exited` | terminal line with reason | lifecycle |
| `turn_started` | turn separator | turn header |
| `injections_collected` | collapsed chip on the separator (`2 services · 1 timeout`) | injection group |
| `output_*` | streamed markdown | output row |
| `verb_*` | inline tool card | action row |
| `turn_done` | separator gets duration/tokens | turn footer |
| `turn_failed` | error block | error row |
| `turn_cancelled` | system line | lifecycle |

## Attribution rule

`output_*` and `verb_*` carry no turn index. The frontend
attributes them to the agent's **open turn** (the interval
between that agent's `turn_started` and its `turn_done` /
`turn_failed` / `turn_cancelled`). Each agent has at most one
open turn, so the attribution is unambiguous. This is a
frontend rule; the protocol stays flat.

## State

One normalized store fed by the event stream, plus UI
selection.

```
agents:   Map<agent_id, {depth, parent_id, pacing, status, children}>
turns:    Map<agent_id, Map<turn_index, TurnView>>
streams:  Map<stream_id, {agent_id, text, open}>
verbs:    per agent+turn, ordered by seq
services: per agent+turn (last injections_collected)
ui:       {selectedAgent, selectedTurn, view, filters, followTail, pausedPending}
```

`TurnView` accumulates: inbox count, usage, latency, retries,
calls, history_chars, status (open/done/failed/cancelled),
output text, verb cards, injection summary.

## Controls

- **input**: optimistic bubble keyed by `mid`; reconciled when
  `user_input` with that `mid` arrives. Send is never disabled
  (the agent may be mid-turn by design).
- **pause**: three states -- running / pending / paused.
  Clicking sends `{"t":"pause"}`; `agent_paused` confirms,
  `agent_resumed` clears. Clicking while pending withdraws.
- **selection**: clicking an agent in the tree or a spawn card
  switches the center; the inspector follows the selection.
- **reconnect**: browser `seq` gap renders a "missed N events"
  marker; a changed `boot` in `hello` starts a fresh session.

## Persistence

Append the raw envelope log to IndexedDB per boot session and
replay it on load to rebuild the store. This replaces the old
rounds mapper/database (one source instead of two). A session
cap (number of events or bytes) trims the oldest.

## Tech

Keep React 19, Vite, Tailwind 4, zustand, react-markdown,
katex, shadcn-style primitives.

Drop `@assistant-ui/*`: its runtime is message-centric, while
this UI is turn-centric with two synchronized views; keeping it
would force the turn model through a message model.

Delete from the current `src/`: `history/mapper`, `persistence/
db`, `convert.ts`, `ui/RecordTool.tsx`, `ui/pause.tsx`,
`ui/App.tsx`, and rewrite `protocol.ts`, `store.ts`, `ws.ts`.

## Testing

- reducer tests: fixture event logs -> expected store
  (streaming merge, verb attribution, pause confirmation,
  reconnect gap, session replay)
- component tests: conversation renders a turn; ledger renders
  filters and expansion
- no browser automation required for the first pass

## Open questions

1. The protocol does not carry the **rendered observation**
   (delta/injection/inbox text as the model saw it). Should the
   inspector show raw event data only, or do we add a field to
   `turn_done` (e.g. `observation_chars`) or a separate
   `turn_observation` event?
2. Persist the raw event log (as above), or keep memory-only
   since the backend already has a bounded history ring?
3. When a subagent is active, should the center auto-follow it,
   or stay on root until the user clicks? (Recommendation:
   stay on root, but show a "a1 is speaking" badge on the tree.)
