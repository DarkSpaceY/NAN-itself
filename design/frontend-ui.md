# Frontend design: single surface + record window

Status: design, not yet implemented.

Decisions locked:

| topic | decision |
|---|---|
| main surface | one conversation surface; no dual-page |
| turns record | owned by a **separate record service**, rendered in its own window |
| theme | system light/dark |
| accent | one teal accent, interaction/active only |
| icons | inline SVG only; no emoji, no text glyphs |
| density | conversation comfortable, record compact |
| surfaces | flat, hairline borders, no shadows, no nested cards |

Consequence: the earlier right-hand inspector is dropped. Turn
metrics, injections and call detail belong to the record
window; the main surface keeps only what the conversation
needs.

## Shell

```
┌──────────────────────────────────────────────────────────────┐
│ status strip   model · boot · paused · service health        │
├──────────┬───────────────────────────────────────────────────┤
│ left rail│ conversation (selected agent)                     │
│ sessions │                                                   │
│ agent    │                                                   │
│ tree     │                                                   │
│          ├───────────────────────────────────────────────────┤
│          │ composer                              pause        │
└──────────┴───────────────────────────────────────────────────┘
```

- left rail: past sessions, then the live agent tree; SVG
  connectors, status by shape+color, depth by indentation
- status strip: model, boot id, pause state, service health as
  a dot matrix (one square per service)
- center: the conversation of exactly one agent
- composer: never disabled; the agent may be mid-turn by design

## Main surface: conversation

- turn separator: index, wall time, duration bar, token bar;
  clickable to focus, but it opens nothing here
- inbox messages of a turn: user on the right; parent messages
  and subagent reports on the left with a source label
- assistant output: markdown, streamed, caret while open
- verbs: one-line cards (`shell · rg -n ... · 0.4s`), click to
  expand arguments and result
- spawn: a card with child id and task; clicking switches the
  center to the child; a breadcrumb returns
- failures: inline blocks for `turn_failed` / `turn_cancelled`
- idle: one muted line `idle · 2 queued`; never a fake
  "thinking" state
- a record button in the status strip opens the record window

Nothing on this surface duplicates record data: no token
tables, no injection lists, no call trees.

## Record window (separate service)

The record service is an independent local process. It
subscribes to the gateway event stream, stores turns, and
serves its own page; the main UI only links to it.

Responsibilities:

- own the authoritative turn history across reconnects
- render the trace: turn list with role-tagged rows (system /
  user / context / assistant / tool), duration and call
  timelines, injection results, usage, retries
- search and filter within a session
- expose a deep link `?agent=<id>&turn=<n>` so the main UI can
  open a specific turn

The main UI does not embed record and does not persist events
itself; its store is memory-only and rebuilds from `hello` +
live events, marking `seq` gaps.

Transport is still open: the record service can either
subscribe to the gateway's `/ws` (same protocol, read-only) or
the core can push events to it. Recommendation: subscribe to
`/ws`, since it needs no new core-side surface and reuses one
contract.

## Style tokens

```
accent        teal, one hue; only for interactive/active
status        color + shape: filled=running, ring=idle, cross=exited
surface       flat; 1px hairline #border; no shadow; radius 4 (input 6)
type          UI sans; mono for telemetry, ids, code
              body 13/1.55; long-form 15/1.7; tabular-nums everywhere
density       conversation: 8-12px gaps; record: 4-6px rows
motion        caret blink + one pulse on the active agent; nothing else
icons         inline SVG, 14/16px, 1.5 stroke, currentColor
```

Required SVG set: pause/run, send, chevron, check, cross, clock,
streaming indicator, branch, terminal, injection-in.

## Visualization over text

- agent tree: real SVG connector lines, not box-drawing
  characters
- turn duration and usage: inline micro-bars, not bare numbers
- service health: dot matrix, not a text list
- turn timeline in record: a horizontal stacked bar (input /
  model / tools) per turn, as in the reference products
- status: shape-encoded, so color is never the only signal

## Event attribution

`output_*` and `verb_*` carry no turn index. Both the main
surface and the record service attribute them to the agent's
**open turn** (between that agent's `turn_started` and
`turn_done` / `turn_failed` / `turn_cancelled`). One open turn
per agent makes this unambiguous.

## Controls and reconnect

- input: optimistic bubble keyed by `mid`, reconciled on the
  echoed `user_input`
- pause: running / pending / paused; send `{"t":"pause"}` and
  let `agent_paused` confirm; a second click while pending
  withdraws
- reconnect: `seq` gap renders a "missed N events" marker; a
  changed `boot` starts a fresh session
- selection: tree or spawn card switches the center agent;
  selection is local, never sent to the core

## Tech

Keep React 19, Vite, Tailwind 4, zustand, react-markdown,
katex. Drop `@assistant-ui/*`: it is message-centric, this UI
is turn-centric.

Delete: `history/mapper`, `persistence/db`, `convert.ts`,
`ui/RecordTool.tsx`, `ui/pause.tsx`, `ui/App.tsx`; rewrite
`protocol.ts`, `store.ts`, `ws.ts`.

## Testing

- reducer: fixture event logs -> expected store (stream
  merging, verb attribution, pause confirmation, gap marker)
- components: conversation renders a turn; tree renders depth
  and status shapes; composer reconciles a mid
- no browser automation for the first pass

## Open questions

1. Record transport: subscribe to `/ws` (recommended) or push
   from the core?
2. Does the record window need write access (e.g. annotate a
   turn), or is it read-only?
3. Should the main surface keep any on-demand detail popover
   (one verb's full output), or is expanding the inline card
   enough?
