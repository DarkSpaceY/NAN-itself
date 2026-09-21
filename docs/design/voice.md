# Design: Full-Duplex Voice Module

Status: **implemented** (2026-09-21, all decisions locked with the user).
Voice is the first bidirectional (reactive) module: it perceives speech,
decides, and acts — speaking — through the channel downlink. This doc
fixes the v1 contract.

## Decisions locked

- **Semantic / surface split.** The main agent emits a semantic *task*;
  a small local SLM (~0.6B) composes the actual utterance (wording,
  rhythm, tone directives). No mainstream framework documents this
  layer (closest prior art: Moshi's Inner Monologue, inverted); the
  task protocol below is ours to define.
- **TTS: CosyVoice2-0.5B** (evaluate Fun-CosyVoice3-0.5B as a drop-in —
  control surface is compatible). Cross-platform: no MLX (core
  principle 2).
- **STT: faster-whisper** (already in the stack).
- **Mic ownership: audio module publishes, voice consumes.** The audio
  module keeps the microphone, publishes a rolling PCM ring as facts;
  voice is a downstream consumer (`requires = ["audio"]`). STT is
  removed from the audio module.
- **v1 scope (all module-internal, zero framework changes):**
  task → SLM → sentence-streamed TTS playback, barge-in, and the SLM
  fast path (autonomous replies to trivial turns). *Not in v1:*
  backchannel fillers, preemptive generation, AEC (v1 = headphone mode
  or the gating below), voice-clone UI.

## Architecture

```
audio module (owns mic)
  capture loop ── publishes ──▶ PCM ring snapshot (facts, DataSpace)
                               + ambient features (existing behavior)

voice module (requires = ["audio"])            [down: channel slot]
  dialogue loop                                ┌──────────────────┐
    PCM ring ─▶ VAD gate ─▶ endpointing ─▶ STT │ say (depth N)    │
                  │                (0.5s)  ─▶  └──────────────────┘
                  │                                ▲ task JSON
                  barge-in (≥0.5s speech          │ from main agent
                  while speaking)                 │
    speak loop ◀── utterances ─── SLM ◀───────────┘
    (CosyVoice stream)      {instruct, text}
         │
         ▶ speaker
```

State machine: `idle → listening → thinking (main agent turn or SLM
fast path) → speaking → (barge-in → idle)`. The full state is part of
the `query()` projection.

## The uplink (facts)

- Transcript events publish as facts: `{speaker: "user", text, at,
  final: bool}`. The agent reads them through `query()` projection
  (`[Voice] user said: ...`), exactly like every other ambient module.
- The module never interprets; the agent decides what a user utterance
  means. Exception: the SLM fast path (below), which reports what it
  answered.

## The downlink: `say` channel

ActionSurface with one channel. Composite name: `module:voice/say`.

```python
class TaskPayload(BaseModel):
    intent: str          # "answer" | "ask" | "confirm" | "narrate" | ...
    key_points: str      # semantic content, prose
    tone: str = ""       # optional tone hint, e.g. "warm", "concise"
    interruptible: bool = True
```

- `depth=N` FIFO: a long reply may be written as several tasks;
  barge-in drains the queue.
- The **SLM output** (consumed by TTS, never by the agent) is:

```python
class Utterance(BaseModel):
    instruct: str        # CosyVoice instruct directive, whitelisted
    text: str            # utterance text with <strong>/<laughter> tokens
```

- The split maps onto CosyVoice's control surface: the instruct text
  (`用开心的语气说<|endofprompt|>`) and inline fine-grained tokens
  (`[laughter]`, `[breath]`, `<strong>`) are the emotion levers, so the
  SLM prompt embeds a **whitelist** of legal directives/tokens and its
  output is validated against it — hallucinated directives are dropped
  before synthesis. Voice identity (reference audio), sample rate and
  speed fallbacks are module config, not per-task fields.

## SLM fast path (the "0.6B speaks for the agent" boundary)

- The SLM answers **trivial turns autonomously** (greetings, timers,
  "what time is it", small confirmations) without a main-agent turn.
  Rules:
  1. a fixed, versioned allowlist/prompt describes what it may answer
     (no opinions, no actions, no memory-dependent claims);
  2. anything non-trivial escalates: the module publishes the utterance
     as a pending user turn for the main agent;
  3. everything the SLM said is projected in `query()` as
     `[Voice] answered myself: ...` so the agent stays consistent;
  4. a task written to `say` always preempts fast-path chatter.
- The SLM runs locally (transformers, **Qwen3-0.6B** in non-thinking
  mode for latency), loaded in `start()` like every other backend.

## Speak loop and barge-in

- TTS synthesis is **sentence-streamed** (`stream=True` chunks) — first
  audio lands while later sentences still synthesize.
- Playback keeps the VAD armed on the incoming audio ring. Continuous
  speech ≥ 0.5 s (LiveKit adaptive threshold) while speaking =
  barge-in: stop playback, drain the `say` queue, `emit_event("user
  interrupted")`, return to listening. The event flows to the agent in
  the next projection.
- Echo (v1): headphones assumed; speaker playback is possible but
  self-hearing must be expected. AEC (far-end reference = our own TTS
  stream) is v2 — it is the reason listen and say live in one module.

## Crash semantics (core principle 7)

All provisioning (CosyVoice, whisper, SLM, speaker) happens in
`start()` before any loop; failures raise — the Facade marks the module
DOWN with the error and retries with backoff. Missing TTS weights is a
loud failure that revives when weights land, never a silent limbo.

## Config and layout

- `builtin/modules/voice.py` (module), `backend/nan_itself/utils/tts.py`
  (CosyVoice adapter, lazy-loaded like `utils/vision.py`),
  `backend/nan_itself/utils/dialogue.py` (SLM adapter) *open* — may
  merge if thin.
- Weights under `models/voice/` (`tts/cosyvoice2-0.5b`, `slm/...`,
  whisper reuses the audio module's copy); env overrides `NAN_VOICE_*`.
- New dependency: `cosyvoice` runtime (pynini via conda-forge has macOS
  arm64 wheels; `ttsfrd` is Linux-only — macOS/Windows use the
  WeTextProcessing fallback). Packaging (decided): **upstream checkout**
  — FunAudioLLM/CosyVoice cloned to `models/voice/cosyvoice/`
  (path overridable via `NAN_VOICE_*`), injected into `sys.path` by the
  `utils/tts.py` adapter; the repo itself stays free of third-party
  code.
- Persistence: `serialize_state()` keeps counters and the last
  transcript ring only; channel residue is not persisted.

## Open items

Resolved 2026-09-21 (locked):

1. SLM model id: **Qwen3-0.6B** (non-thinking mode); prompt template is
   written at implementation time with the directive whitelist.
2. CosyVoice packaging: **upstream checkout** (see Config and layout);
   Windows story = WeTextProcessing fallback.
3. Task queue semantics: **drain-on-barge-in only** — no
   `queue_position`/cancel API in v1; the FIFO + drain covers long
   replies.

Still open (non-blocking):

4. Fun-CosyVoice3-0.5B evaluation — quality vs v2, same control API;
   done after v1 works, if weights are available.
