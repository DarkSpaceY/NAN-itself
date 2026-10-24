// Events of a single agent -> timeline. One Mapper per agent.
//
// Routing (which event belongs to whom) lives in hub.ts; this file is
// only responsible for folding one agent's events into rounds and
// parts. All methods are synchronous and replayable: feeding the same
// event sequence twice yields the same result (the precondition for
// reconnect replay).

import { errorOf, obj, str, textOf, verbOf, type Envelope } from "../protocol";
import type { InputPart, Part, Round, TextPart, VerbPart } from "./parts";

type Listener = (mapper: Mapper) => void;

export class Mapper {
  readonly agentId: string;
  private rounds: Round[] = [];
  private open: Round | null = null;
  private seq = 0;
  private listeners = new Set<Listener>();

  /** output_* family pairs by event id: id -> the part it belongs to. */
  private streams = new Map<string, TextPart>();

  /** verb calls: verb name -> the most recent part not yet closed. */
  private running = new Map<string, VerbPart>();

  /** User messages taken over from overwritten attempts, prepended to the next round. */
  private carry: Part[] = [];

  /** Messages sent before the current round ends: kept out of the timeline until the next round boundary. */
  private queued: InputPart[] = [];

  constructor(agentId: string) {
    this.agentId = agentId;
  }

  /** Whether this agent has produced any renderable content. */
  get isQuiet(): boolean {
    return this.rounds.length === 0;
  }

  getRounds(): readonly Round[] {
    return this.rounds;
  }

  /** Messages emitted but not yet in the timeline because the current round is still running. */
  getQueued(): readonly InputPart[] {
    return this.queued;
  }

  getMaxSeq(): number {
    return this.seq;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private notify(): void {
    for (const fn of this.listeners) fn(this);
  }

  private stamp(round: Round): void {
    round.appliedSeq = Math.max(round.appliedSeq, this.seq);
  }

  private startRound(env: Envelope, index?: number): Round {
    const round: Round = {
      index: index ?? this.rounds.length + 1,
      ts: env.ts,
      appliedSeq: env.seq,
      closed: false,
      outcome: null,
      detail: null,
      parts: this.takeCarry(),
    };
    this.rounds.push(round);
    this.open = round;
    return round;
  }

  /** Current round; created on demand -- no content lost when events arrive before turn_started. */
  private current(env: Envelope): Round {
    if (this.open && !this.open.closed) return this.open;
    return this.startRound(env);
  }

  private closeOpen(
    env: Envelope,
    outcome?: Round["outcome"],
    detail?: string | null,
    killVerbs = false,
  ): void {
    if (!this.open) return;
    const round = this.open;

    round.closed = true;
    round.outcome = outcome ?? null;
    round.detail = detail ?? null;

    // When a round ends mid-stream, the backend will not send
    // output_done / verb_done: the caret must go out. A round-level
    // failure marks running verbs red; a pause interruption adds no
    // extra marker and leaves the lines as they are.
    for (const part of round.parts) {
      if (part.kind === "text" && part.streaming) part.streaming = false;
      if (killVerbs && part.kind === "verb" && part.state === "running")
        part.state = "failed";
    }
    this.streams.clear();
    this.running.clear();

    this.stamp(round);
    this.open = null;
    void env;
  }

  /**
   * Retry semantics are overwrite: before a new round starts, the whole
   * failed / interrupted attempt with the same turn_index is removed,
   * and its user messages are handed over to the new round. The error
   * line therefore stays visible during backoff and is overwritten as
   * soon as the retry speaks.
   */
  private rollback(index: number): void {
    const dead = this.rounds.filter(
      (round) =>
        round.index === index &&
        (round.outcome === "failed" || round.outcome === "cancelled"),
    );
    if (dead.length === 0) return;

    this.carry.push(
      ...dead.flatMap((round) =>
        round.parts.filter((part) => part.kind === "input"),
      ),
    );
    this.rounds = this.rounds.filter((round) => !dead.includes(round));
  }

  /**
   * The round boundary takes the queued messages: same as the backend --
   * the inbox drains only when the next round starts. Until then they
   * stay off the timeline, hanging above the composer.
   */
  private drainQueued(): InputPart[] {
    const taken = this.queued.splice(0);
    for (const part of taken) part.status = "consumed";
    return taken;
  }

  private takeCarry(): Part[] {
    const carried = this.carry;
    this.carry = [];
    return carried;
  }

  /** Seed existing rounds when replaying history, avoiding duplicate rendering. */
  seed(rounds: Round[]): void {
    this.rounds = [...rounds];
    this.seq = rounds.reduce((max, r) => Math.max(max, r.appliedSeq), 0);
    this.notify();
  }

  /**
   * Apply one event. The caller guarantees the event belongs to this
   * agent; seq deduplication happens at the hub layer, this only
   * guarantees idempotence (re-applying the same seq adds nothing).
   */
  feed(env: Envelope): void {
    if (env.seq > this.seq) this.seq = env.seq;

    switch (env.t) {
      case "turn_started": {
        const index =
          typeof env.content.turn_index === "number"
            ? env.content.turn_index
            : undefined;
        // Before the new round opens: overwrite the previous attempt, then take the queued messages
        if (index !== undefined) this.rollback(index);
        const drained = this.drainQueued();
        const round = this.current(env);
        if (drained.length > 0) round.parts.unshift(...drained);
        if (index !== undefined) round.index = index;
        this.stamp(round);
        break;
      }

      case "user_input": {
        // The backend echoes on receipt while the current round may still
        // be running. The message must not enter the stream: queue it
        // first, and it lands at the head of the timeline at the next
        // round boundary.
        const part: InputPart = {
          kind: "input",
          id: env.id ?? `in-${env.seq}`,
          text: textOf(env),
          mid: str(env.content.mid) ?? undefined,
          status: this.open && !this.open.closed ? "queued" : "echo",
        };

        if (part.status === "queued") {
          this.queued.push(part);
        } else {
          this.startRound(env).parts.push(part);
        }
        break;
      }

      case "output_started": {
        const round = this.current(env);
        const part: TextPart = {
          kind: "text",
          id: env.id ?? `out-${env.seq}`,
          text: "",
          streaming: true,
        };
        round.parts.push(part);
        if (env.id) this.streams.set(env.id, part);
        this.stamp(round);
        break;
      }

      case "output_delta": {
        const part = env.id ? this.streams.get(env.id) : undefined;
        const round = this.current(env);
        if (part) {
          part.text += textOf(env);
        } else {
          // Replay starts mid-stream: output_started was evicted from the
          // ring buffer. Still merge by event id, otherwise every delta
          // becomes its own segment.
          const orphan: TextPart = {
            kind: "text",
            id: env.id ?? `out-${env.seq}`,
            text: textOf(env),
            streaming: true,
          };
          round.parts.push(orphan);
          if (env.id) this.streams.set(env.id, orphan);
        }
        this.stamp(round);
        break;
      }

      case "output_done":
      case "output_cancelled": {
        // output_cancelled only means the stream was continued by a tool
        // call; the text still goes into history. Handle it the same as
        // done, without an error marker.
        const part = env.id ? this.streams.get(env.id) : undefined;
        if (part) {
          part.streaming = false;
          if (env.id) this.streams.delete(env.id);
        }
        if (this.open) this.stamp(this.open);
        break;
      }

      case "verb_started": {
        const round = this.current(env);
        const verb = verbOf(env);
        const part: VerbPart = {
          kind: "verb",
          id: env.id ?? `verb-${env.seq}`,
          verb,
          args: obj(env.content.arguments) ?? {},
          state: "running",
        };
        round.parts.push(part);
        this.running.set(verb, part);
        this.stamp(round);
        break;
      }

      case "verb_result": {
        const part = this.running.get(verbOf(env));
        if (part) {
          part.text = textOf(env);
          part.detail = obj(env.content.detail);
        }
        if (this.open) this.stamp(this.open);
        break;
      }

      case "verb_done": {
        const verb = verbOf(env);
        const part = this.running.get(verb);
        if (part) {
          part.state = "done";
          const d = env.content.duration_s;
          part.durationS = typeof d === "number" ? d : undefined;
          this.running.delete(verb);
        }
        if (this.open) this.stamp(this.open);
        break;
      }

      case "verb_failed": {
        const verb = verbOf(env);
        const part = this.running.get(verb);
        if (part) {
          part.state = "failed";
          part.error = errorOf(env) ?? { type: "error", message: "" };
          this.running.delete(verb);
        }
        if (this.open) this.stamp(this.open);
        break;
      }

      case "turn_done": {
        this.closeOpen(env, "done");
        break;
      }

      case "turn_failed":
        // The error line stays visible during backoff; the next round's turn_started overwrites it as soon as the retry begins
        this.closeOpen(env, "failed", str(env.content.error), true);
        break;

      case "turn_cancelled": {
        // pause interruption: the whole round is voided, its content does not enter history
        const requeued = env.content.requeued_count;
        this.closeOpen(
          env,
          "cancelled",
          typeof requeued === "number" ? `${requeued} input requeued` : null,
        );
        break;
      }

      case "agent_exited":
        this.closeOpen(env);
        break;

      default:
        break;
    }

    this.settleInputs();
    this.notify();
  }

  /**
   * Echoes sent while idle, not yet picked up by a round: any output in
   * the same round means they were consumed. queued interjections do
   * not come through here; the round boundary takes them.
   */
  private settleInputs(): void {
    for (const round of this.rounds) {
      round.parts.forEach((part, index) => {
        if (part.kind !== "input" || part.status !== "echo") return;
        const consumed = round.parts
          .slice(index + 1)
          .some((later) => later.kind === "text" || later.kind === "verb");
        if (consumed) part.status = "consumed";
      });
    }
  }

  /** Locate the round holding a part, for rail jumps and scroll anchoring. */
  locate(partId: string): { round: number; index: number } | null {
    for (const round of this.rounds) {
      const index = round.parts.findIndex((p) => p.id === partId);
      if (index >= 0) return { round: round.index, index };
    }
    return null;
  }
}
