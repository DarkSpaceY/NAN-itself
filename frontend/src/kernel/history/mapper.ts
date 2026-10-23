// 单个 agent 的事件 → 时间线。一个 agent 一个 Mapper。
//
// 路由（哪个事件属于谁）在 hub.ts；这里只负责把一个 agent 的事件
// 折叠成轮与 part。所有方法都是同步的、可重放的：同一串事件灌
// 两次得到同一结果（重连重放的前提）。

import { errorOf, obj, str, textOf, verbOf, type Envelope } from "../protocol";
import type { InputPart, Part, Round, TextPart, VerbPart } from "./parts";

type Listener = (mapper: Mapper) => void;

export class Mapper {
  readonly agentId: string;
  private rounds: Round[] = [];
  private open: Round | null = null;
  private seq = 0;
  private listeners = new Set<Listener>();

  /** output_* 家族按事件 id 配对：id → 所在 part。 */
  private streams = new Map<string, TextPart>();

  /** verb 调用：verb 名 → 最近一个未收尾的 part。 */
  private running = new Map<string, VerbPart>();

  /** 从被覆盖的尝试里接过来的用户消息，挂到下一轮开头。 */
  private carry: Part[] = [];

  /** 本轮还没结束就发出的消息：不进时间线，等下一个轮边界。 */
  private queued: InputPart[] = [];

  constructor(agentId: string) {
    this.agentId = agentId;
  }

  /** 该 agent 是否已产出可渲染内容。 */
  get isQuiet(): boolean {
    return this.rounds.length === 0;
  }

  getRounds(): readonly Round[] {
    return this.rounds;
  }

  /** 已发出、但本轮还没结束所以进不了时间线的消息。 */
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

  /** 当前轮；没有就按需新建 —— 事件先于 turn_started 到达时不丢内容。 */
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

    // 轮在流中途结束时，后端不会再发 output_done / verb_done：
    // 光标必须熄掉。轮级失败把跑着的 verb 标红；pause 打断则不
    // 加任何额外标记，行保持原状。
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
   * 重试的语义是覆盖：新的一轮开始前，把同一 turn_index 的失败 /
   * 被打断尝试整轮拿掉，再把其中的用户消息转交给新轮。
   * 所以在退避等待期间错误行还看得见，重试一开口就被覆盖掉。
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
   * 轮边界取走排队的消息：与后端一致 —— inbox 在下一轮开始时才 drain。
   * 在这之前它们不出现在时间线上，只挂在 composer 上方。
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

  /** 重放历史时把已存在的轮塞进来，避免重复渲染。 */
  seed(rounds: Round[]): void {
    this.rounds = [...rounds];
    this.seq = rounds.reduce((max, r) => Math.max(max, r.appliedSeq), 0);
    this.notify();
  }

  /**
   * 应用一个事件。调用方保证事件属于本 agent；seq 去重在 hub 层做，
   * 这里只保证幂等（同 seq 重复应用不产生新内容）。
   */
  feed(env: Envelope): void {
    if (env.seq > this.seq) this.seq = env.seq;

    switch (env.t) {
      case "turn_started": {
        const index =
          typeof env.content.turn_index === "number"
            ? env.content.turn_index
            : undefined;
        // 新的轮开口之前：先覆盖掉上一次尝试，再取走排队等待的消息
        if (index !== undefined) this.rollback(index);
        const drained = this.drainQueued();
        const round = this.current(env);
        if (drained.length > 0) round.parts.unshift(...drained);
        if (index !== undefined) round.index = index;
        this.stamp(round);
        break;
      }

      case "user_input": {
        // 后端收到即发回执，当前轮可能还在跑。此时消息不能进流：先排队，
        // 等下一个轮边界才落到时间线开头。
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
          // 重放从流中间开始：output_started 被环形缓冲挤掉了。
          // 仍然按事件 id 归并，否则每个增量都会变成独立的一段。
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
        // output_cancelled 只是说明这段流被工具调用接续，文本本身
        // 照样进 history，所以这里与 done 同处理，不标记异常。
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
        // 退避期间错误行可见；重试一开始就被下一轮的 turn_started 覆盖掉
        this.closeOpen(env, "failed", str(env.content.error), true);
        break;

      case "turn_cancelled": {
        // pause 打断：这一轮整体作废，内容不进 history
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
   * 空闲时发出、还没有轮来接的回执：同一轮里出现输出就说明已被消费。
   * queued 的插话不走这里，它由轮边界统一取走。
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

  /** 定位某 part 所在轮，供 rail 跳转与滚动锚定。 */
  locate(partId: string): { round: number; index: number } | null {
    for (const round of this.rounds) {
      const index = round.parts.findIndex((p) => p.id === partId);
      if (index >= 0) return { round: round.index, index };
    }
    return null;
  }
}
