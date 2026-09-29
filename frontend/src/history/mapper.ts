// 事件流 → 历史对话轮 parts 的映射。
//
// 设计约束：
// - 对话轮以 user_input 为边界，append-only；轮内 parts 顺序即事件序
// - 历史流是纯文字 + 内联工具卡片（kitn.ai/ui 视觉模式），不含面
// - seq 单调守卫挡住同 boot 内重放/直播重叠；跨 boot（页面刷新）时
//   IndexedDB 已有同 mid 的旧轮，用「同 key 重放覆盖替换」去重
// - surface.* / 未知事件不产生 parts

import { classifyEvent, type Envelope } from "../protocol";

// ------------------------------------------------------------------
// parts 模型
// ------------------------------------------------------------------

export type ToolState = "running" | "done" | "failed" | "void";

export interface TextPart {
  kind: "text";
  id: string; // 事件 id（assistant 流）或 round 派生 id（user）
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
  cancelled?: boolean;
}

export interface ToolPart {
  kind: "tool";
  id: string; // record 事件 id
  category: string;
  state: ToolState;
  title: string;
  entries?: Array<Record<string, unknown>>;
  error?: { type: string; message: string };
  durationS?: number;
}

export type Part = TextPart | ToolPart;

export interface HistoryRound {
  /** user_input 的 mid；缺省时用 seq 派生。跨 boot 去重键。 */
  key: string;
  ts: number;
  seq: number;
  closed: boolean;
  parts: Part[];
}

// ------------------------------------------------------------------
// 卡片标题：按 payload 形状提取人类可读摘要，未知则兜底
// ------------------------------------------------------------------

export function toolTitle(category: string, payload: Record<string, unknown>): string {
  const candidates = [
    payload.name,
    payload.tool,
    payload.task,
    payload.agent_id,
    payload.module,
    payload.channel,
  ];
  for (const c of candidates) {
    if (typeof c === "string" && c.trim()) return c;
  }
  const keys = Object.keys(payload);
  if (keys.length > 0) return `${category}(${keys.slice(0, 3).join(", ")})`;
  return category;
}

// ------------------------------------------------------------------
// Mapper
// ------------------------------------------------------------------

type Listener = (round: HistoryRound) => void;

export class Mapper {
  private rounds: HistoryRound[] = [];
  private byKey = new Map<string, HistoryRound>();
  private lastSeq = 0;
  private listeners = new Set<Listener>();

  /**
   * 单调递增的变化版本号。`rounds` 数组是原地 push 的，引用恒定，
   * 不能当 React 快照用；版本号才是稳定的 getSnapshot 返回值。
   */
  private version = 0;

  /** 事件 id → 定位（part 所在轮 + 下标）。 */
  private partIndex = new Map<string, { round: HistoryRound; index: number }>();

  // -- 订阅 ----------------------------------------------------------

  /** 每次有轮内容变化（新建/更新/关闭）都会回调该轮对象。 */
  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(round: HistoryRound) {
    this.version++;
    for (const fn of this.listeners) fn(round);
  }

  // -- 状态 ----------------------------------------------------------

  getRounds(): readonly HistoryRound[] {
    return this.rounds;
  }

  /** 供 useSyncExternalStore 做快照：内容每次变化都会自增。 */
  getVersion(): number {
    return this.version;
  }

  getCurrent(): HistoryRound | null {
    return this.rounds.at(-1) ?? null;
  }

  // -- 种子（IndexedDB 恢复） ------------------------------------------

  /**
   * 用持久化的历史轮做种子。只登记 key 与已配对事件 id；不动
   * lastSeq——后端重启后 seq 归零，抬高会挡掉整个重放。同 boot
   * 刷新场景由「同 key 重放覆盖」去重。
   */
  seed(rounds: HistoryRound[]): void {
    for (const r of rounds) {
      this.rounds.push(r);
      this.byKey.set(r.key, r);
      for (const p of r.parts) {
        this.partIndex.set(p.id, { round: r, index: r.parts.indexOf(p) });
      }
    }
    // seed 发生在首帧之后（boot 是异步的），必须主动通知一次，
    // 否则恢复出来的历史不会渲染。
    const last = rounds.at(-1);
    if (last) this.emit(last);
  }

  // -- 事件入口 --------------------------------------------------------

  feed(env: Envelope): void {
    if (classifyEvent(env) !== "live") return;
    if (env.seq <= this.lastSeq) return;
    this.lastSeq = env.seq;

    switch (env.t) {
      case "user_input":
        this.onUserInput(env);
        return;
      case "output_started":
        this.onOutputStarted(env);
        return;
      case "output_delta":
        this.updateTextPart(env, (p) => {
          p.text += typeof env.content.text === "string" ? env.content.text : "";
        });
        return;
      case "output_done":
        this.updateTextPart(env, (p) => {
          p.streaming = false;
        });
        return;
      case "output_cancelled":
        this.updateTextPart(env, (p) => {
          p.streaming = false;
          p.cancelled = true;
        });
        return;
      case "record_started":
        this.onRecordStarted(env);
        return;
      case "record_detail":
        this.updateToolPart(env, (p) => {
          p.entries = Array.isArray(env.content.entries)
            ? (env.content.entries as Array<Record<string, unknown>>)
            : undefined;
        });
        return;
      case "record_done":
        this.updateToolPart(env, (p) => {
          p.state = "done";
          p.durationS =
            typeof env.content.duration_s === "number" ? env.content.duration_s : undefined;
        });
        return;
      case "record_failed": {
        const err = env.content.error;
        this.updateToolPart(env, (p) => {
          p.state = "failed";
          p.error =
            typeof err === "object" && err !== null
              ? {
                  type: String((err as Record<string, unknown>).type ?? ""),
                  message: String((err as Record<string, unknown>).message ?? ""),
                }
              : undefined;
        });
        return;
      }
      case "record_void":
        this.updateToolPart(env, (p) => {
          p.state = "void";
        });
        return;
    }
  }

  // -- 各事件处理 -------------------------------------------------------

  private onUserInput(env: Envelope): void {
    // 上一轮就此关闭
    const prev = this.getCurrent();
    if (prev && !prev.closed) {
      prev.closed = true;
      this.emit(prev);
    }

    const mid = typeof env.content.mid === "string" ? env.content.mid : null;
    const key = mid ?? `seq-${env.seq}`;
    const text = typeof env.content.text === "string" ? env.content.text : "";

    const existing = this.byKey.get(key);
    if (existing) {
      // 跨 boot 重放覆盖：同 mid 的轮以重放为准重建
      const round: HistoryRound = {
        key,
        ts: env.ts,
        seq: env.seq,
        closed: false,
        parts: [{ kind: "text", id: `${key}:user`, role: "user", text }],
      };
      const idx = this.rounds.indexOf(existing);
      if (idx >= 0) this.rounds[idx] = round;
      this.byKey.set(key, round);
      this.partIndex.clear();
      this.emit(round);
      return;
    }

    const round: HistoryRound = {
      key,
      ts: env.ts,
      seq: env.seq,
      closed: false,
      parts: [{ kind: "text", id: `${key}:user`, role: "user", text }],
    };
    this.rounds.push(round);
    this.byKey.set(key, round);
    this.partIndex.set(round.parts[0]!.id, { round, index: 0 });
    this.emit(round);
  }

  private onOutputStarted(env: Envelope): void {
    const round = this.getCurrent();
    if (!round) return;
    const part: TextPart = {
      kind: "text",
      id: env.id!,
      role: "assistant",
      text: "",
      streaming: true,
    };
    round.parts.push(part);
    this.partIndex.set(env.id!, { round, index: round.parts.length - 1 });
    this.emit(round);
  }

  private onRecordStarted(env: Envelope): void {
    const round = this.getCurrent();
    if (!round) return;
    const category = String(env.content.category ?? "unknown");
    const payload =
      typeof env.content.payload === "object" && env.content.payload !== null
        ? (env.content.payload as Record<string, unknown>)
        : {};
    const part: ToolPart = {
      kind: "tool",
      id: env.id!,
      category,
      state: "running",
      title: toolTitle(category, payload),
    };
    round.parts.push(part);
    this.partIndex.set(env.id!, { round, index: round.parts.length - 1 });
    this.emit(round);
  }

  private updateTextPart(env: Envelope, mutate: (p: TextPart) => void): void {
    const loc = this.partIndex.get(env.id!);
    if (!loc) return;
    const part = loc.round.parts[loc.index];
    if (!part || part.kind !== "text") return;
    mutate(part);
    this.emit(loc.round);
  }

  private updateToolPart(env: Envelope, mutate: (p: ToolPart) => void): void {
    const loc = this.partIndex.get(env.id!);
    if (!loc) return;
    const part = loc.round.parts[loc.index];
    if (!part || part.kind !== "tool") return;
    mutate(part);
    this.emit(loc.round);
  }
}
