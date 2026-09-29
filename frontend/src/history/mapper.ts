// 事件流 → 历史对话轮 parts 的映射。
//
// 设计约束：
// - 对话轮以 user_input 为边界，append-only；轮内 parts 顺序即事件序
// - 事件归属锚定「进行中的轮」（open），不是数组末位——重放历史轮时
//   末位可能是更晚的轮
// - 历史流是纯文字 + 内联工具卡片（kitn.ai/ui 视觉模式），不含面
// - 与后端重放对账：每轮持久化 appliedSeq 游标，hello 时按 boot 取出
//   游标，重放里 seq 更小的事件全部跳过（同 boot 内 seq 才可比）
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
  /**
   * 该轮最后应用过的事件 seq（持久化游标）。刷新后据此跳过重放里
   * 已经见过的部分，避免重复追加文本。
   */
  appliedSeq: number;
  /** 产生该轮的后端 boot（seq 只在同一 boot 内单调）。 */
  bootId: string | null;
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

  /** 当前后端 boot（seq 只在同一 boot 内可比）。 */
  private bootId: string | null = null;

  /**
   * 当前「进行中」的轮。事件归属以它为锚，而不是数组末位——重放
   * 历史轮时末位可能是更晚的轮，按末位挂会把输出写错轮。
   */
  private open: HistoryRound | null = null;

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
    // feed() 已把 lastSeq 推到当前事件，故这里就是该轮的游标
    round.appliedSeq = Math.max(round.appliedSeq, this.lastSeq);
    this.version++;
    for (const fn of this.listeners) fn(round);
  }

  private indexRound(round: HistoryRound): void {
    round.parts.forEach((p, i) => this.partIndex.set(p.id, { round, index: i }));
  }

  private unindexRound(round: HistoryRound): void {
    for (const p of round.parts) this.partIndex.delete(p.id);
  }

  // -- 状态 ----------------------------------------------------------

  getRounds(): readonly HistoryRound[] {
    return this.rounds;
  }

  /** 供 useSyncExternalStore 做快照：内容每次变化都会自增。 */
  getVersion(): number {
    return this.version;
  }

  /** 当前进行中的轮（无则 null）。 */
  getCurrent(): HistoryRound | null {
    return this.open;
  }

  // -- 握手 ------------------------------------------------------------

  /**
   * hello：确定重放游标与进行中的轮。
   *
   * - boot 变了说明后端重启、seq 从头计数，必须把游标归零，否则
   *   新事件会被 seq 守卫全部丢弃（界面就此冻住）
   * - boot 未变则用本地持久化的 appliedSeq 作游标，重放里已见过的
   *   事件全部跳过，不重复追加
   * - 半途刷新时最后一轮尚未关闭，恢复为进行中，后续 delta 才有归属
   */
  onHello(bootId: string | null): void {
    if (this.bootId !== bootId) {
      this.lastSeq = 0;
    }
    this.bootId = bootId;

    let cursor = 0;
    for (const r of this.rounds) {
      if (r.bootId === bootId) cursor = Math.max(cursor, r.appliedSeq);
    }
    this.lastSeq = Math.max(this.lastSeq, cursor);

    const last = this.rounds.at(-1);
    this.open = last && !last.closed && last.bootId === bootId ? last : null;
  }

  // -- 种子（IndexedDB 恢复） ------------------------------------------

  /**
   * 用持久化的历史轮做种子。只登记 key / 事件 id / 游标，不动
   * lastSeq——游标要等 hello 拿到 boot 之后才能确定（seq 只在同一
   * boot 内可比）。
   */
  seed(rounds: HistoryRound[]): void {
    for (const r of rounds) {
      // 旧版记录没有这两个字段：游标当 0、boot 当未知，
      // 代价是这些轮会被完整重放一次
      r.appliedSeq ??= 0;
      r.bootId ??= null;
      this.rounds.push(r);
      this.byKey.set(r.key, r);
      this.indexRound(r);
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
    const prev = this.open;
    if (prev && !prev.closed) {
      prev.closed = true;
      this.emit(prev);
    }

    const mid = typeof env.content.mid === "string" ? env.content.mid : null;
    const key = mid ?? `seq-${env.seq}`;
    const text = typeof env.content.text === "string" ? env.content.text : "";

    const round: HistoryRound = {
      key,
      ts: env.ts,
      seq: env.seq,
      appliedSeq: env.seq,
      bootId: env.boot_id ?? null,
      closed: false,
      parts: [{ kind: "text", id: `${key}:user`, role: "user", text }],
    };

    const existing = this.byKey.get(key);
    if (existing) {
      // 同 mid 重放：以重放为准重建该轮（旧部件的事件索引一并摘掉）
      const idx = this.rounds.indexOf(existing);
      if (idx >= 0) this.rounds[idx] = round;
      this.unindexRound(existing);
    } else {
      this.rounds.push(round);
    }

    this.byKey.set(key, round);
    this.indexRound(round);
    this.open = round;
    this.emit(round);
  }

  private onOutputStarted(env: Envelope): void {
    const id = env.id!;

    // 重见同一 id：说明对应的 part 已存在（本地已存过一段），
    // 重建它而不是再挂一个，否则同一段文本会出现两次
    const seen = this.partIndex.get(id);
    if (seen) {
      const part = seen.round.parts[seen.index];
      if (part?.kind === "text") {
        part.text = "";
        part.streaming = true;
        part.cancelled = false;
        this.emit(seen.round);
      }
      return;
    }

    const round = this.open;
    if (!round) return;

    const part: TextPart = {
      kind: "text",
      id,
      role: "assistant",
      text: "",
      streaming: true,
    };
    round.parts.push(part);
    this.partIndex.set(id, { round, index: round.parts.length - 1 });
    this.emit(round);
  }

  private onRecordStarted(env: Envelope): void {
    const round = this.open;
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
