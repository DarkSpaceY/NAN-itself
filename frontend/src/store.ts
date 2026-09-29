// 订阅式状态源：单一 store，事件 reducer，UI 层订阅渲染。
// 设计约束：方案B —— 对话轮（user_input 边界）为常驻历史，
// 记录/输出为舞台面素材；store 不持有任何渲染概念（parts 映射
// 在阶段2的 mapper 中，布局在阶段3的 layout 引擎中）。

import {
  classifyEvent,
  isEnvelope,
  type Envelope,
  type DetailEntry,
  type RecordCategory,
} from "./protocol";

// ------------------------------------------------------------------
// 状态模型
// ------------------------------------------------------------------

export type ConnectionState = "connecting" | "open" | "closed";

export interface Round {
  seq: number;
  ts: number;
  text: string;
  mid?: string;
}

export interface OutputItem {
  kind: "output";
  id: string;
  seq: number;
  ts: number;
  state: "streaming" | "done" | "cancelled";
  text: string;
  durationS?: number;
}

export type RecordState = "running" | "done" | "failed" | "void";

export interface RecordItem {
  kind: "record";
  id: string;
  seq: number;
  ts: number;
  category: RecordCategory;
  payload: Record<string, unknown>;
  state: RecordState;
  entries?: DetailEntry[];
  error?: { type: string; message: string };
  durationS?: number;
}

export type Item = OutputItem | RecordItem;

export interface NanState {
  connection: ConnectionState;
  bootId: string | null;
  /** 已应用的最大事件 seq（hello 之后用于去重重放）。 */
  lastSeq: number;
  /** 最近一次 status 事件的状态。 */
  status: "working" | "idle" | null;
  /** 对话轮：user_input 边界，append-only。 */
  rounds: Round[];
  /** 舞台素材：按事件 id 配对聚合的 output/record。 */
  items: Record<string, Item>;
}

export function initialState(): NanState {
  return {
    connection: "closed",
    bootId: null,
    lastSeq: 0,
    status: null,
    rounds: [],
    items: {},
  };
}

// ------------------------------------------------------------------
// 纯 reducer：事件 → 状态
// ------------------------------------------------------------------

/**
 * 应用一个事件信封。规则：
 * - seq 单调去重（重放与直播重叠时丢弃旧事件）
 * - user_input 开新轮
 * - 同 id 事件配对：record_started → detail/done/failed/void；
 *   output_started → delta…/done/cancelled
 * - 未知 / surface.* 事件忽略
 */
export function applyEvent(state: NanState, env: Envelope): NanState {
  if (classifyEvent(env) !== "live") return state;
  if (env.seq <= state.lastSeq) return state;

  const next: NanState = {
    ...state,
    lastSeq: env.seq,
  };

  switch (env.t) {
    case "user_input": {
      const text = typeof env.content.text === "string" ? env.content.text : "";
      const mid = typeof env.content.mid === "string" ? env.content.mid : undefined;
      next.rounds = [
        ...state.rounds,
        { seq: env.seq, ts: env.ts, text, mid },
      ];
      return next;
    }

    case "status": {
      const state_ = env.content.state;
      next.status = state_ === "working" || state_ === "idle" ? state_ : state.status;
      return next;
    }

    case "output_started": {
      next.items = {
        ...state.items,
        [env.id!]: {
          kind: "output",
          id: env.id!,
          seq: env.seq,
          ts: env.ts,
          state: "streaming",
          text: "",
        },
      };
      return next;
    }

    case "output_delta": {
      const item = state.items[env.id!];
      if (!item || item.kind !== "output") return next;
      const text = typeof env.content.text === "string" ? env.content.text : "";
      next.items = {
        ...state.items,
        [env.id!]: { ...item, text: item.text + text },
      };
      return next;
    }

    case "output_done": {
      const item = state.items[env.id!];
      if (!item || item.kind !== "output") return next;
      next.items = {
        ...state.items,
        [env.id!]: {
          ...item,
          state: "done",
          durationS: num(env.content.duration_s),
        },
      };
      return next;
    }

    case "output_cancelled": {
      const item = state.items[env.id!];
      if (!item || item.kind !== "output") return next;
      next.items = {
        ...state.items,
        [env.id!]: { ...item, state: "cancelled" },
      };
      return next;
    }

    case "record_started": {
      next.items = {
        ...state.items,
        [env.id!]: {
          kind: "record",
          id: env.id!,
          seq: env.seq,
          ts: env.ts,
          category: (env.content.category as RecordCategory) ?? "unknown",
          payload:
            typeof env.content.payload === "object" && env.content.payload !== null
              ? (env.content.payload as Record<string, unknown>)
              : {},
          state: "running",
        },
      };
      return next;
    }

    case "record_detail": {
      const item = state.items[env.id!];
      if (!item || item.kind !== "record") return next;
      const entries = Array.isArray(env.content.entries)
        ? (env.content.entries as DetailEntry[])
        : undefined;
      next.items = {
        ...state.items,
        [env.id!]: { ...item, entries },
      };
      return next;
    }

    case "record_done": {
      const item = state.items[env.id!];
      if (!item || item.kind !== "record") return next;
      next.items = {
        ...state.items,
        [env.id!]: { ...item, state: "done", durationS: num(env.content.duration_s) },
      };
      return next;
    }

    case "record_failed": {
      const item = state.items[env.id!];
      if (!item || item.kind !== "record") return next;
      const err = env.content.error;
      next.items = {
        ...state.items,
        [env.id!]: {
          ...item,
          state: "failed",
          error:
            typeof err === "object" && err !== null
              ? {
                  type: String((err as Record<string, unknown>).type ?? ""),
                  message: String((err as Record<string, unknown>).message ?? ""),
                }
              : undefined,
        },
      };
      return next;
    }

    case "record_void": {
      const item = state.items[env.id!];
      if (!item || item.kind !== "record") return next;
      next.items = {
        ...state.items,
        [env.id!]: { ...item, state: "void" },
      };
      return next;
    }
  }

  return next;
}

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

// ------------------------------------------------------------------
// Store：订阅 + 连接生命周期动作
// ------------------------------------------------------------------

type Listener = () => void;

export class Store {
  private state: NanState = initialState();
  private listeners = new Set<Listener>();

  getState(): NanState {
    return this.state;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private set(next: NanState) {
    this.state = next;
    for (const fn of this.listeners) fn();
  }

  /** 应用一个服务端消息（信封 or hello）。非信封消息忽略；
   *  reducer 无变化时（重复 seq / 忽略类型）不通知。 */
  applyMessage(msg: unknown): void {
    if (!isEnvelope(msg)) return;
    const next = applyEvent(this.state, msg);
    if (next !== this.state) this.set(next);
  }

  setConnection(conn: ConnectionState): void {
    if (this.state.connection === conn) return;
    this.set({ ...this.state, connection: conn });
  }

  /** hello 握手：记录 bootId。seq 不在此消费——重放事件的 seq 可能
   * 低于 hello.seq，lastSeq 只能随事件自身推进。 */
  applyHello(hello: { seq: number; content?: { boot?: string } }): void {
    const bootId =
      typeof hello.content?.boot === "string" ? hello.content.boot : null;
    if (bootId === this.state.bootId) return;
    this.set({ ...this.state, bootId });
  }

  /** 断线重连后本地模型失效：重放会重新灌入，清空重建。 */
  resetForReplay(): void {
    this.set({ ...initialState(), connection: this.state.connection });
  }
}
