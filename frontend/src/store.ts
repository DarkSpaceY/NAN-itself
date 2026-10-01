// 订阅式状态源：单一 store，事件 reducer，UI 层订阅渲染。
// 职责：连接状态、status、对话轮（user_input 边界）与 seq 去重。
// output/record 事件不在这里聚合——历史渲染所需的 parts 映射在
// mapper 中；本 store 只推进 lastSeq 参与去重。

import {
  classifyEvent,
  isEnvelope,
  type Envelope,
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

export interface NanState {
  connection: ConnectionState;
  bootId: string | null;
  /** 已应用的最大事件 seq（hello 之后用于去重重放）。 */
  lastSeq: number;
  /** 最近一次 status 事件的状态。 */
  status: "working" | "idle" | "paused" | null;
  /** 对话轮：user_input 边界，append-only。 */
  rounds: Round[];
}

export function initialState(): NanState {
  return {
    connection: "closed",
    bootId: null,
    lastSeq: 0,
    status: null,
    rounds: [],
  };
}

// ------------------------------------------------------------------
// 纯 reducer：事件 → 状态
// ------------------------------------------------------------------

/**
 * 应用一个事件信封。规则：
 * - seq 单调去重（重放与直播重叠时丢弃旧事件）；任何 live 事件都
 *   推进 lastSeq（含 output/record，虽然它们不产生状态）
 * - user_input 开新轮
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
      next.status =
        state_ === "working" || state_ === "idle" || state_ === "paused"
          ? state_
          : state.status;
      return next;
    }
  }

  return next;
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
