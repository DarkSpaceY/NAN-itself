// 连接与全局状态的唯一来源。
//
// 只管三件事：套接字连接态、boot 身份、以及从事件流推出的全局
// 状态（暂停）。对话内容不在这里 —— 见 history/hub.ts。
//
// 纯 reducer + 订阅，无 React、无 DOM，可直接单测。

import { isLive, type Envelope, str } from "./protocol";

export type Connection = "connecting" | "open" | "closed";

export interface GlobalState {
  connection: Connection;
  /** 后端 boot；变化意味着 seq 不再可比，历史必须重建。 */
  bootId: string | null;
  /** 已应用的最大 seq，重放去重用。 */
  lastSeq: number;
  /** root 的 agent_id。后端的 id 是随机串，只能从 agent 树里学。 */
  rootId: string | null;
  /** root 的暂停态；null 表示尚未收到任何信号。 */
  paused: boolean | null;
  /** 模型名，来自 hello 的 content.model；未知为 null。 */
  model: string | null;
}

export function initialGlobal(): GlobalState {
  return {
    connection: "closed",
    bootId: null,
    lastSeq: 0,
    rootId: null,
    paused: null,
    model: null,
  };
}

/**
 * 应用一个事件。返回同一引用表示无变化（重复 seq、未知类型），
 * 订阅者不会被唤醒。
 */

/** 事件说的是不是 root：优先认 agent_id，其次认 depth 0。 */
function isRoot(state: GlobalState, env: Envelope): boolean {
  const agentId = str(env.content.agent_id);
  if (state.rootId !== null) return agentId === state.rootId;
  return env.content.depth === 0;
}

export function reduceGlobal(state: GlobalState, env: Envelope): GlobalState {
  if (!isLive(env)) return state;
  if (env.seq <= state.lastSeq) return state;

  const next: GlobalState = { ...state, lastSeq: env.seq };

  switch (env.t) {
    case "agent_paused":
      // 暂停是协作式的：只有 root 停下来才算整机暂停。
      // root 的 agent_id 是随机串，认身份不能靠字符串。
      if (isRoot(state, env)) next.paused = true;
      return next;

    case "agent_resumed":
      if (isRoot(state, env)) next.paused = false;
      return next;

    case "status": {
      const s = str(env.content.state);
      if (s === "paused") next.paused = true;
      else if (s === "working" || s === "idle") next.paused = false;
      return next;
    }

    default:
      return next;
  }
}

type Listener = () => void;

export class GlobalStore {
  private state: GlobalState = initialGlobal();
  private listeners = new Set<Listener>();

  getState(): GlobalState {
    return this.state;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private commit(next: GlobalState): void {
    this.state = next;
    for (const fn of this.listeners) fn();
  }

  apply(env: Envelope): void {
    const next = reduceGlobal(this.state, env);
    if (next !== this.state) this.commit(next);
  }

  setConnection(connection: Connection): void {
    if (this.state.connection === connection) return;
    this.commit({ ...this.state, connection });
  }

  /** hub 从 agent 树里学到 root 身份后回填；未变则不动。 */
  setRoot(rootId: string | null): void {
    if (this.state.rootId === rootId) return;
    this.commit({ ...this.state, rootId });
  }

  /**
   * hello 握手。boot 变了说明后端重启过：seq 归零、历史作废，
   * 但连接态与最近一次暂停信号保留到新事件覆盖。
   */
  applyHello(hello: {
    seq: number;
    content?: { boot?: string; model?: string; status?: { state?: string } };
  }): void {
    const bootId = hello.content?.boot ?? null;
    const bootChanged = bootId !== this.state.bootId;
    const state = hello.content?.status?.state;
    const paused =
      state === "paused"
        ? true
        : state === "working" || state === "idle"
          ? false
          : this.state.paused;

    // lastSeq 不取 hello.seq：重放事件的 seq 可能低于它，游标只能
    // 跟着事件自身走。
    this.commit({
      connection: this.state.connection,
      bootId,
      lastSeq: bootChanged ? 0 : this.state.lastSeq,
      rootId: bootChanged ? null : this.state.rootId,
      paused,
      model: hello.content?.model ?? this.state.model,
    });
  }
}
