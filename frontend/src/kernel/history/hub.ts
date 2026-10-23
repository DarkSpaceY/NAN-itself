// 事件路由：一条流 → 每个 agent 一条时间线。
//
// hub 是唯一入口。它按 agent_id 把事件分给对应的 Mapper，同时维护
// agent 树与全局状态。root 的身份要等 agent_started 才知道，之前
// 到达的事件先压在一个有界缓冲里，root 一出现就按 seq 顺序补灌。

import { agentIdOf, isLive, type Envelope } from "../protocol";
import { emptyTree, pathTo, reduceTree, type AgentNode, type AgentTree } from "../agents";
import type { GlobalStore } from "../store";
import { Mapper } from "./mapper";
import type { InputPart } from "./parts";

/** 未定归属的事件缓冲上限：超过就丢最旧的，桌面是实时视图不是日志。 */
const PENDING_LIMIT = 64;

/** 已退出 agent 的时间线最多留这么多份，正在看的那个不淘汰。 */
const TIMELINE_KEEP = 16;

type Listener = () => void;

export class Hub {
  readonly global: GlobalStore;
  private tree: AgentTree = emptyTree();
  private mappers = new Map<string, Mapper>();
  private pending: Envelope[] = [];
  /** 已退出 agent 的退出顺序，用于淘汰时间线。 */
  private exited: string[] = [];
  /** 当前面板正在看谁：它即使已退出也不淘汰。 */
  private pinned: string | null = null;
  private lastSeq = 0;
  private version = 0;
  private listeners = new Set<Listener>();

  constructor(global: GlobalStore) {
    this.global = global;
  }

  // -- 订阅 ----------------------------------------------------------

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  getVersion(): number {
    return this.version;
  }

  private bump(): void {
    this.version += 1;
    for (const fn of this.listeners) fn();
  }

  private mapperFor(agentId: string): Mapper {
    const existing = this.mappers.get(agentId);
    if (existing) return existing;
    const mapper = new Mapper(agentId);
    mapper.subscribe(() => this.bump());
    this.mappers.set(agentId, mapper);
    return mapper;
  }

  // -- 读取 ----------------------------------------------------------

  get rootId(): string | null {
    return this.tree.rootId;
  }

  get agents(): AgentTree {
    return this.tree;
  }

  mapper(agentId: string): Mapper | undefined {
    return this.mappers.get(agentId);
  }

  rounds(agentId: string | null) {
    if (!agentId) return [];
    return this.mapperFor(agentId).getRounds();
  }

  /** 发出但还没进时间线的消息（agent 本轮仍在跑）。 */
  queued(agentId: string | null): readonly InputPart[] {
    if (!agentId) return [];
    return this.mappers.get(agentId)?.getQueued() ?? [];
  }

  /** 面包屑路径：root → ... → agentId。 */
  path(agentId: string | null): AgentNode[] {
    if (!agentId) return [];
    return pathTo(this.tree, agentId);
  }

  node(agentId: string | null): AgentNode | undefined {
    return agentId ? this.tree.nodes.get(agentId) : undefined;
  }

  // -- 写入 ----------------------------------------------------------

  /** 唯一入口。非 live 事件、重复 seq 直接丢弃。 */
  ingest(env: Envelope): void {
    if (!isLive(env)) return;
    if (env.seq <= this.lastSeq) return;
    this.lastSeq = env.seq;

    // 树先走：root 身份（agent_started）必须先被 store 知道，
    // 否则 root 的 agent_paused 会被当成子 agent 的信号丢掉。
    reduceTree(this.tree, env);
    this.global.setRoot(this.tree.rootId);
    this.global.apply(env);

    const agentId = agentIdOf(env);

    // root 未知：事件先压住（保持 seq 顺序）。root 一现身 —— 可能是
    // agent_started，也可能是任何带 depth 0 的事件（重放从半截开始
    // 时 agent_started 已被环形缓冲挤掉）—— 再按序补灌。
    if (this.tree.rootId === null) {
      if (agentId || env.t === "user_input") {
        this.pending.push(env);
        if (this.pending.length > PENDING_LIMIT) this.pending.shift();
      }
      this.bump();
      return;
    }

    const rootId = this.tree.rootId;
    this.flush(rootId);

    if (env.t === "user_input") {
      this.mapperFor(rootId).feed(env);
    } else if (agentId) {
      this.mapperFor(agentId).feed(env);
    }

    // 退出只回收时间线，节点留着：面包屑和 open 按钮永远指得到人。
    if (env.t === "agent_exited") {
      const who = agentIdOf(env);
      if (who) this.exited.push(who);
      this.trimTimelines();
    }

    this.bump();
  }

  /** 把面板正在看的 agent 钉住，避免它的历史被淘汰掉。 */
  pin(agentId: string | null): void {
    this.pinned = agentId;
  }

  private trimTimelines(): void {
    while (this.exited.length > TIMELINE_KEEP) {
      const oldest = this.exited.find((id) => id !== this.pinned && id !== this.tree.rootId);
      if (!oldest) return;
      this.exited = this.exited.filter((id) => id !== oldest);
      this.mappers.delete(oldest);
    }
  }

  /** 补灌攒下的 user_input：它们没有 agent_id，只能进 root。 */
  private flush(rootId: string): void {
    if (this.pending.length === 0) return;
    const queued = this.pending;
    this.pending = [];
    for (const env of queued) this.mapperFor(rootId).feed(env);
  }

  /** hello 换了 boot：历史全部作废，树与时间线重建。 */
  reset(): void {
    this.tree = emptyTree();
    this.mappers.clear();
    this.pending = [];
    this.exited = [];
    this.pinned = null;
    this.lastSeq = 0;
    this.global.setRoot(null);
    this.bump();
  }
}
