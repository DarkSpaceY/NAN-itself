// Event routing: one stream -> one timeline per agent.
//
// The hub is the single entry. It dispatches events by agent_id to the
// matching Mapper and maintains the agent tree and global state. The
// root's identity is only learned at agent_started; events arriving
// before that are held in a bounded buffer and back-filled in seq
// order once the root appears.

import { agentIdOf, isLive, type Envelope } from "../protocol";
import { emptyTree, pathTo, reduceTree, type AgentNode, type AgentTree } from "../agents";
import type { GlobalStore } from "../store";
import { Mapper } from "./mapper";
import type { InputPart } from "./parts";

/** Cap for events with unassigned ownership: drop the oldest beyond it; the desktop is a live view, not a log. */
const PENDING_LIMIT = 64;

/** Keep at most this many timelines for exited agents; the one being viewed is never evicted. */
const TIMELINE_KEEP = 16;

type Listener = () => void;

export class Hub {
  readonly global: GlobalStore;
  private tree: AgentTree = emptyTree();
  private mappers = new Map<string, Mapper>();
  private pending: Envelope[] = [];
  /** Exit order of exited agents, used to evict timelines. */
  private exited: string[] = [];
  /** Who the panel is currently viewing: never evicted even after exit. */
  private pinned: string | null = null;
  private lastSeq = 0;
  private version = 0;
  private listeners = new Set<Listener>();

  constructor(global: GlobalStore) {
    this.global = global;
  }

  // -- subscription ---------------------------------------------------

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

  // -- read ------------------------------------------------------------

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

  /** Messages emitted but not yet in the timeline (agent's round still running). */
  queued(agentId: string | null): readonly InputPart[] {
    if (!agentId) return [];
    return this.mappers.get(agentId)?.getQueued() ?? [];
  }

  /** Breadcrumb path: root -> ... -> agentId. */
  path(agentId: string | null): AgentNode[] {
    if (!agentId) return [];
    return pathTo(this.tree, agentId);
  }

  node(agentId: string | null): AgentNode | undefined {
    return agentId ? this.tree.nodes.get(agentId) : undefined;
  }

  // -- write -----------------------------------------------------------

  /** The single entry. Non-live events and duplicate seq are dropped outright. */
  ingest(env: Envelope): void {
    if (!isLive(env)) return;
    if (env.seq <= this.lastSeq) return;
    this.lastSeq = env.seq;

    // Tree first: the root identity (agent_started) must reach the store
    // before anything else, otherwise the root's agent_paused is mistaken
    // for a subagent's signal and dropped.
    reduceTree(this.tree, env);
    this.global.setRoot(this.tree.rootId);
    this.global.apply(env);

    const agentId = agentIdOf(env);

    // Root unknown: hold events (keeping seq order). Once the root shows
    // up -- either as agent_started or as any event carrying depth 0
    // (when replay starts mid-way, agent_started was already evicted
    // from the ring buffer) -- back-fill in order.
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

    // Exit only reclaims the timeline; the node stays: breadcrumbs and
    // the open button always find their target.
    if (env.t === "agent_exited") {
      const who = agentIdOf(env);
      if (who) this.exited.push(who);
      this.trimTimelines();
    }

    this.bump();
  }

  /** Pin the agent the panel is viewing so its history is not evicted. */
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

  /** Back-fill the held user_input events: they carry no agent_id and can only go to the root. */
  private flush(rootId: string): void {
    if (this.pending.length === 0) return;
    const queued = this.pending;
    this.pending = [];
    for (const env of queued) this.mapperFor(rootId).feed(env);
  }

  /** hello brought a different boot: all history is voided; tree and timelines are rebuilt. */
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
