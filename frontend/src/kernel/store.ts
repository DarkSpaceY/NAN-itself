// The single source of connection and global state.
//
// Only three concerns: socket connection state, boot identity, and the
// global state (pause) derived from the event stream. Conversation
// content is not here -- see history/hub.ts.
//
// Pure reducer + subscription, no React, no DOM, directly unit-testable.

import { isLive, type Envelope, str } from "./protocol";

export type Connection = "connecting" | "open" | "closed";

export interface GlobalState {
  connection: Connection;
  /** Backend boot; a change means seq is no longer comparable and history must be rebuilt. */
  bootId: string | null;
  /** Highest applied seq, for replay deduplication. */
  lastSeq: number;
  /** The root's agent_id. Backend ids are random strings; learn it only from the agent tree. */
  rootId: string | null;
  /** The root's paused state; null means no signal received yet. */
  paused: boolean | null;
  /** Model name, from hello's content.model; null when unknown. */
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
 * Apply one event. Returning the same reference means no change
 * (duplicate seq, unknown type); subscribers are not woken.
 */

/** Whether the event speaks for the root: agent_id first, depth 0 as fallback. */
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
      // Pausing is cooperative: only the root stopping counts as a
      // machine-wide pause. The root's agent_id is a random string;
      // identity cannot be matched by string.
      if (isRoot(state, env)) next.paused = true;
      return next;

    case "agent_resumed":
      if (isRoot(state, env)) next.paused = false;
      return next;

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

  /** Back-filled by the hub once it learns the root identity from the agent tree; no-op if unchanged. */
  setRoot(rootId: string | null): void {
    if (this.state.rootId === rootId) return;
    this.commit({ ...this.state, rootId });
  }

  /**
   * hello handshake. A changed boot means the backend restarted: seq
   * resets to zero and history is voided, while the connection state
   * and the latest pause signal persist until new events overwrite them.
   */
  applyHello(hello: {
    seq: number;
    content?: { boot?: string; model?: string; paused?: boolean };
  }): void {
    const bootId = hello.content?.boot ?? null;
    const bootChanged = bootId !== this.state.bootId;
    const paused = hello.content?.paused ?? this.state.paused;

    // Do not take hello.seq as lastSeq: replayed events may carry lower
    // seqs; the cursor can only follow the events themselves.
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
