// Agent tree: agent_id -> identity and status.
//
// The desktop assembles this tree from the event stream for breadcrumb
// navigation and status dots. The backend's identity fields are
// agent_id / parent_id / depth (see agent_started in core.py). Events
// may arrive out of order, so a missing parent gets a placeholder until
// agent_started fills it in.

import { str, type Envelope } from "./protocol";

export type AgentStatus = "working" | "idle" | "exited" | "paused";

export interface AgentNode {
  agent_id: string;
  parent_id: string | null;
  depth: number;
  /** Read from agent_started's pacing; defaults to event-driven. */
  pacing: "interval" | "event-driven";
  status: AgentStatus;
  /** Reason of the most recent agent_exited. */
  exitReason: string | null;
  /** seq at exit, used to evict the oldest exited nodes. */
  exitedSeq: number | null;
  /** The agent's turn count (turn_started counter). */
  turns: number;
  startedSeq: number;
}

/** Keep at most this many exited subagent nodes. Nodes are tiny; timelines are the bulk. */
const EXITED_KEEP = 512;

export interface AgentTree {
  nodes: Map<string, AgentNode>;
  /** The root's agent_id; determined by the first agent_started at depth 0. */
  rootId: string | null;
}

export function emptyTree(): AgentTree {
  return { nodes: new Map(), rootId: null };
}

function ensure(tree: AgentTree, agentId: string, seq: number): AgentNode {
  const existing = tree.nodes.get(agentId);
  if (existing) return existing;
  const node: AgentNode = {
    agent_id: agentId,
    parent_id: null,
    depth: 0,
    pacing: "event-driven",
    status: "idle",
    exitReason: null,
    exitedSeq: null,
    turns: 0,
    startedSeq: seq,
  };
  tree.nodes.set(agentId, node);
  return node;
}

/**
 * Apply one event, updating the tree in place. Returns whether anything
 * changed. The desktop treats only agent_started as the birth of a
 * node; other events create a placeholder node for unknown agents so
 * no event is lost.
 */
export function reduceTree(tree: AgentTree, env: Envelope): boolean {
  const agentId = str(env.content.agent_id);
  if (!agentId) return false;

  // agent_started may already have been evicted from the ring buffer
  // (replay starts mid-way). Any event carrying depth is a fallback
  // identity source: depth 0 can only be the root.
  if (tree.rootId === null && env.content.depth === 0) {
    const node = ensure(tree, agentId, env.seq);
    node.depth = 0;
    tree.rootId = agentId;
  }

  switch (env.t) {
    case "agent_started": {
      const node = ensure(tree, agentId, env.seq);
      node.parent_id = str(env.content.parent_id);
      const depth = env.content.depth;
      node.depth = typeof depth === "number" ? depth : 0;
      node.pacing =
        str(env.content.pacing) === "interval" ? "interval" : "event-driven";
      node.status = "idle";
      if (node.depth === 0 || tree.rootId === null) tree.rootId = agentId;
      return true;
    }

    case "agent_paused": {
      ensure(tree, agentId, env.seq).status = "paused";
      return true;
    }

    case "agent_resumed": {
      ensure(tree, agentId, env.seq).status = "idle";
      return true;
    }

    case "agent_exited": {
      const node = ensure(tree, agentId, env.seq);
      node.status = "exited";
      node.exitReason = str(env.content.reason);
      node.exitedSeq = env.seq;
      pruneExited(tree);
      return true;
    }

    case "turn_started": {
      const node = ensure(tree, agentId, env.seq);
      node.turns += 1;
      node.status = "working";
      return true;
    }

    case "turn_done":
    case "turn_failed": {
      const node = ensure(tree, agentId, env.seq);
      if (node.status !== "paused" && node.status !== "exited")
        node.status = "idle";
      return true;
    }

    default:
      return false;
  }
}

/** Keep only the most recent exited nodes; the root is never evicted. */
function pruneExited(tree: AgentTree): void {
  const exited = [...tree.nodes.values()]
    .filter((node) => node.status === "exited" && node.agent_id !== tree.rootId)
    .sort((a, b) => (a.exitedSeq ?? 0) - (b.exitedSeq ?? 0));

  for (const node of exited.slice(
    0,
    Math.max(0, exited.length - EXITED_KEEP),
  )) {
    tree.nodes.delete(node.agent_id);
  }
}

/** Path from root to the agent (both ends included); returns an empty array for unknown agents. */
export function pathTo(tree: AgentTree, agentId: string): AgentNode[] {
  const chain: AgentNode[] = [];
  let cursor: string | null = agentId;
  const guard = new Set<string>();

  while (cursor && !guard.has(cursor)) {
    guard.add(cursor);
    const node = tree.nodes.get(cursor);
    if (!node) break;
    chain.unshift(node);
    cursor = node.parent_id;
  }

  // Only chains traceable back to the root count as valid paths
  if (chain.length === 0) return [];
  if (tree.rootId !== null && chain[0]?.agent_id !== tree.rootId) return [];
  return chain;
}
