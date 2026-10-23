// Agent 树：agent_id → 身份与状态。
//
// 桌面从事件流里拼出这棵树，用于面包屑导航和状态点。后端的身份
// 字段是 agent_id / parent_id / depth（见 core.py 的 agent_started）。
// 事件可能乱序到达，所以父节点缺失时先占位，等 agent_started 补齐。

import { str, type Envelope } from "./protocol";

export type AgentStatus = "working" | "idle" | "exited" | "paused";

export interface AgentNode {
  agent_id: string;
  parent_id: string | null;
  depth: number;
  /** 从 agent_started 的 pacing 读出；缺省按事件驱动。 */
  pacing: "interval" | "event-driven";
  status: AgentStatus;
  /** 最近一次 agent_exited 的原因。 */
  exitReason: string | null;
  /** 退出时的 seq，用于淘汰最老的已退出节点。 */
  exitedSeq: number | null;
  /** 该 agent 的轮次数（turn_started 计数）。 */
  turns: number;
  startedSeq: number;
}

/** 已退出的 subagent 节点最多留这么多个。节点很小，时间线才是大头。 */
const EXITED_KEEP = 512;

export interface AgentTree {
  nodes: Map<string, AgentNode>;
  /** root 的 agent_id；第一次 agent_started depth 0 时确定。 */
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
 * 应用一个事件，原地更新树。返回是否有变化。
 * 桌面只把 agent_started 当作节点诞生点；其他事件遇到未知 agent
 * 会补一个占位节点，避免事件丢失。
 */
export function reduceTree(tree: AgentTree, env: Envelope): boolean {
  const agentId = str(env.content.agent_id);
  if (!agentId) return false;

  // agent_started 可能已经被环形缓冲挤掉（重放从半截开始）。
  // 带 depth 的事件就是备用身份源：depth 0 只可能是 root。
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

/** 只保留最近的若干个已退出节点，root 永不淘汰。 */
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

/** 从 root 到该 agent 的路径（含两端）；未知 agent 返回空数组。 */
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

  // 只把能追到 root 的链当作有效路径
  if (chain.length === 0) return [];
  if (tree.rootId !== null && chain[0]?.agent_id !== tree.rootId) return [];
  return chain;
}
