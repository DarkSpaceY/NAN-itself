import { describe, expect, it } from "vitest";

import { emptyTree, pathTo, reduceTree, type AgentTree } from "../src/kernel/agents";
import type { Envelope } from "../src/kernel/protocol";

function env(seq: number, t: string, content: Record<string, unknown>): Envelope {
  return { t, seq, ts: seq, content };
}

describe("agents", () => {
  it("builds the tree from agent_started", () => {
    const tree = emptyTree();
    reduceTree(tree, env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    reduceTree(tree, env(2, "agent_started", { agent_id: "a3", depth: 1, parent_id: "r0" }));

    expect(tree.rootId).toBe("r0");
    expect(tree.nodes.get("a3")?.parent_id).toBe("r0");
    expect(tree.nodes.get("a3")?.depth).toBe(1);
    expect(pathTo(tree, "a3").map((n) => n.agent_id)).toEqual(["r0", "a3"]);
  });

  it("fills a placeholder when agent_started arrives late", () => {
    const tree: AgentTree = emptyTree();
    reduceTree(tree, env(1, "turn_started", { agent_id: "r0", turn_index: 1 }));
    reduceTree(tree, env(2, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));

    expect(tree.rootId).toBe("r0");
    expect(tree.nodes.get("r0")?.turns).toBe(1);
  });

  it("tracks status and turns", () => {
    const tree = emptyTree();
    reduceTree(tree, env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    reduceTree(tree, env(2, "turn_started", { agent_id: "r0", turn_index: 1 }));
    expect(tree.nodes.get("r0")?.status).toBe("working");

    reduceTree(tree, env(3, "turn_done", { agent_id: "r0", turn_index: 1 }));
    expect(tree.nodes.get("r0")?.status).toBe("idle");

    reduceTree(tree, env(4, "agent_exited", { agent_id: "r0", reason: "finished" }));
    expect(tree.nodes.get("r0")?.status).toBe("exited");
    expect(tree.nodes.get("r0")?.exitReason).toBe("finished");
  });

  it("keeps exited nodes around so navigation stays possible", () => {
    const tree = emptyTree();
    reduceTree(tree, env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));

    for (let i = 0; i < 12; i += 1) {
      const id = `c${i}`;
      reduceTree(tree, env(10 + i, "agent_started", { agent_id: id, depth: 1, parent_id: "r0" }));
      reduceTree(tree, env(20 + i, "agent_exited", { agent_id: id, reason: "finished" }));
    }

    // 节点本身很轻，留着才能让 open / 面包屑指得到人
    expect(tree.nodes.has("c0")).toBe(true);
    expect(tree.nodes.has("c11")).toBe(true);
    expect(tree.nodes.get("c11")?.status).toBe("exited");
    expect(tree.nodes.get("r0")?.status).not.toBe("exited");
  });

  it("rejects paths that do not reach the root", () => {
    const tree = emptyTree();
    reduceTree(tree, env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    reduceTree(tree, env(2, "agent_started", { agent_id: "orphan", depth: 1, parent_id: "ghost" }));

    expect(pathTo(tree, "orphan")).toEqual([]);
    expect(pathTo(tree, "r0")).toHaveLength(1);
  });
});
