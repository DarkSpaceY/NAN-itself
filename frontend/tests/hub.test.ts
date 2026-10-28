import { describe, expect, it } from "vitest";

import { Hub } from "../src/kernel/history/hub";
import { GlobalStore } from "../src/kernel/store";
import type { Envelope } from "../src/kernel/protocol";

function env(seq: number, t: string, content: Record<string, unknown>, id?: string): Envelope {
  return { t, seq, ts: seq, content, ...(id ? { id } : {}) };
}

function makeHub() {
  const global = new GlobalStore();
  return { hub: new Hub(global), global };
}

describe("hub", () => {
  it("buffers events that arrive before root is known, then flushes in order", () => {
    const { hub } = makeHub();

    hub.ingest(env(1, "user_input", { text: "hi" }));
    hub.ingest(env(2, "output_started", { agent_id: "r0" }, "o1"));
    hub.ingest(env(3, "output_delta", { agent_id: "r0", text: "hello" }, "o1"));
    hub.ingest(env(4, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));

    const rounds = hub.rounds("r0");
    expect(rounds).toHaveLength(1);
    expect(rounds[0].parts.map((p) => p.kind)).toEqual(["input", "text"]);
    expect(rounds[0].parts[1]).toMatchObject({ text: "hello" });
  });

  it("learns root from depth when the replay starts mid-run", () => {
    const { hub } = makeHub();

    // 环形缓冲挤掉了 agent_started：重放从半截开始
    hub.ingest(env(1, "turn_started", { agent_id: "r0", turn_index: 4 }));
    hub.ingest(env(2, "output_started", { agent_id: "r0", depth: 0 }, "o1"));
    hub.ingest(env(3, "output_delta", { agent_id: "r0", depth: 0, text: "mid" }, "o1"));

    expect(hub.rootId).toBe("r0");
    expect(hub.rounds("r0")[0].parts[0]).toMatchObject({ kind: "text", text: "mid" });
  });

  it("routes events to the agent's own mapper", () => {
    const { hub } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    hub.ingest(env(2, "agent_started", { agent_id: "a3", depth: 1, parent_id: "r0" }));
    hub.ingest(env(3, "output_started", { agent_id: "a3" }, "o2"));
    hub.ingest(env(4, "output_delta", { agent_id: "a3", text: "child work" }, "o2"));

    expect(hub.rounds("a3")[0].parts[0]).toMatchObject({ kind: "text", text: "child work" });
    expect(hub.rounds("r0")).toHaveLength(0);
    expect(hub.path("a3").map((n) => n.agent_id)).toEqual(["r0", "a3"]);
  });

  it("evicts old timelines but keeps the nodes reachable", () => {
    const { hub } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));

    let seq = 2;
    for (let i = 0; i < 17; i += 1) {
      const id = `c${i}`;
      hub.ingest(env(seq++, "agent_started", { agent_id: id, depth: 1, parent_id: "r0" }));
      hub.ingest(env(seq++, "agent_exited", { agent_id: id, reason: "finished" }));
    }

    // 时间线只留最近 16 份，但节点还在，open 不会变成死按钮
    expect(hub.mapper("c0")).toBeUndefined();
    expect(hub.mapper("c16")).toBeDefined();
    expect(hub.node("c0")?.status).toBe("exited");
  });

  it("never evicts the timeline the panel is looking at", () => {
    const { hub } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));

    let seq = 2;
    for (let i = 0; i < 20; i += 1) {
      const id = `c${i}`;
      hub.ingest(env(seq++, "agent_started", { agent_id: id, depth: 1, parent_id: "r0" }));
      if (i === 0) hub.pin(id);
      hub.ingest(env(seq++, "agent_exited", { agent_id: id, reason: "finished" }));
    }

    expect(hub.mapper("c0")).toBeDefined();
    expect(hub.mapper("c1")).toBeUndefined();
  });

  it("walks the breadcrumb path through nested subagents", () => {
    const { hub } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    hub.ingest(env(2, "agent_started", { agent_id: "a3", depth: 1, parent_id: "r0" }));
    hub.ingest(env(3, "agent_started", { agent_id: "a7", depth: 2, parent_id: "a3" }));

    expect(hub.path("a7").map((n) => n.agent_id)).toEqual(["r0", "a3", "a7"]);
  });

  it("has no breadcrumb for an agent whose root is unreachable", () => {
    const { hub } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    // 只到 turn_started，没有 agent_started：占位节点 parent 为空
    hub.ingest(env(2, "turn_started", { agent_id: "ghost", depth: 2, turn_index: 1 }));

    expect(hub.path("ghost")).toEqual([]);
  });

  it("drops duplicate and non-live events", () => {
    const { hub, global } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    hub.ingest(env(2, "unknown_thing", { agent_id: "r0" }));

    expect(global.getState().lastSeq).toBe(1);
    expect(hub.agents.nodes.size).toBe(1);
  });

  it("applies pause only for root", () => {
    const { hub, global } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    hub.ingest(env(2, "agent_paused", { agent_id: "a3", depth: 1 }));
    expect(global.getState().paused).toBeNull();

    hub.ingest(env(3, "agent_paused", { agent_id: "r0", depth: 0 }));
    expect(global.getState().paused).toBe(true);

    hub.ingest(env(4, "agent_resumed", { agent_id: "r0", depth: 0 }));
    expect(global.getState().paused).toBe(false);
  });

  it("resets tree and timelines on a new boot", () => {
    const { hub } = makeHub();
    hub.ingest(env(1, "agent_started", { agent_id: "r0", depth: 0, parent_id: null }));
    hub.reset();

    expect(hub.rootId).toBeNull();
    expect(hub.rounds("r0")).toHaveLength(0);
  });
});
