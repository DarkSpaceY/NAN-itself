import { describe, expect, it } from "vitest";

import { Mapper } from "../src/kernel/history/mapper";
import type { Envelope } from "../src/kernel/protocol";
import type { Round } from "../src/kernel/history/parts";

function env(seq: number, t: string, content: Record<string, unknown>, id?: string): Envelope {
  return { t, seq, ts: seq, content, ...(id ? { id } : {}) };
}

const base = { agent_id: "r0", depth: 0 };

function kinds(round: Round): string[] {
  return round.parts.map((p) => p.kind);
}

describe("mapper", () => {
  it("keeps the backend order: whole stream ends, then verbs run", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "turn_started", { ...base, turn_index: 1, inbox_count: 1 }));
    m.feed(env(2, "output_started", base, "o1"));
    m.feed(env(3, "output_delta", { ...base, text: "let me look" }, "o1"));
    m.feed(env(4, "output_cancelled", base, "o1"));
    m.feed(env(5, "verb_started", { ...base, verb: "shell", arguments: { command: "ls" } }));
    m.feed(env(6, "verb_result", { ...base, verb: "shell", text: "a.py" }));
    m.feed(env(7, "verb_done", { ...base, verb: "shell", duration_s: 0.4 }));
    m.feed(env(8, "turn_done", { ...base, turn_index: 1 }));

    const round = m.getRounds()[0];
    expect(kinds(round)).toEqual(["text", "verb"]);
    expect(round.parts[0]).toMatchObject({ text: "let me look", streaming: false });
    expect(round.parts[1]).toMatchObject({ verb: "shell", state: "done", durationS: 0.4 });
    expect(round.closed).toBe(true);
    expect(round.outcome).toBe("done");
  });

  it("merges deltas whose output_started fell out of the replay window", () => {
    const m = new Mapper("r0");
    // 重放从流中间开始：没有 output_started，只有同一 id 的增量
    m.feed(env(3, "output_delta", { ...base, text: "ou" }, "o1"));
    m.feed(env(4, "output_delta", { ...base, text: "t" }, "o1"));
    m.feed(env(5, "output_delta", { ...base, text: "run" }, "o1"));

    const round = m.getRounds()[0];
    expect(round.parts).toHaveLength(1);
    expect(round.parts[0]).toMatchObject({ kind: "text", text: "outrun" });
  });

  it("streams reasoning before the answer and closes it", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "turn_started", { ...base, turn_index: 1 }));
    m.feed(env(2, "thinking_started", base, "k1"));
    m.feed(env(3, "thinking_delta", { ...base, text: "think " }, "k1"));
    m.feed(env(4, "thinking_delta", { ...base, text: "first" }, "k1"));
    m.feed(env(5, "thinking_done", base, "k1"));
    m.feed(env(6, "output_started", base, "o1"));
    m.feed(env(7, "output_delta", { ...base, text: "answer" }, "o1"));
    m.feed(env(8, "output_done", base, "o1"));

    const round = m.getRounds()[0];
    expect(kinds(round)).toEqual(["thinking", "text"]);
    expect(round.parts[0]).toMatchObject({
      kind: "thinking",
      text: "think first",
      streaming: false,
    });
    expect(round.parts[1]).toMatchObject({ kind: "text", text: "answer" });
  });

  it("merges thinking deltas whose thinking_started fell out of the replay window", () => {
    const m = new Mapper("r0");
    m.feed(env(3, "thinking_delta", { ...base, text: "rea" }, "k1"));
    m.feed(env(4, "thinking_delta", { ...base, text: "son" }, "k1"));

    const round = m.getRounds()[0];
    expect(round.parts).toHaveLength(1);
    expect(round.parts[0]).toMatchObject({ kind: "thinking", text: "reason" });
  });

  it("does not mark a tool-driven stream as cancelled", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "output_started", base, "o1"));
    m.feed(env(2, "output_delta", { ...base, text: "text kept in history" }, "o1"));
    m.feed(env(3, "output_cancelled", base, "o1"));

    const part = m.getRounds()[0].parts[0];
    expect(part).toMatchObject({ kind: "text", streaming: false });
    expect(part).not.toHaveProperty("cancelled");
  });

  it("attaches turn_started to the round opened by the input echo", () => {
    const m = new Mapper("r0");
    // 后端时序：user_input 先发，agent 开轮时才有 turn_started
    m.feed(env(1, "user_input", { text: "hi" }));
    m.feed(env(2, "turn_started", { ...base, turn_index: 1, inbox_count: 1 }));
    m.feed(env(3, "output_started", base, "o1"));
    m.feed(env(4, "output_delta", { ...base, text: "hello" }, "o1"));

    expect(m.getRounds()).toHaveLength(1);
    const round = m.getRounds()[0];
    expect(round.index).toBe(1);
    expect(kinds(round)).toEqual(["input", "text"]);
    expect(round.parts[0]).toMatchObject({ kind: "input", status: "consumed" });
  });

  it("keeps a mid-turn message out of the stream until the next turn", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "user_input", { text: "先看目录" }));
    m.feed(env(2, "turn_started", { ...base, turn_index: 1 }));
    m.feed(env(3, "output_started", base, "o1"));
    m.feed(env(4, "output_delta", { ...base, text: "正在看" }, "o1"));
    // 这一轮还在跑，用户又发一条：不能进流，先排队
    m.feed(env(5, "user_input", { text: "顺便跑下测试" }));
    m.feed(env(6, "output_done", base, "o1"));
    m.feed(env(7, "turn_done", { ...base, turn_index: 1 }));

    expect(kinds(m.getRounds()[0])).toEqual(["input", "text"]);
    expect(m.getQueued().map((part) => part.text)).toEqual(["顺便跑下测试"]);

    // 下一个轮开口时才落到时间线开头
    m.feed(env(8, "turn_started", { ...base, turn_index: 2 }));
    m.feed(env(9, "output_started", base, "o2"));
    m.feed(env(10, "output_delta", { ...base, text: "跑完了" }, "o2"));

    const rounds = m.getRounds();
    expect(rounds).toHaveLength(2);
    expect(kinds(rounds[0])).toEqual(["input", "text"]);
    expect(kinds(rounds[1])).toEqual(["input", "text"]);
    expect(rounds[1].parts[0]).toMatchObject({ text: "顺便跑下测试", status: "consumed" });
    expect(m.getQueued()).toHaveLength(0);
  });

  it("pairs verb calls by name across started/result/done", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "verb_started", { ...base, verb: "spawn", arguments: { task: "x" } }));
    m.feed(env(2, "verb_result", { ...base, verb: "spawn", text: "ok", detail: { child_id: "a3" } }));
    m.feed(env(3, "verb_done", { ...base, verb: "spawn", duration_s: 0.01 }));

    const part = m.getRounds()[0].parts[0];
    expect(part).toMatchObject({ verb: "spawn", state: "done", text: "ok", detail: { child_id: "a3" } });
  });

  it("records verb failures", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "verb_started", { ...base, verb: "shell", arguments: {} }));
    m.feed(env(2, "verb_failed", { ...base, verb: "shell", error: { type: "Timeout", message: "30s" } }));

    expect(m.getRounds()[0].parts[0]).toMatchObject({
      state: "failed",
      error: { type: "Timeout", message: "30s" },
    });
  });

  it("overwrites the failed attempt as soon as the retry starts", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "turn_started", { ...base, turn_index: 3 }));
    m.feed(env(2, "turn_failed", { ...base, error: "boom", backoff_s: 1, retry_index: 0 }));

    const [failed] = m.getRounds();
    expect(failed.outcome).toBe("failed");
    expect(failed.detail).toBe("boom");
    expect(failed.closed).toBe(true);

    m.feed(env(3, "turn_started", { ...base, turn_index: 3 }));

    const rounds = m.getRounds();
    expect(rounds).toHaveLength(1);
    expect(rounds[0].outcome).toBeNull();
    expect(rounds[0].index).toBe(3);
  });

  it("closes the caret and running verbs when a turn dies mid-stream", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "turn_started", { ...base, turn_index: 1 }));
    m.feed(env(2, "output_started", base, "o1"));
    m.feed(env(3, "output_delta", { ...base, text: "半句" }, "o1"));
    m.feed(env(4, "verb_started", { ...base, verb: "shell", arguments: {} }));

    m.feed(env(5, "turn_failed", { ...base, error: "boom" }, undefined));

    const [round] = m.getRounds();
    expect(round.parts[0]).toMatchObject({ kind: "text", streaming: false });
    expect(round.parts[1]).toMatchObject({ kind: "verb", state: "failed" });
  });

  it("rolls the failed attempt away when the retry starts", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "user_input", { text: "跑测试" }));
    m.feed(env(2, "turn_started", { ...base, turn_index: 3 }));
    m.feed(env(3, "output_started", base, "o1"));
    m.feed(env(4, "output_delta", { ...base, text: "半句" }, "o1"));
    m.feed(env(5, "turn_failed", { ...base, error: "boom" }));

    m.feed(env(6, "turn_started", { ...base, turn_index: 3 }));
    m.feed(env(7, "output_started", base, "o2"));
    m.feed(env(8, "output_delta", { ...base, text: "完整结论" }, "o2"));
    m.feed(env(9, "output_done", base, "o2"));
    m.feed(env(10, "turn_done", { ...base, turn_index: 3 }));

    const rounds = m.getRounds();
    expect(rounds).toHaveLength(1);
    expect(rounds[0].outcome).toBe("done");
    expect(rounds[0].parts.map((part) => part.kind)).toEqual(["input", "text"]);
    expect(rounds[0].parts[1]).toMatchObject({ text: "完整结论" });
  });

  it("marks a paused turn as cancelled", () => {
    const m = new Mapper("r0");
    m.feed(env(1, "turn_started", { ...base, turn_index: 1 }));
    m.feed(env(2, "turn_cancelled", { ...base, requeued_count: 2 }));

    expect(m.getRounds()[0].outcome).toBe("cancelled");
    expect(m.getRounds()[0].detail).toBe("2 input requeued");
  });

  it("is idempotent under replay", () => {
    const events = [
      env(1, "turn_started", { ...base, turn_index: 1 }),
      env(2, "output_started", base, "o1"),
      env(3, "output_delta", { ...base, text: "a" }, "o1"),
      env(4, "output_delta", { ...base, text: "b" }, "o1"),
      env(5, "output_done", base, "o1"),
    ];

    const once = new Mapper("r0");
    for (const e of events) once.feed(e);
    const twice = new Mapper("r0");
    for (const e of events) twice.feed(e);

    expect(twice.getRounds()[0].parts).toEqual(once.getRounds()[0].parts);
  });
});