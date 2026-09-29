import { describe, it, expect } from "vitest";
import { Mapper, toolTitle, type HistoryRound } from "../src/history/mapper";
import type { Envelope } from "../src/protocol";

function env(
  t: Envelope["t"],
  content: Record<string, unknown>,
  seq: number,
  extra: Partial<Envelope> = {},
): Envelope {
  return { seq, ts: 1000 + seq, t, content, ...extra };
}

function roundOf(m: Mapper): HistoryRound | null {
  return m.getCurrent();
}

describe("toolTitle", () => {
  it("prefers payload fields, falls back to keys/category", () => {
    expect(toolTitle("tool_call", { tool: "fs.read", provider: "x" })).toBe("fs.read");
    expect(toolTitle("subagent_spawn", { task: "do it" })).toBe("do it");
    expect(toolTitle("module_query", {})).toBe("module_query");
    expect(toolTitle("module_query", { a: 1, b: 2, c: 3, d: 4 })).toBe("module_query(a, b, c)");
  });
});

describe("Mapper", () => {
  it("user_input opens a round with user text part; next input closes it", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    let r = roundOf(m)!;
    expect(r.key).toBe("m1");
    expect(r.closed).toBe(false);
    expect(r.parts).toEqual([{ kind: "text", id: "m1:user", role: "user", text: "hi" }]);

    m.feed(env("user_input", { text: "again", mid: "m2" }, 2));
    expect(m.getRounds().length).toBe(2);
    expect(m.getRounds()[0]!.closed).toBe(true);
  });

  it("assistant stream: output parts accumulate then finish", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m.feed(env("output_started", {}, 2, { id: "e1" }));
    m.feed(env("output_delta", { text: "he" }, 3, { id: "e1" }));
    m.feed(env("output_delta", { text: "y" }, 4, { id: "e1" }));

    let r = roundOf(m)!;
    expect(r.parts[1]).toMatchObject({ kind: "text", role: "assistant", text: "hey", streaming: true });

    m.feed(env("output_done", { duration_s: 2 }, 5, { id: "e1" }));
    r = roundOf(m)!;
    expect(r.parts[1]).toMatchObject({ text: "hey", streaming: false });
  });

  it("cancelled output marks the part", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m.feed(env("output_started", {}, 2, { id: "e1" }));
    m.feed(env("output_cancelled", {}, 3, { id: "e1" }));
    expect(roundOf(m)!.parts[1]).toMatchObject({ cancelled: true, streaming: false });
  });

  it("record lifecycle becomes a tool card with detail and states", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m.feed(
      env("record_started", { category: "tool_call", payload: { tool: "fs.read" } }, 2, { id: "r1" }),
    );
    m.feed(env("record_detail", { entries: [{ kind: "field", label: "status", value: "ok" }] }, 3, { id: "r1" }));
    m.feed(env("record_done", { duration_s: 0.5 }, 4, { id: "r1" }));

    const part = roundOf(m)!.parts[1]!;
    expect(part).toMatchObject({
      kind: "tool",
      title: "fs.read",
      state: "done",
      durationS: 0.5,
      entries: [{ kind: "field", label: "status", value: "ok" }],
    });
  });

  it("record failed carries error; void marks void", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m.feed(env("record_started", { category: "finish", payload: {} }, 2, { id: "r1" }));
    m.feed(env("record_failed", { error: { type: "x", message: "boom" } }, 3, { id: "r1" }));
    expect(roundOf(m)!.parts[1]).toMatchObject({ state: "failed", error: { type: "x", message: "boom" } });

    m.feed(env("record_started", { category: "sleep", payload: {} }, 4, { id: "r2" }));
    m.feed(env("record_void", {}, 5, { id: "r2" }));
    expect(roundOf(m)!.parts[2]).toMatchObject({ state: "void" });
  });

  it("ignores surface.*, unknown types and stale seq", () => {
    const m = new Mapper();
    m.feed(env("surface.stage.focus" as never, {}, 1));
    m.feed(env("nope" as never, {}, 2));
    m.feed(env("user_input", { text: "a" }, 3));
    m.feed(env("user_input", { text: "dup" }, 3)); // stale
    expect(m.getRounds().length).toBe(1);
    expect(roundOf(m)!.parts[0]).toMatchObject({ text: "a" });
  });

  it("seed restores rounds; same-key replay replaces the round", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m.feed(env("output_started", {}, 2, { id: "e1" }));
    m.feed(env("output_done", {}, 3, { id: "e1" }));

    // 模拟页面刷新：新 mapper + 种子
    const m2 = new Mapper();
    m2.seed(structuredClone(m.getRounds()) as HistoryRound[]);
    expect(m2.getRounds().length).toBe(1);

    // 重放同 mid 轮 → 覆盖替换而不是重复
    m2.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m2.feed(env("output_started", {}, 2, { id: "e1" }));
    expect(m2.getRounds().length).toBe(1);
    expect(m2.getRounds()[0]!.parts.length).toBe(2);
  });

  it("seed does not block cross-boot replay (seq reset)", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "old", mid: "old-1" }, 100));
    m.feed(env("user_input", { text: "old2", mid: "old-2" }, 101));

    const m2 = new Mapper();
    m2.seed(structuredClone(m.getRounds()) as HistoryRound[]);

    // 后端重启：seq 从 1 重新计数
    m2.feed(env("user_input", { text: "new", mid: "new-1" }, 1));
    expect(m2.getRounds().length).toBe(3);
    expect(m2.getRounds()[2]!.parts[0]).toMatchObject({ text: "new" });
  });

  it("events without an open round are dropped", () => {
    const m = new Mapper();
    m.feed(env("output_started", {}, 1, { id: "e1" }));
    m.feed(env("record_started", { category: "tool_call", payload: {} }, 2, { id: "r1" }));
    expect(m.getRounds().length).toBe(0);
  });

  it("notifies subscribers with the touched round", () => {
    const m = new Mapper();
    const seen: string[] = [];
    m.subscribe((r) => seen.push(r.key));
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m.feed(env("output_started", {}, 2, { id: "e1" }));
    m.feed(env("output_delta", { text: "x" }, 3, { id: "e1" }));
    m.feed(env("user_input", { text: "b", mid: "m2" }, 4)); // 关闭 m1 + 开 m2 = 2 次回调
    expect(seen).toEqual(["m1", "m1", "m1", "m1", "m2"]);
  });
});
