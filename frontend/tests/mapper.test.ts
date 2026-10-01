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

/**
 * 标准一轮的开场：user_input 回执 → turn_started 锚点。
 * 用户气泡在 turn_started 时落位。
 */
function feedTurn(m: Mapper, seqs: [number, number], text?: string): void {
  if (text !== undefined) {
    m.feed(env("user_input", { text, mid: `m${seqs[0]}` }, seqs[0]));
  }
  m.feed(env("turn_started", {}, seqs[1]));
}

/** 一轮的用户消息文本（回执在 turn_started 时落位）。 */
function userTexts(r: HistoryRound): string[] {
  return r.parts
    .filter((p) => p.kind === "text" && p.role === "user")
    .map((p) => (p as { text: string }).text);
}

function roundOf(m: Mapper): HistoryRound | null {
  return m.getCurrent();
}

/** 该轮内所有助手文本片段，按顺序。 */
function assistantTexts(r: HistoryRound): string[] {
  return r.parts
    .filter((p) => p.kind === "text" && p.role === "assistant")
    .map((p) => (p as { text: string }).text);
}

/** 该轮的助手文本部件（断言其字段用）。 */
function assistantPart(r: HistoryRound) {
  return r.parts.find((p) => p.kind === "text" && p.role === "assistant");
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
  it("user_input is a receipt only: pending card, no round", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    // 回执不构造轮，只进等待卡
    expect(m.getRounds().length).toBe(0);
    expect(m.getPendingTexts()).toEqual(["hi"]);
  });

  it("turn_started consumes pending receipts into user text parts", () => {
    const m = new Mapper();
    m.feed(env("user_input", { text: "hi" }, 1));
    m.feed(env("user_input", { text: "again" }, 2));
    // 输入在等待卡里，不截断任何流
    expect(m.getPendingTexts()).toEqual(["hi", "again"]);

    m.feed(env("turn_started", {}, 3));
    expect(m.getPendingTexts()).toEqual([]);
    const r = roundOf(m)!;
    expect(r.key).toBe("turn-3");
    expect(r.closed).toBe(false);
    expect(userTexts(r)).toEqual(["hi", "again"]);
  });

  it("next turn_started closes the previous round", () => {
    const m = new Mapper();
    feedTurn(m, [1, 2], "a");
    feedTurn(m, [3, 4], "b");
    expect(m.getRounds().length).toBe(2);
    expect(m.getRounds()[0]!.closed).toBe(true);
    expect(userTexts(m.getRounds()[0]!)).toEqual(["a"]);
    expect(userTexts(m.getRounds()[1]!)).toEqual(["b"]);
  });

  it("autonomous turn_started opens a round without user parts", () => {
    const m = new Mapper();
    m.feed(env("turn_started", {}, 1));
    const r = roundOf(m)!;
    expect(r.key).toBe("turn-1");
    expect(userTexts(r)).toEqual([]);
  });

  it("assistant stream: output parts accumulate then finish", () => {
    const m = new Mapper();
    feedTurn(m, [1, 2], "hi");
    m.feed(env("output_started", {}, 3, { id: "e1" }));
    m.feed(env("output_delta", { text: "he" }, 4, { id: "e1" }));
    m.feed(env("output_delta", { text: "y" }, 5, { id: "e1" }));

    let r = roundOf(m)!;
    expect(r.parts[1]).toMatchObject({ kind: "text", role: "assistant", text: "hey", streaming: true });

    m.feed(env("output_done", { duration_s: 2 }, 6, { id: "e1" }));
    r = roundOf(m)!;
    expect(r.parts[1]).toMatchObject({ text: "hey", streaming: false });
  });

  it("cancelled output marks the part", () => {
    const m = new Mapper();
    feedTurn(m, [1, 2], "hi");
    m.feed(env("output_started", {}, 3, { id: "e1" }));
    m.feed(env("output_cancelled", {}, 4, { id: "e1" }));
    expect(roundOf(m)!.parts[1]).toMatchObject({ cancelled: true, streaming: false });
  });

  it("record lifecycle becomes a tool card with detail and states", () => {
    const m = new Mapper();
    feedTurn(m, [1, 2], "hi");
    m.feed(
      env("record_started", { category: "tool_call", payload: { tool: "fs.read" } }, 3, { id: "r1" }),
    );
    m.feed(env("record_detail", { entries: [{ kind: "field", label: "status", value: "ok" }] }, 4, { id: "r1" }));
    m.feed(env("record_done", { duration_s: 0.5 }, 5, { id: "r1" }));

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
    feedTurn(m, [1, 2], "hi");
    m.feed(env("record_started", { category: "finish", payload: {} }, 3, { id: "r1" }));
    m.feed(env("record_failed", { error: { type: "x", message: "boom" } }, 4, { id: "r1" }));
    expect(roundOf(m)!.parts[1]).toMatchObject({ state: "failed", error: { type: "x", message: "boom" } });

    m.feed(env("record_started", { category: "sleep", payload: {} }, 5, { id: "r2" }));
    m.feed(env("record_void", {}, 6, { id: "r2" }));
    expect(roundOf(m)!.parts[2]).toMatchObject({ state: "void" });
  });

  it("ignores surface.*, unknown types and stale seq", () => {
    const m = new Mapper();
    m.feed(env("surface.stage.focus" as never, {}, 1));
    m.feed(env("nope" as never, {}, 2));
    m.feed(env("user_input", { text: "a" }, 3));
    m.feed(env("turn_started", {}, 4));
    m.feed(env("user_input", { text: "stale" }, 4)); // stale
    m.feed(env("turn_started", {}, 4)); // stale
    expect(m.getRounds().length).toBe(1);
    expect(userTexts(roundOf(m)!)).toEqual(["a"]);
  });

  it("seed restores rounds; same-key replay replaces the round", () => {
    const m = new Mapper();
    feedTurn(m, [1, 2], "hi");
    m.feed(env("output_started", {}, 3, { id: "e1" }));
    m.feed(env("output_done", {}, 4, { id: "e1" }));

    // 模拟页面刷新：新 mapper + 种子
    const m2 = new Mapper();
    m2.seed(structuredClone(m.getRounds()) as HistoryRound[]);
    expect(m2.getRounds().length).toBe(1);

    // 重放同 key 轮 → 覆盖替换而不是重复
    m2.feed(env("user_input", { text: "hi", mid: "m1" }, 1));
    m2.feed(env("turn_started", {}, 2));
    m2.feed(env("output_started", {}, 3, { id: "e1" }));
    expect(m2.getRounds().length).toBe(1);
    expect(m2.getRounds()[0]!.parts.length).toBe(2);
  });

  it("seed does not block cross-boot replay (seq reset)", () => {
    const m = new Mapper();
    feedTurn(m, [100, 101], "old");
    feedTurn(m, [102, 103], "old2");

    const m2 = new Mapper();
    m2.seed(structuredClone(m.getRounds()) as HistoryRound[]);

    // 后端重启：seq 从 1 重新计数
    feedTurn(m2, [1, 2], "new");
    expect(m2.getRounds().length).toBe(3);
    expect(userTexts(m2.getRounds()[2]!)).toEqual(["new"]);
  });

  it("input during a running turn waits in pending until the next turn_started", () => {
    const m = new Mapper();
    m.feed(env("turn_started", {}, 1)); // 自主轮
    m.feed(env("output_started", {}, 2, { id: "e1" }));
    m.feed(env("output_delta", { text: "thinking..." }, 3, { id: "e1" }));
    // 运行中输入：只进等待卡，轮不被打断、不被截断
    m.feed(env("user_input", { text: "hi" }, 4));
    expect(m.getRounds().length).toBe(1);
    expect(m.getPendingTexts()).toEqual(["hi"]);

    // 下一轮开始：等待卡落位成新轮的用户消息
    m.feed(env("turn_started", {}, 5));
    expect(m.getRounds().length).toBe(2);
    expect(m.getRounds()[0]!.closed).toBe(true);
    expect(userTexts(m.getRounds()[1]!)).toEqual(["hi"]);
    expect(m.getPendingTexts()).toEqual([]);
  });

  it("notifies subscribers with the touched round", () => {
    const m = new Mapper();
    const seen: string[] = [];
    m.subscribe((r) => seen.push(r?.key ?? "(none)"));
    m.feed(env("user_input", { text: "hi" }, 1));
    m.feed(env("turn_started", {}, 2));
    m.feed(env("output_started", {}, 3, { id: "e1" }));
    m.feed(env("output_delta", { text: "x" }, 4, { id: "e1" }));
    // 运行中的输入：等待卡 touch（回执无轮上下文，通知 open 轮）
    m.feed(env("user_input", { text: "b" }, 5));
    m.feed(env("turn_started", {}, 6)); // 关 turn-2 + 开 turn-6 = 2 次回调
    expect(seen).toEqual(["(none)", "turn-2", "turn-2", "turn-2", "turn-2", "turn-2", "turn-6"]);
  });

  // 回归：rounds 数组是原地 push 的，引用恒定，React 的
  // useSyncExternalStore/useMemo 只能靠版本号判断变化。
  it("version increments on every content change, not just new rounds", () => {
    const m = new Mapper();
    expect(m.getVersion()).toBe(0);

    feedTurn(m, [1, 2], "hi");
    const v1 = m.getVersion();
    expect(v1).toBeGreaterThan(0);

    // 流式 delta 不新增轮，但必须让版本号前进（否则 UI 不重渲染）
    m.feed(env("output_started", {}, 3, { id: "e1" }));
    m.feed(env("output_delta", { text: "x" }, 4, { id: "e1" }));
    expect(m.getVersion()).toBeGreaterThan(v1);
    expect(m.getRounds().length).toBe(1);

    // 回执（无开启轮）也推进版本号（等待卡更新）
    const m2 = new Mapper();
    m2.feed(env("user_input", { text: "b" }, 1));
    expect(m2.getVersion()).toBeGreaterThan(0);
  });

  it("seed notifies subscribers so restored history renders", () => {
    const m = new Mapper();
    feedTurn(m, [1, 2], "hi");

    const m2 = new Mapper();
    let notified = 0;
    m2.subscribe(() => notified++);
    m2.seed(structuredClone(m.getRounds()) as HistoryRound[]);

    expect(notified).toBe(1);
    expect(m2.getVersion()).toBeGreaterThan(0);
  });

  // 全新客户端（无本地历史）：全靠后端重放建轮，每轮的输出必须
  // 挂回自己那一轮，而不是被挂到数组末位。
  it("fresh client rebuilds rounds from replay, each keeping its own output", () => {
    const m = new Mapper();
    m.onHello("b1");
    const events: Envelope[] = [
      env("user_input", { text: "A", mid: "mA" }, 1, { boot_id: "b1" }),
      env("turn_started", {}, 2, { boot_id: "b1" }),
      env("output_started", {}, 3, { id: "eA", boot_id: "b1" }),
      env("output_delta", { text: "reply-A" }, 4, { id: "eA", boot_id: "b1" }),
      env("output_done", {}, 5, { id: "eA", boot_id: "b1" }),
      env("user_input", { text: "B", mid: "mB" }, 6, { boot_id: "b1" }),
      env("turn_started", {}, 7, { boot_id: "b1" }),
      env("output_started", {}, 8, { id: "eB", boot_id: "b1" }),
      env("output_delta", { text: "reply-B" }, 9, { id: "eB", boot_id: "b1" }),
      env("output_done", {}, 10, { id: "eB", boot_id: "b1" }),
    ];
    for (const e of events) m.feed(e);

    const rounds = m.getRounds();
    expect(rounds.length).toBe(2);
    expect(assistantTexts(rounds[0]!)).toEqual(["reply-A"]);
    expect(assistantTexts(rounds[1]!)).toEqual(["reply-B"]);
  });

  // F5（同一 boot）：游标挡住重放里已经应用过的部分，文本不重复。
  it("same-boot replay after seed is skipped by the persisted cursor", () => {
    const live = new Mapper();
    live.onHello("b1");
    const events: Envelope[] = [
      env("user_input", { text: "A", mid: "mA" }, 1, { boot_id: "b1" }),
      env("turn_started", {}, 2, { boot_id: "b1" }),
      env("output_started", {}, 3, { id: "eA", boot_id: "b1" }),
      env("output_delta", { text: "reply-A" }, 4, { id: "eA", boot_id: "b1" }),
      env("output_done", {}, 5, { id: "eA", boot_id: "b1" }),
    ];
    for (const e of events) live.feed(e);

    const m = new Mapper();
    m.seed(structuredClone(live.getRounds()) as HistoryRound[]);
    m.onHello("b1");
    for (const e of events) m.feed(structuredClone(e));

    const rounds = m.getRounds();
    expect(rounds.length).toBe(1);
    expect(assistantTexts(rounds[0]!)).toEqual(["reply-A"]);
  });

  // 后端重启：seq 从头计数，游标必须归零，否则新事件会被 seq 守卫
  // 全部丢弃（界面冻住）。
  it("hello with a new boot resets the cursor so new events apply", () => {
    const live = new Mapper();
    live.onHello("b1");
    live.feed(env("user_input", { text: "A", mid: "mA" }, 400, { boot_id: "b1" }));
    live.feed(env("turn_started", {}, 401, { boot_id: "b1" }));
    live.feed(env("output_started", {}, 402, { id: "eA", boot_id: "b1" }));

    const m = new Mapper();
    m.seed(structuredClone(live.getRounds()) as HistoryRound[]);
    m.onHello("b2"); // 新 boot，seq 回到 1
    feedTurn(m, [1, 2], "new");

    expect(m.getRounds().length).toBe(2);
    expect(userTexts(m.getRounds()[1]!)).toEqual(["new"]);
  });

  // 半途刷新：轮未关闭，恢复为进行中，后续 delta 接着追加而不是丢
  it("mid-turn resume appends later deltas to the in-flight round", () => {
    const live = new Mapper();
    live.onHello("b1");
    live.feed(env("user_input", { text: "A", mid: "mA" }, 1, { boot_id: "b1" }));
    live.feed(env("turn_started", {}, 2, { boot_id: "b1" }));
    live.feed(env("output_started", {}, 3, { id: "eA", boot_id: "b1" }));
    live.feed(env("output_delta", { text: "part1" }, 4, { id: "eA", boot_id: "b1" }));

    const m = new Mapper();
    m.seed(structuredClone(live.getRounds()) as HistoryRound[]);
    m.onHello("b1");
    m.feed(env("output_delta", { text: "part2" }, 5, { id: "eA", boot_id: "b1" }));
    m.feed(env("output_done", {}, 6, { id: "eA", boot_id: "b1" }));

    expect(assistantTexts(m.getRounds()[0]!)).toEqual(["part1part2"]);
  });

  // 旧版记录没有游标，且重放窗口里 turn_started 已滑出：重见同一
  // output id 时重建该 part，而不是再挂一个。
  it("known output id is rebuilt instead of duplicated", () => {
    const m = new Mapper();
    m.seed([
      {
        key: "turn-2",
        ts: 1000,
        seq: 2,
        appliedSeq: 0,
        bootId: null,
        closed: false,
        parts: [
          { kind: "text", id: "turn-2:u0", role: "user", text: "A" },
          { kind: "text", id: "eA", role: "assistant", text: "reply-A" },
        ],
      },
    ]);
    m.onHello(null);
    m.feed(env("output_started", {}, 3, { id: "eA" }));
    m.feed(env("output_delta", { text: "reply-A" }, 4, { id: "eA" }));

    expect(assistantTexts(m.getRounds()[0]!)).toEqual(["reply-A"]);
  });

  // 流未结束就刷新：streaming 是瞬时状态，不能跟着落盘的状态复活
  // （否则界面上留下永不消失的光标）。
  it("seed clears transient streaming state, later deltas re-arm it", () => {
    const live = new Mapper();
    live.onHello("b1");
    live.feed(env("user_input", { text: "A", mid: "mA" }, 1, { boot_id: "b1" }));
    live.feed(env("turn_started", {}, 2, { boot_id: "b1" }));
    live.feed(env("output_started", {}, 3, { id: "eA", boot_id: "b1" }));
    live.feed(env("output_delta", { text: "half" }, 4, { id: "eA", boot_id: "b1" }));
    expect(assistantPart(live.getRounds()[0]!)).toMatchObject({ streaming: true });

    const m = new Mapper();
    m.seed(structuredClone(live.getRounds()) as HistoryRound[]);
    expect(assistantPart(m.getRounds()[0]!)).toMatchObject({ streaming: false });

    // 这一路流其实还活着：下一个 delta 到达时重新标为流式中
    m.onHello("b1");
    m.feed(env("output_delta", { text: "rest" }, 5, { id: "eA", boot_id: "b1" }));
    expect(assistantPart(m.getRounds()[0]!)).toMatchObject({
      streaming: true,
      text: "halfrest",
    });
  });

  // 兜底：没有 turn_started 的旧事件流挂到隐式轮，不丢弃
  it("events without turn_started fall back to an implicit round", () => {
    const m = new Mapper();
    m.feed(env("output_started", {}, 1, { id: "e1" }));
    m.feed(env("record_started", { category: "tool_call", payload: {} }, 2, { id: "r1" }));
    expect(m.getRounds().length).toBe(1);
    expect(m.getRounds()[0].key).toBe("auto-1");

    // turn_started 到达后：隐式轮关闭，正式轮开启
    m.feed(env("turn_started", {}, 3));
    expect(m.getRounds().length).toBe(2);
    expect(m.getRounds()[0].key).toBe("auto-1");
    expect(m.getRounds()[0].closed).toBe(true);
    expect(m.getRounds()[1].key).toBe("turn-3");
  });
});
