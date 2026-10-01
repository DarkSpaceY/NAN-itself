import { describe, it, expect } from "vitest";
import { applyEvent, initialState, Store } from "../src/store";
import { classifyEvent, isEnvelope, newMid, type Envelope } from "../src/protocol";

function env(
  t: Envelope["t"],
  content: Record<string, unknown>,
  seq: number,
  extra: Partial<Envelope> = {},
): Envelope {
  return { seq, ts: 1000 + seq, t, content, ...extra };
}

describe("classifyEvent", () => {
  it("known types are live", () => {
    expect(classifyEvent(env("user_input", {}, 1))).toBe("live");
    expect(classifyEvent(env("record_started", {}, 1))).toBe("live");
  });

  it("surface.* and unknown are not live", () => {
    expect(classifyEvent(env("surface.stage.focus" as never, {}, 1))).toBe("surface");
    expect(classifyEvent(env("nope" as never, {}, 1) as never)).toBe("unknown");
  });
});

describe("isEnvelope", () => {
  it("accepts valid envelope, rejects hello/pong", () => {
    expect(isEnvelope(env("status", { state: "idle" }, 1))).toBe(true);
    expect(isEnvelope({ t: "hello", seq: 1, content: {} })).toBe(false);
    expect(isEnvelope({ t: "pong", content: {} })).toBe(false);
  });
});

describe("applyEvent", () => {
  it("user_input appends a round", () => {
    let s = initialState();
    s = applyEvent(s, env("user_input", { text: "hi", mid: "m1" }, 5));
    expect(s.rounds).toEqual([{ seq: 5, ts: 1005, text: "hi", mid: "m1" }]);
  });

  it("rejects out-of-order seq (replay/live overlap dedupe)", () => {
    let s = initialState();
    s = applyEvent(s, env("user_input", { text: "a" }, 3));
    const before = s;
    s = applyEvent(s, env("user_input", { text: "b" }, 3));
    expect(s).toBe(before);
    s = applyEvent(s, env("user_input", { text: "b" }, 2));
    expect(s).toBe(before);
  });

  it("status updates state", () => {
    let s = initialState();
    s = applyEvent(s, env("status", { state: "working" }, 1));
    expect(s.status).toBe("working");
    s = applyEvent(s, env("status", { state: "idle" }, 2));
    expect(s.status).toBe("idle");
    s = applyEvent(s, env("status", { state: "bogus" }, 3));
    expect(s.status).toBe("idle");
  });

  it("status paused flows through the store", () => {
    const store = new Store();
    store.applyMessage(env("status", { state: "paused" }, 1));
    expect(store.getState().status).toBe("paused");
    store.applyMessage(env("status", { state: "working" }, 2));
    expect(store.getState().status).toBe("working");
  });

  it("output/record events only advance lastSeq (no items state)", () => {
    let s = initialState();
    s = applyEvent(s, env("output_started", {}, 1, { id: "e1" }));
    s = applyEvent(s, env("record_started", { category: "tool_call" }, 2, { id: "r1" }));
    expect(s.lastSeq).toBe(2);
    expect(s.rounds).toEqual([]);
    // 旧 seq 的重复事件仍被去重
    const before = s;
    s = applyEvent(s, env("output_delta", { text: "x" }, 1, { id: "e1" }));
    expect(s).toBe(before);
  });
});

describe("Store", () => {
  it("notifies subscribers on change and not on no-op", () => {
    const store = new Store();
    let count = 0;
    store.subscribe(() => (count += 1));
    store.applyMessage(env("user_input", { text: "hi" }, 1));
    expect(count).toBe(1);
    store.applyMessage(env("user_input", { text: "dup" }, 1));
    expect(count).toBe(1);
    store.applyMessage({ t: "pong", content: {} });
    expect(count).toBe(1);
  });

  it("applyHello records bootId and replays rebuild state", () => {
    const store = new Store();
    store.applyHello({ seq: 0, content: { boot: "b1" } });
    expect(store.getState().bootId).toBe("b1");

    store.applyMessage(env("user_input", { text: "r1" }, 1));
    store.applyHello({ seq: 5, content: { boot: "b1" } });
    // 已有事件时不回退 lastSeq
    expect(store.getState().lastSeq).toBe(1);

    store.resetForReplay();
    expect(store.getState().rounds).toEqual([]);
    store.applyMessage(env("user_input", { text: "r1" }, 1));
    expect(store.getState().rounds).toHaveLength(1);
  });

  it("connection transitions", () => {
    const store = new Store();
    store.setConnection("connecting");
    expect(store.getState().connection).toBe("connecting");
    store.setConnection("open");
    store.setConnection("open");
    expect(store.getState().connection).toBe("open");
    store.resetForReplay();
    expect(store.getState().connection).toBe("open");
  });
});

describe("newMid", () => {
  it("yields unique ids", () => {
    expect(newMid()).not.toBe(newMid());
  });
});
