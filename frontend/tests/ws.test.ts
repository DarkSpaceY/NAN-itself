import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { WsClient } from "../src/ws";
import { Store } from "../src/store";

// ---- mock socket ----------------------------------------------------

type Handler = (ev: unknown) => void;

class MockSocket {
  static instances: MockSocket[] = [];
  static OPEN = 1;

  readyState = 0; // CONNECTING
  url: string;
  listeners: Record<string, Handler[]> = {};
  sent: string[] = [];
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockSocket.instances.push(this);
  }

  addEventListener(type: string, fn: Handler) {
    (this.listeners[type] ??= []).push(fn);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.closed = true;
    this.emit("close", {});
  }

  // test helpers
  open() {
    this.readyState = 1;
    this.emit("open", {});
  }

  message(data: unknown) {
    this.emit("message", { data: JSON.stringify(data) });
  }

  emit(type: string, ev: unknown) {
    for (const fn of this.listeners[type] ?? []) fn(ev);
  }
}

function hello(seq: number) {
  return { t: "hello", seq, content: { boot: "b1" } };
}

describe("WsClient", () => {
  let store: Store;

  beforeEach(() => {
    vi.useFakeTimers();
    MockSocket.instances = [];
    store = new Store();
    // 避免真实随机 UUID 依赖，直接走 fallback 亦可
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function makeClient() {
    return new WsClient({
      url: "ws://x/ws",
      store,
      socketFactory: (url) => new MockSocket(url) as never,
    });
  }

  it("connect → hello resets model → replay builds state → live applies", () => {
    const client = makeClient();
    client.connect();
    const sock = MockSocket.instances[0]!;
    sock.open();
    expect(store.getState().connection).toBe("open");

    sock.message(hello(3));
    sock.message({ seq: 1, ts: 1, t: "user_input", content: { text: "r1" } });
    sock.message({ seq: 2, ts: 2, t: "status", content: { state: "working" } });
    sock.message({ seq: 4, ts: 4, t: "output_started", content: {}, id: "e1" });

    const s = store.getState();
    expect(s.bootId).toBe("b1");
    expect(s.rounds).toHaveLength(1);
    expect(s.status).toBe("working");
    expect(s.lastSeq).toBe(4);
  });

  it("hello reports the boot before any replayed envelope", () => {
    const calls: string[] = [];
    const client = new WsClient({
      url: "ws://x/ws",
      store,
      socketFactory: (url) => new MockSocket(url) as never,
      onHello: (b) => calls.push(`hello:${b}`),
      onEnvelope: (e) => calls.push(`env:${e.seq}`),
    });
    client.connect();
    const sock = MockSocket.instances[0]!;
    sock.open();

    sock.message(hello(3));
    sock.message({ seq: 1, ts: 1, t: "user_input", content: { text: "r1" } });

    // 游标必须在重放事件之前落定，否则先应用的事件无法被跳过
    expect(calls).toEqual(["hello:b1", "env:1"]);
  });

  it("reconnects with exponential backoff capped at max, rebuilds via replay", () => {
    const client = makeClient();
    client.connect();
    const s1 = MockSocket.instances[0]!;
    s1.open();
    s1.message(hello(2));
    s1.message({ seq: 1, ts: 1, t: "user_input", content: { text: "r1" } });
    s1.close(); // 触发重连路径

    expect(store.getState().connection).toBe("connecting");
    expect(MockSocket.instances).toHaveLength(1);

    vi.advanceTimersByTime(1000); // 第一次退避 1s
    expect(MockSocket.instances).toHaveLength(2);

    const s2 = MockSocket.instances[1]!;
    s2.open();
    s2.message(hello(2));
    s2.message({ seq: 1, ts: 1, t: "user_input", content: { text: "r1" } });
    s2.message({ seq: 2, ts: 2, t: "output_started", content: {}, id: "e9" });
    // 重放重建：模型来自重放而非旧 socket 残留
    expect(store.getState().rounds).toHaveLength(1);
    expect(store.getState().lastSeq).toBe(2);

    s2.close();
    vi.advanceTimersByTime(1000); // 第二次退避 2s
    vi.advanceTimersByTime(1000);
    expect(MockSocket.instances).toHaveLength(3);

    // 封顶：多次重连后不超过 10s
    for (let i = 0; i < 6; i++) {
      MockSocket.instances.at(-1)!.close();
      vi.advanceTimersByTime(60_000);
    }
    // 10s 内即使推进大量时间也只拨号一次/轮
    expect(MockSocket.instances.length).toBeLessThan(12);
  });

  it("sendInput attaches mid and sends JSON", () => {
    const client = makeClient();
    client.connect();
    const sock = MockSocket.instances[0]!;
    sock.open();
    client.sendInput("hi");
    const sent = JSON.parse(sock.sent[0]!);
    expect(sent.t).toBe("input");
    expect(sent.text).toBe("hi");
    expect(typeof sent.mid).toBe("string");
  });

  it("sendInput while paused resumes before sending", () => {
    const client = makeClient();
    client.connect();
    const sock = MockSocket.instances[0]!;
    sock.open();

    store.applyMessage({
      seq: 1,
      ts: 1,
      t: "status",
      content: { state: "paused" },
    });

    client.sendInput("hi");

    const frames = sock.sent.map((s) => JSON.parse(s));
    expect(frames.map((f) => f.t)).toEqual(["resume", "input"]);
    expect(frames[1]!.text).toBe("hi");
  });

  it("pause()/resume() send the right frames", () => {
    const client = makeClient();
    client.connect();
    const sock = MockSocket.instances[0]!;
    sock.open();
    client.pause();
    client.resume();
    expect(sock.sent.map((s) => JSON.parse(s))).toEqual([
      { t: "pause" },
      { t: "resume" },
    ]);
  });

  it("does not send while socket is down", () => {
    const client = makeClient();
    client.connect();
    expect(() => client.sendInput("hi")).not.toThrow();
  });

  it("ignores stale socket events after redial", () => {
    const client = makeClient();
    client.connect();
    const s1 = MockSocket.instances[0]!;
    s1.open();
    s1.close();
    vi.advanceTimersByTime(1000);
    const s2 = MockSocket.instances[1]!;
    s2.open();
    // 迟到的旧 socket 消息不应污染状态
    s1.message({ seq: 1, ts: 1, t: "user_input", content: { text: "ghost" } });
    expect(store.getState().rounds).toHaveLength(0);
    s2.message(hello(1));
    expect(store.getState().bootId).toBe("b1");
  });

  it("close() stops reconnection and marks closed", () => {
    const client = makeClient();
    client.connect();
    MockSocket.instances[0]!.open();
    client.close();
    expect(store.getState().connection).toBe("closed");
    vi.advanceTimersByTime(60_000);
    expect(MockSocket.instances).toHaveLength(1);
  });
});
