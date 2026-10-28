import { afterEach, describe, expect, it, vi } from "vitest";

import { Hub } from "../src/kernel/history/hub";
import { GlobalStore } from "../src/kernel/store";
import type { SurfaceSpec } from "../src/kernel/protocol";
import { WsTransport } from "../src/transport/ws";

// A socket that never opens: the test drives the parser directly and
// leaves the connection lifecycle (and its window timers) untouched.
class IdleSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;

  close(): void {}

  send(): void {}
}

function makeTransport() {
  const transport = new WsTransport(new Hub(new GlobalStore()), "ws://127.0.0.1:1/ws");
  const added: SurfaceSpec[] = [];
  const removed: string[] = [];
  transport.start(
    (spec) => added.push(spec),
    (id) => removed.push(id),
  );
  return { transport, added, removed };
}

function deliver(transport: WsTransport, frame: unknown): void {
  (transport as unknown as { handle(raw: string): void }).handle.call(
    transport,
    JSON.stringify(frame),
  );
}

describe("ws transport", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("delivers surface frames, which carry no seq", () => {
    vi.stubGlobal("WebSocket", IdleSocket);
    const { transport, added, removed } = makeTransport();

    deliver(transport, {
      t: "surface_added",
      content: { service_id: "record", url: "http://127.0.0.1:9003/", title: "Record" },
    });
    deliver(transport, { t: "surface_removed", content: { service_id: "record" } });

    expect(added).toEqual([
      { service_id: "record", url: "http://127.0.0.1:9003/", title: "Record" },
    ]);
    expect(removed).toEqual(["record"]);
    transport.stop();
  });

  it("still hands sequenced events to the hub", () => {
    vi.stubGlobal("WebSocket", IdleSocket);
    const { transport, added } = makeTransport();

    deliver(transport, {
      t: "user_input",
      seq: 1,
      ts: 1,
      content: { text: "hi" },
    });

    expect(added).toEqual([]);
    transport.stop();
  });
});
