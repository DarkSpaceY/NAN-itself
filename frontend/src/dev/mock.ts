// 开发用 mock 后端：绕过真实 WS，脚本化喂事件流。
// 使用：vite dev 下访问 http://localhost:5173/?mock

import type { Envelope } from "../protocol";

export interface MockClient {
  connect(): void;
  sendInput(text: string): void;
}

export interface MockClientOptions {
  onEnvelope: (env: Envelope) => void;
  onConnection: (state: "connecting" | "open" | "closed") => void;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function createMockClient(opts: MockClientOptions): MockClient {
  let seq = 0;
  let running = false;

  const emit = (t: Envelope["t"], content: Record<string, unknown>, extra: Partial<Envelope> = {}) => {
    opts.onEnvelope({ seq: ++seq, ts: Date.now() / 1000, t, content, ...extra });
  };

  async function runScenario(userText: string, mid: string) {
    if (running) return;
    running = true;
    const id = (n: number) => `mock-${Date.now()}-${n}`;

    emit("user_input", { text: userText, mid });

    emit("status", { state: "working" });

    // 工具卡片：读文件
    const r1 = id(1);
    emit("record_started", { category: "tool_call", payload: { tool: "fs.read", path: "src/main.ts" } }, { id: r1 });
    await sleep(700);
    emit(
      "record_detail",
      { entries: [{ kind: "field", label: "status", value: "ok" }, { kind: "text", text: "// NAN frontend — bootstrap" }] },
      { id: r1 },
    );
    await sleep(300);
    emit("record_done", { duration_s: 1.1 }, { id: r1 });

    // 流式回复：刻意包含各种 markdown 元素，便于肉眼核对渲染
    const stream = id(2);
    emit("output_started", {}, { id: stream });
    const reply = [
      "已经读完 `main.ts`。当前是**数据层 + dock** 的骨架：",
      "",
      "1. store —— 订阅式状态源",
      "2. ws —— 重连与重放",
      "3. mapper —— 把事件流折叠成对话轮",
      "",
      "```ts",
      "const faces = deriveFaces(Object.values(state.items));",
      "const layout = computeLayout(faces);",
      "```",
      "",
      "| 层 | 职责 |",
      "| --- | --- |",
      "| store | 事件 → 状态 |",
      "| mapper | 事件 → 对话轮 |",
      "",
      "行内公式 $E = mc^2$，块级公式：",
      "",
      "$$\\int_0^1 x^2\\,dx = \\frac{1}{3}$$",
      "",
      "> 引用一段话，看看 blockquote 与长文本折行的表现。",
    ].join("\n");
    for (let i = 0; i < reply.length; i += 3) {
      emit("output_delta", { text: reply.slice(i, i + 3) }, { id: stream });
      await sleep(24);
    }
    emit("output_done", { duration_s: 2.4 }, { id: stream });

    // 第二个工具卡片：失败态
    const r2 = id(3);
    emit("record_started", { category: "subagent_spawn", payload: { agent_id: "a1b2c3", depth: 1, task: "survey layout engines" } }, { id: r2 });
    await sleep(900);
    emit("record_failed", { error: { type: "spawn_failed", message: "depth limit reached" } }, { id: r2 });

    emit("status", { state: "idle" });
    running = false;
  }

  return {
    connect() {
      opts.onConnection("connecting");
      setTimeout(() => opts.onConnection("open"), 400);
    },
    sendInput(text) {
      void runScenario(text, `mock-mid-${Date.now()}`);
    },
  };
}
