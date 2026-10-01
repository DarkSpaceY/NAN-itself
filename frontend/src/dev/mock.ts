// 开发用 mock 后端：绕过真实 WS，脚本化喂事件流。
// 使用：vite dev 下访问 http://localhost:5173/?mock

import type { Envelope } from "../protocol";

export interface MockClient {
  connect(): void;
  sendInput(text: string): void;
  pause(): void;
  resume(): void;
}

export interface MockClientOptions {
  onEnvelope: (env: Envelope) => void;
  onConnection: (state: "connecting" | "open" | "closed") => void;
}

export function createMockClient(opts: MockClientOptions): MockClient {
  let seq = 0;
  let running = false; // 用户触发的场景在跑
  let paused = false;
  let pauseTimer: ReturnType<typeof setTimeout> | null = null;
  let alive = false; // 生命循环已启动
  let lifeBusy = false; // 自主轮正在跑（用户场景要等它安静）

  const emit = (t: Envelope["t"], content: Record<string, unknown>, extra: Partial<Envelope> = {}) => {
    opts.onEnvelope({ seq: ++seq, ts: Date.now() / 1000, t, content, ...extra });
  };

  const sleep = async (ms: number) => {
    const end = Date.now() + ms;
    while (Date.now() < end) {
      // 暂停是协作式的：等待期间遇到暂停就挂着
      while (paused && Date.now() < end) await new Promise((r) => setTimeout(r, 150));
      await new Promise((r) => setTimeout(r, Math.min(150, Math.max(0, end - Date.now()))));
    }
  };

  const id = (n: number) => `mock-${Date.now()}-${n}`;

  const record = async (
    category: string,
    payload: Record<string, unknown>,
    ms: number,
    detail?: { entries: Array<Record<string, unknown>> },
    fail?: { type: string; message: string },
  ) => {
    const rid = id(Math.floor(Math.random() * 1e9));
    emit("record_started", { category, payload }, { id: rid });
    await sleep(ms);
    if (detail) emit("record_detail", detail, { id: rid });
    if (fail) emit("record_failed", { error: fail }, { id: rid });
    else emit("record_done", { duration_s: ms / 1000 }, { id: rid });
  };

  const say = async (lines: string[], speedMs = 26) => {
    const sid = id(Math.floor(Math.random() * 1e9));
    emit("output_started", {}, { id: sid });
    const text = lines.join("\n");
    for (let i = 0; i < text.length; i += 3) {
      emit("output_delta", { text: text.slice(i, i + 3) }, { id: sid });
      await sleep(speedMs);
    }
    emit("output_done", { duration_s: (text.length * speedMs) / 1000 }, { id: sid });
  };

  // ----------------------------------------------------------------
  // 自主轮：NAN 式持续运行——没有 user_input 的轮次，动作轻盈、
  // 出声克制。每 8~16 秒醒一次，挑一件小事做，然后继续休息。
  // ----------------------------------------------------------------

  const AMBIENT_THOUGHTS = [
    ["翻了翻新闻源，没什么值得打扰你的。", "把两条感兴趣的存进了待读清单。"],
    ["试了试新写的 index 脚本，跑通了。", "顺手把输出的缩进修了修。"],
    ["观察到 vision 模块整个下午都很安静。", "安静是好事。"],
  ];

  async function lifeTurn() {
    emit("status", { state: "working" });
    // 每个自主轮也要有轮锚点：否则所有自主活动堆进同一个永不
    // 关闭的轮，组永远不收起
    emit("turn_started", {});
    const roll = Math.random();

    if (roll < 0.4) {
      // 无事可做：休息，出声克制
      await record("sleep", { seconds: 30 }, 400);
    } else if (roll < 0.62) {
      // 盘点/查看
      const pick = ["list_skills", "list_channels", "show_channels"][
        Math.floor(Math.random() * 3)
      ];
      await record(pick, {}, 300);
    } else if (roll < 0.74) {
      // 观察
      await record(
        "module_query",
        { module: "vision" },
        700,
        { entries: [{ kind: "field", label: "faces", value: "0 detected" }, { kind: "field", label: "motion", value: "idle" }] },
      );
    } else if (roll < 0.9) {
      // 想了点什么，短句出声
      const thought = AMBIENT_THOUGHTS[Math.floor(Math.random() * AMBIENT_THOUGHTS.length)];
      await say(thought);
    } else {
      // 自我打理
      await record(
        "invoke_skill",
        { skill: "workspace-tidy" },
        1200,
        { entries: [{ kind: "field", label: "moved", value: "3 files" }, { kind: "text", text: "- scratch/ → notes/\n- 旧截图归档到 archive/" }] },
      );
    }
    emit("status", { state: "idle" });
  }

  async function lifeLoop() {
    if (alive) return;
    alive = true;
    emit("status", { state: "working" });
    // 开场先来一轮，让 ?mock 一打开就有东西看
    lifeBusy = true;
    await lifeTurn();
    lifeBusy = false;
    for (;;) {
      await sleep(1000 + Math.random() * 1000);
      // 用户场景优先：等它跑完再继续自主轮
      while (running) await sleep(300);
      lifeBusy = true;
      await lifeTurn();
      lifeBusy = false;
    }
  }

  // 用户场景队列：echo 立即（后端 ingest 即发 user_input），动作 FIFO 排队
  const scenarioQueue: Array<{ text: string; mid: string }> = [];

  async function pump() {
    if (running) return;
    running = true;
    while (scenarioQueue.length) {
      scenarioQueue.shift();
      emit("status", { state: "working" });
      // 真实语义：模型要等当前自主轮结束才看到输入
      while (alive && lifeBusy) await sleep(200);
      // 轮锚点：等待卡在此落位（用户气泡来自之前的 user_input 回执）
      emit("turn_started", {});
      await scenarioSteps();
    }
    emit("status", { state: "idle" });
    running = false;
  }

  async function scenarioSteps() {
    const id = (n: number) => `mock-${Date.now()}-${n}`;

    // 一轮的解剖：观察 → 说话 → 行动 → 说话 → 行动 → 收尾。
    // 文字穿插在动作间隙，每轮只有一个观察。

    // 1. 模块观察
    const q = id(1);
    emit("record_started", { category: "module_query", payload: { module: "vision" } }, { id: q });
    await sleep(600);
    emit(
      "record_detail",
      { entries: [{ kind: "field", label: "faces", value: "1 detected" }, { kind: "field", label: "motion", value: "idle" }] },
      { id: q },
    );
    await sleep(300);
    emit("record_done", { duration_s: 0.4 }, { id: q });

    // 2. 开场白：说要做什么
    const intro = id(2);
    emit("output_started", {}, { id: intro });
    const introText = "看到了，画面里有 **1 张脸**。我先看下项目文件，再给你汇总。";
    for (let i = 0; i < introText.length; i += 3) {
      emit("output_delta", { text: introText.slice(i, i + 3) }, { id: intro });
      await sleep(24);
    }
    emit("output_done", { duration_s: 0.8 }, { id: intro });

    // 3. 行动：读取文件
    const it = id(3);
    emit("record_started", { category: "invoke_tool", payload: { tool: "fs.read", path: "src/main.ts" } }, { id: it });
    await sleep(700);
    emit(
      "record_detail",
      { entries: [{ kind: "field", label: "status", value: "ok" }, { kind: "text", text: "// NAN frontend — bootstrap" }] },
      { id: it },
    );
    await sleep(300);
    emit("record_done", { duration_s: 1.1 }, { id: it });

    // 4. 中间结论
    const mid1 = id(4);
    emit("output_started", {}, { id: mid1 });
    const midText = [
      "看完了。当前是**数据层 + 对话流**的骨架：",
      "",
      "1. store —— 订阅式状态源",
      "2. ws —— 重连与重放",
      "3. mapper —— 把事件流折叠成对话轮",
      "",
      "接着确认通道状态，然后给你完整结论。",
    ].join("\n");
    for (let i = 0; i < midText.length; i += 3) {
      emit("output_delta", { text: midText.slice(i, i + 3) }, { id: mid1 });
      await sleep(24);
    }
    emit("output_done", { duration_s: 1.6 }, { id: mid1 });

    // 5. 行动：通道 + spawn（失败）
    const lc = id(5);
    emit("record_started", { category: "list_channels", payload: {} }, { id: lc });
    await sleep(400);
    emit("record_done", { duration_s: 0.1 }, { id: lc });

    const sp = id(6);
    emit("record_started", { category: "spawn", payload: { agent_id: "a1b2c3", depth: 1, task: "survey layout engines" } }, { id: sp });
    await sleep(900);
    emit("record_failed", { error: { type: "spawn_failed", message: "depth limit reached" } }, { id: sp });

    // 6. 收尾：完整 markdown 全要素（便于核对渲染）
    const stream = id(7);
    emit("output_started", {}, { id: stream });
    const reply = [
      "汇总：骨架没问题，`mapper` 已把事件流折叠成对话轮。",
      "",
      "```ts",
      "const messages = roundsToMessages(mapper.getRounds());",
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
      "> 子代理没法派（深度到顶了），不影响主线。我先歇会儿。",
    ].join("\n");
    for (let i = 0; i < reply.length; i += 3) {
      emit("output_delta", { text: reply.slice(i, i + 3) }, { id: stream });
      await sleep(24);
    }
    emit("output_done", { duration_s: 2.4 }, { id: stream });

    // 7. sleep：正常收尾
    const sl = id(8);
    emit("record_started", { category: "sleep", payload: { seconds: 30 } }, { id: sl });
    await sleep(500);
    emit("record_done", { duration_s: 0.1 }, { id: sl });
  }

  return {
    connect() {
      opts.onConnection("connecting");
      setTimeout(() => {
        opts.onConnection("open");
        void lifeLoop();
      }, 400);
    },
    sendInput(text) {
      // 与 WsClient 同一条客户端策略：暂停态下发消息先恢复。
      // （策略在客户端；后端不关心暂停与输入的先后。）
      if (paused || pauseTimer !== null) {
        this.resume();
      }
      // echo 立即（后端 ingest 即发 user_input 事件）；动作 FIFO 排队
      const mid = `mock-mid-${Date.now()}`;
      emit("user_input", { text, mid });
      scenarioQueue.push({ text, mid });
      void pump();
    },
    pause() {
      if (paused || pauseTimer !== null) return;
      // 协作式：模拟「当前轮跑完才停」的延迟后进入暂停态
      pauseTimer = setTimeout(() => {
        pauseTimer = null;
        paused = true;
        emit("status", { state: "paused" });
      }, 300);
    },
    resume() {
      if (pauseTimer !== null) {
        clearTimeout(pauseTimer);
        pauseTimer = null;
      }
      if (!paused) return;
      paused = false;
      emit("status", { state: "working" });
    },
  };
}
