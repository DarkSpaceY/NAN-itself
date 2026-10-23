// 开发用事件流：与网关同一信封形状，不连后端。
//
// 不是"顺利的一轮"演示，而是照着 nan/agent 的真实分支把每条路径
// 走一遍：
//
//   1. 纯文本轮            output_done 收尾
//   2. 只有工具调用的轮    output_started → output_cancelled（没有文本）
//   3. 轮级失败            turn_failed → 同一 turn_index 退避重来
//   4. 两层 spawn          root → a3 → a7，带 unknown verb 的 verb_failed
//   5. pause 打断          agent_paused → turn_cancelled → resume 后重开
//
// 另外回放 injections_collected（一个成功、一个失败的服务）和带
// report 的 agent_exited。上行消息也按真实语义回放。

import type { ClientMessage, Envelope, SurfaceSpec } from "../kernel/protocol";
import type { Hub } from "../kernel/history/hub";
import type { SurfaceHandler, Transport } from "./ws";

const BOOT = "mock-0001";
const ROOT = "r0";
const CHILD = "a3";
const GRANDCHILD = "a7";

/** 事件之间的节奏；流式增量单独用 STREAM_MS 控制。 */
const PACE_MS = 220;
const STREAM_MS = 40;
/** 每个 output_delta 的字符数：模拟 token 粒度，能看清光标跟随。 */
const CHUNK = 2;

type Emit = (t: string, content: Record<string, unknown>, id?: string) => void;
/** 返回下一步之前的等待毫秒数；省略表示默认 PACE_MS。 */
type Step = (emit: Emit) => number | void;

const ROOT_BASE = { agent_id: ROOT, depth: 0 };
const CHILD_BASE = { agent_id: CHILD, depth: 1 };
const GRANDCHILD_BASE = { agent_id: GRANDCHILD, depth: 2 };

const DEMO_SURFACES: SurfaceSpec[] = [
  {
    service_id: "diff",
    url: "https://example.com/",
    title: "diff",
    icon: "M4 4h10l6 6v10H4z",
  },
  {
    service_id: "tests",
    url: "https://example.com/",
    title: "tests",
    icon: "M3 3h18v18H3zm8.5 13.5l5.5-6-1.5-1.4-4 4.4-2-2-1.4 1.4z",
  },
  {
    service_id: "logs",
    url: "https://example.com/",
    title: "logs",
    icon: "M3 5h18v14H3zm6.2 4.4l-2 2 2 2 1.2-1.2-1-1 1-1zm3.2 4.1h5v1.5h-5z",
  },
  // 没有 icon：应退化为首字母 P
  {
    service_id: "plan",
    url: "https://example.com/",
    title: "plan",
  },
];

let streamSeq = 0;

function outId(): string {
  streamSeq += 1;
  return `mock-out-${streamSeq}`;
}

function userInput(steps: Step[], text: string): void {
  steps.push((e) => e("user_input", { text }));
}

/** 按码点切块，避免把 emoji / 代理对劈开。 */
function pieces(text: string, size = CHUNK): string[] {
  const chars = Array.from(text);
  const out: string[] = [];
  for (let i = 0; i < chars.length; i += size)
    out.push(chars.slice(i, i + size).join(""));
  return out;
}

/** 在同一条已开的流上追加文本，不重开流。 */
function delta(
  steps: Step[],
  base: Record<string, unknown>,
  id: string,
  text: string,
): void {
  for (const piece of pieces(text)) {
    steps.push((e) => {
      e("output_delta", { ...base, text: piece }, id);
      return STREAM_MS;
    });
  }
}

/** 开一条流，把整段文本拆成小增量逐条发；收尾事件由调用方补。 */
function stream(
  steps: Step[],
  base: Record<string, unknown>,
  id: string,
  text: string,
): void {
  steps.push((e) => {
    e("output_started", base, id);
    return STREAM_MS;
  });
  delta(steps, base, id, text);
}

interface TurnOpts {
  index: number;
  text: string;
  inbox?: number;
  /** 有 tool call：后端发 output_cancelled，文本本身照样进 history。 */
  cancelled?: boolean;
  retries?: number;
  calls?: number;
  latency?: number;
  history?: number;
}

/** 一轮：turn_started → 流式输出 → turn_done。 */
function textTurn(
  steps: Step[],
  base: Record<string, unknown>,
  opts: TurnOpts,
): void {
  const out = outId();
  steps.push((e) =>
    e("turn_started", {
      ...base,
      turn_index: opts.index,
      inbox_count: opts.inbox ?? 1,
    }),
  );
  stream(steps, base, out, opts.text);
  steps.push((e) =>
    e(opts.cancelled ? "output_cancelled" : "output_done", base, out),
  );
  steps.push((e) =>
    e("turn_done", {
      ...base,
      turn_index: opts.index,
      latency_s: opts.latency ?? 1.2,
      retries: opts.retries ?? 0,
      calls: opts.calls ?? 0,
      history_chars: opts.history ?? 1800,
    }),
  );
}

interface VerbOpts {
  detail?: Record<string, unknown>;
  error?: { type: string; message: string };
  duration?: number;
}

/** 一个 verb 调用：started → (result → done) | failed。 */
function verb(
  steps: Step[],
  base: Record<string, unknown>,
  name: string,
  args: Record<string, unknown>,
  text: string,
  opts: VerbOpts = {},
): void {
  steps.push((e) =>
    e("verb_started", { ...base, verb: name, arguments: args }),
  );

  if (opts.error) {
    steps.push((e) =>
      e("verb_failed", { ...base, verb: name, error: opts.error }),
    );
    return;
  }

  steps.push((e) =>
    e("verb_result", {
      ...base,
      verb: name,
      text,
      ...(opts.detail ? { detail: opts.detail } : {}),
    }),
  );
  steps.push((e) =>
    e("verb_done", { ...base, verb: name, duration_s: opts.duration ?? 0.2 }),
  );
}

export class MockTransport implements Transport {
  readonly label = "mock";

  private seq = 0;
  private timer: number | null = null;
  private queue: Step[] = [];
  /** root 已开到的 turn_index；重试与 pause 重开会复用同一个。 */
  private rootTurns = 0;
  private started = false;

  constructor(private readonly hub: Hub) {}

  start(onSurface: SurfaceHandler): void {
    if (this.started) return;
    this.started = true;

    this.hub.global.setConnection("open");
    this.hub.global.applyHello({
      seq: 0,
      content: { boot: BOOT, model: "mock-model", paused: false },
    });

    for (const spec of DEMO_SURFACES) onSurface(spec);
    this.queue = this.bootScript();
    this.run();
  }

  stop(): void {
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    this.hub.global.setConnection("closed");
  }

  send(msg: ClientMessage): void {
    switch (msg.t) {
      case "input": {
        if (!msg.text.trim()) return;
        this.emit("user_input", { text: msg.text });
        // 与后端一致：回执先到，agent 下一轮才 turn_started
        const steps: Step[] = [];
        textTurn(steps, ROOT_BASE, {
          index: this.rootTurns + 1,
          text: `收到：${msg.text}。这是 mock 回放，不真的跑工具。`,
          latency: 0.9,
        });
        this.queue.push(...steps);
        if (this.timer === null) this.run();
        break;
      }
      case "pause":
        this.emit("agent_paused", { agent_id: ROOT, depth: 0 });
        break;
      case "resume":
        this.emit("agent_resumed", { agent_id: ROOT, depth: 0 });
        break;
      default:
        break;
    }
  }

  // -- 驱动机 --------------------------------------------------------

  /** 按上一步给出的延迟推进；队列空了就停，等新的上行消息再唤醒。 */
  private run(wait = 0): void {
    this.timer = window.setTimeout(() => {
      this.timer = null;
      const step = this.queue.shift();
      if (!step) return;
      const next = step((t, content, id) => this.emit(t, content, id));
      this.run(next ?? PACE_MS);
    }, wait);
  }

  private emit(t: string, content: Record<string, unknown>, id?: string): void {
    if (
      t === "turn_started" &&
      content.agent_id === ROOT &&
      typeof content.turn_index === "number"
    ) {
      this.rootTurns = Math.max(this.rootTurns, content.turn_index);
    }

    this.seq += 1;
    const env: Envelope = {
      t,
      seq: this.seq,
      ts: Date.now() / 1000,
      content,
    };
    if (id) env.id = id;
    this.hub.ingest(env);
  }

  // -- 剧本 ----------------------------------------------------------

  private bootScript(): Step[] {
    const steps: Step[] = [
      (e) =>
        e("agent_started", {
          ...ROOT_BASE,
          parent_id: null,
          pacing: "event-driven",
        }),
    ];

    // 1. 纯文本轮。
    userInput(steps, "看一遍仓库，给我一个结论");
    textTurn(steps, ROOT_BASE, {
      index: 1,
      text: "收到：看一遍仓库。先扫目录结构，再读关键模块。",
      latency: 1.2,
    });

    // 2. 工具轮：一个字的文本都没有，后端照样发 started/cancelled。
    userInput(steps, "verbs 目录里都有什么？");
    const toolOut = outId();
    steps.push(
      (e) => e("turn_started", { ...ROOT_BASE, turn_index: 2, inbox_count: 1 }),
      // 注入服务每个都报账，失败的也照报
      (e) =>
        e("injections_collected", {
          ...ROOT_BASE,
          services: [
            { service_id: "git", ok: true, duration_s: 0.03, text_len: 412 },
            { service_id: "clock", ok: false, duration_s: 0.0, text_len: 0 },
          ],
        }),
      (e) => e("output_started", ROOT_BASE, toolOut),
      (e) => e("output_cancelled", ROOT_BASE, toolOut),
    );
    verb(
      steps,
      ROOT_BASE,
      "shell",
      { command: "ls nan/agent/verbs", timeout: 30 },
      "core.py\nshell.py\nspawn.py\nfinish.py\nsend.py\n...",
      { duration: 0.31 },
    );
    steps.push((e) =>
      e("turn_done", {
        ...ROOT_BASE,
        turn_index: 2,
        latency_s: 1.7,
        retries: 0,
        calls: 1,
        history_chars: 2100,
      }),
    );

    // 3. 轮级失败：整轮作废、退避后同一个 turn_index 从头再来。
    userInput(steps, "跑一下测试");
    const failedOut = outId();
    steps.push((e) =>
      e("turn_started", { ...ROOT_BASE, turn_index: 3, inbox_count: 1 }),
    );
    stream(steps, ROOT_BASE, failedOut, "先跑测试。");
    steps.push((e) =>
      e("turn_failed", {
        ...ROOT_BASE,
        error: "RuntimeError: persona file unreadable",
        backoff_s: 1.0,
        retry_index: 0,
      }),
    );
    textTurn(steps, ROOT_BASE, {
      index: 3,
      text: "先跑测试。",
      retries: 1,
      latency: 3.4,
      history: 2200,
    });

    // 4. 两层 spawn：root → a3 → a7；unknown verb 只失败一次，不中断整轮。
    userInput(steps, "让子 agent 把 verbs 逐个读一遍");
    const spawnOut = outId();
    steps.push((e) =>
      e("turn_started", { ...ROOT_BASE, turn_index: 4, inbox_count: 1 }),
    );
    stream(steps, ROOT_BASE, spawnOut, "派一个子 agent 去读。");
    steps.push((e) => e("output_cancelled", ROOT_BASE, spawnOut));
    verb(
      steps,
      ROOT_BASE,
      "spawn",
      { task: "把 verbs 目录逐个读一遍" },
      `Subagent spawned.\nid: ${CHILD}\ndepth: 1`,
      {
        detail: { child_id: CHILD, depth: 1 },
        duration: 0.01,
      },
    );
    verb(steps, ROOT_BASE, "teleport", { target: "moon" }, "", {
      error: {
        type: "unknown_verb",
        message:
          "Unknown action 'teleport'. Available: shell, spawn, finish, send.",
      },
    });
    steps.push((e) =>
      e("turn_done", {
        ...ROOT_BASE,
        turn_index: 4,
        latency_s: 0.6,
        retries: 0,
        calls: 2,
        history_chars: 2600,
      }),
    );

    // a3：先自己数一遍，再 spawn 一层。
    const childOut = outId();
    steps.push(
      (e) =>
        e("agent_started", {
          ...CHILD_BASE,
          parent_id: ROOT,
          pacing: "event-driven",
        }),
      (e) =>
        e("turn_started", { ...CHILD_BASE, turn_index: 1, inbox_count: 1 }),
    );
    stream(steps, CHILD_BASE, childOut, "先数一遍，再让下一层读细节。");
    steps.push((e) => e("output_cancelled", CHILD_BASE, childOut));
    verb(
      steps,
      CHILD_BASE,
      "shell",
      { command: "ls nan/agent/verbs | wc -l" },
      "9",
      { duration: 0.18 },
    );
    verb(
      steps,
      CHILD_BASE,
      "spawn",
      { task: "读 shell verb 的实现" },
      `Subagent spawned.\nid: ${GRANDCHILD}\ndepth: 2`,
      { detail: { child_id: GRANDCHILD, depth: 2 }, duration: 0.01 },
    );
    steps.push((e) =>
      e("turn_done", {
        ...CHILD_BASE,
        turn_index: 1,
        latency_s: 1.4,
        retries: 0,
        calls: 2,
        history_chars: 1200,
      }),
    );

    // a7 跑一轮后带报告退出。后端一定先发 agent_started，树才连得上。
    steps.push((e) =>
      e("agent_started", {
        ...GRANDCHILD_BASE,
        parent_id: CHILD,
        pacing: "event-driven",
      }),
    );
    textTurn(steps, GRANDCHILD_BASE, {
      index: 1,
      text: "shell verb 只有一条执行路径。",
      latency: 0.8,
    });
    steps.push((e) =>
      e("agent_exited", {
        ...GRANDCHILD_BASE,
        reason: "finished",
        report: "shell verb 读完，实现很短",
      }),
    );

    // 报告进 a3 的 inbox，于是 a3 开第二轮；读完也带报告退出。
    textTurn(steps, CHILD_BASE, {
      index: 2,
      text: "下一层回报：verbs 一共九个，没有隐藏分支。",
      latency: 1.1,
    });
    steps.push((e) =>
      e("agent_exited", {
        ...CHILD_BASE,
        reason: "finished",
        report: "verbs 清点完毕，目录很浅",
      }),
    );

    // 5. 插话：本轮流式到一半用户又发一条。回执立刻到，但 inbox 要到
    //    下一个轮边界才被 drain —— 期间气泡带 queued 标记。
    const reportOut = outId();
    steps.push((e) =>
      e("turn_started", { ...ROOT_BASE, turn_index: 5, inbox_count: 1 }),
    );
    stream(steps, ROOT_BASE, reportOut, "子 agent 回报：verbs 清点完毕。");
    userInput(steps, "顺便把结论记一下");
    delta(steps, ROOT_BASE, reportOut, "结论是目录很浅。");
    steps.push((e) => e("output_done", ROOT_BASE, reportOut));
    steps.push((e) =>
      e("turn_done", {
        ...ROOT_BASE,
        turn_index: 5,
        latency_s: 1.3,
        retries: 0,
        calls: 0,
        history_chars: 4600,
      }),
    );
    textTurn(steps, ROOT_BASE, {
      index: 6,
      text: "记下了：verbs 一共九个，目录很浅。",
      latency: 0.9,
    });

    // 6. pause 打断：已流出的半句随整轮作废，resume 后同一 turn_index 重开。
    userInput(steps, "继续分析");
    const cancelledOut = outId();
    steps.push((e) =>
      e("turn_started", { ...ROOT_BASE, turn_index: 7, inbox_count: 1 }),
    );
    stream(steps, ROOT_BASE, cancelledOut, "继续分析（这半句会被 pause 丢掉）");
    steps.push(
      (e) => e("agent_paused", { agent_id: ROOT, depth: 0 }),
      (e) => e("turn_cancelled", { ...ROOT_BASE, requeued_count: 1 }),
      (e) => e("agent_resumed", { agent_id: ROOT, depth: 0 }),
    );
    textTurn(steps, ROOT_BASE, {
      index: 7,
      text: "继续分析。按目录深度排，结论没变。",
      latency: 1.5,
    });

    return steps;
  }
}
