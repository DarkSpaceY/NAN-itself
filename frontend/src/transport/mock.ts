// Dev-only event stream: same envelope shape as the gateway, no backend.
//
// Not a "smooth round" demo: it walks every real branch of nan/agent:
//
//   1. plain text turn        closed by output_done
//   2. tool-call-only turn    output_started -> output_cancelled (no text)
//   3. round-level failure    turn_failed -> same turn_index retried after backoff
//   4. two-layer spawn        root -> a3 -> a7, with verb_failed for an unknown verb
//   5. pause interruption     agent_paused -> turn_cancelled -> reopened after resume
//
// Also replays injections_collected (one service ok, one failed) and
// agent_exited with report. Upstream messages are replayed with real
// semantics too.

import type { ClientMessage, Envelope, SurfaceSpec } from "../kernel/protocol";
import type { Hub } from "../kernel/history/hub";
import type { SurfaceHandler, Transport } from "./ws";

const BOOT = "mock-0001";
const ROOT = "r0";
const CHILD = "a3";
const GRANDCHILD = "a7";

/** Pacing between events; streaming deltas use STREAM_MS separately. */
const PACE_MS = 220;
const STREAM_MS = 40;
/** Characters per output_delta: simulates token granularity so the caret following is visible. */
const CHUNK = 2;

type Emit = (t: string, content: Record<string, unknown>, id?: string) => void;
/** Returns the wait in ms before the next step; omit for the default PACE_MS. */
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
  // no icon: should degrade to the initial P
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

/** Slice by code point so emoji / surrogate pairs are never split. */
function pieces(text: string, size = CHUNK): string[] {
  const chars = Array.from(text);
  const out: string[] = [];
  for (let i = 0; i < chars.length; i += size)
    out.push(chars.slice(i, i + size).join(""));
  return out;
}

/** Append text onto the same open stream without reopening it. */
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

/** Open a stream and emit the whole text as small deltas; the caller supplies the closing event. */
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
  /** Has a tool call: the backend sends output_cancelled; the text still goes into history. */
  cancelled?: boolean;
  retries?: number;
  calls?: number;
  latency?: number;
  history?: number;
}

/** One round: turn_started -> streamed output -> turn_done. */
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

/** One verb call: started -> (result -> done) | failed. */
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
  /** Highest turn_index the root has opened; retries and pause reopens reuse it. */
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
        // Same as the backend: the echo arrives first; turn_started comes with the agent's next round
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

  // -- driver ---------------------------------------------------------

  /** Advance by the delay the previous step returned; stop when the queue empties, wake on a new upstream message. */
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

  // -- script ---------------------------------------------------------

  private bootScript(): Step[] {
    const steps: Step[] = [
      (e) =>
        e("agent_started", {
          ...ROOT_BASE,
          parent_id: null,
          pacing: "event-driven",
        }),
    ];

    // 1. Plain text turn.
    userInput(steps, "看一遍仓库，给我一个结论");
    textTurn(steps, ROOT_BASE, {
      index: 1,
      text: "收到：看一遍仓库。先扫目录结构，再读关键模块。",
      latency: 1.2,
    });

    // 2. Tool turn: not a single character of text, the backend still sends started/cancelled.
    userInput(steps, "verbs 目录里都有什么？");
    const toolOut = outId();
    steps.push(
      (e) => e("turn_started", { ...ROOT_BASE, turn_index: 2, inbox_count: 1 }),
      // Every injected service reports, failures included
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

    // 3. Round-level failure: the whole round is voided; after backoff the same turn_index starts over.
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

    // 4. Two-layer spawn: root -> a3 -> a7; the unknown verb fails once only, the round continues.
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

    // a3: counts by itself first, then spawns one more layer.
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

    // a7 runs one round then exits with a report. The backend always sends agent_started first, or the tree cannot link.
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

    // The report lands in a3's inbox, so a3 opens a second round; it exits with a report too.
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

    // 5. Interjection: the user sends another message mid-stream. The
    //    echo arrives at once, but the inbox drains only at the next
    //    round boundary -- the bubble carries a queued marker meanwhile.
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

    // 6. pause interruption: the half-streamed sentence is voided with the round; the same turn_index reopens after resume.
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
