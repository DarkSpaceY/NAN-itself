// 事件契约：核心进程 → 桌面，唯一真相。
//
// 信封由 sink 统一盖章（t / id / seq / ts / boot_id），业务字段全部
// 在 content 里，按 t 分派。身份字段是 agent_id / parent_id / depth，
// 由发射点写进 content —— 见 nan/agent/core.py 与 engine.py。
//
// 前端是这条流的渲染器：不推断、不补发、不缓存轮次。重连拿到
// hello 之后由网关重放环形缓冲，seq 只用于去重。

export interface Envelope {
  t: string;
  /** 关联 id：output_* 家族同 id 配对；其余事件也带，可忽略。 */
  id?: string;
  /** 进程内单调序号，跨重连用于去重与缺口检测。 */
  seq: number;
  /** unix 秒。 */
  ts: number;
  /** 只在 sink 带 boot id 时出现。 */
  boot_id?: string;
  content: Record<string, unknown>;
}

/** 服务端下行：事件、握手、心跳应答。 */
export interface Hello {
  t: "hello";
  seq: number;
  content: { boot?: string; model?: string; paused?: boolean };
}

export interface Pong {
  t: "pong";
}

/** 前端上行。 */
export type ClientMessage =
  | { t: "input"; text: string; mid?: string }
  | { t: "pause" }
  | { t: "resume" }
  | { t: "ping" };

// ------------------------------------------------------------------
// content 载荷
// ------------------------------------------------------------------

export interface AgentIdentity {
  agent_id: string;
  depth?: number;
}

export interface AgentStarted extends AgentIdentity {
  parent_id: string | null;
  pacing: "interval" | "event-driven";
}

export interface TurnStarted extends AgentIdentity {
  turn_index: number;
  inbox_count: number;
}

export interface TurnDone extends AgentIdentity {
  turn_index: number;
  latency_s: number;
  retries: number;
  calls: number;
  history_chars: number;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  } | null;
}

export interface TurnFailed extends AgentIdentity {
  error: string;
  backoff_s: number;
  retry_index: number;
}

/** 一个 verb 的调用：名字 + 结构化参数。 */
export interface VerbStarted extends AgentIdentity {
  verb: string;
  arguments: Record<string, unknown>;
}

/** verb 返回给模型的原文；detail 是给界面看的结构化摘要。 */
export interface VerbResult extends AgentIdentity {
  verb: string;
  text: string;
  detail?: Record<string, unknown> | null;
}

export interface VerbDone extends AgentIdentity {
  verb: string;
  duration_s: number;
}

export interface VerbFailed extends AgentIdentity {
  verb: string;
  error: { type: string; message: string };
}

export interface OutputDelta extends AgentIdentity {
  text: string;
}

export interface StatusContent extends AgentIdentity {
  state: "working" | "idle" | "paused";
}

export interface UserInput {
  text: string;
  mid?: string;
}

export interface AgentExited extends AgentIdentity {
  reason: string;
  report?: string | null;
}

/**
 * 窗口注册（`surface_added`）。后端尚未实现：`nan/injection/hub.py`
 * 注册的是上下文服务，不是窗口。桌面临时从 mock 流读取同一形状。
 */
export interface SurfaceSpec {
  service_id: string;
  url: string;
  title: string;
  icon?: string;
}

// ------------------------------------------------------------------
// 已知类型表
// ------------------------------------------------------------------

/**
 * 桌面消费的事件类型。`surface.*` 是窗口注册预留命名空间，暂不消费；
 * 其他未知类型直接忽略（后加事件不能让旧桌面崩）。
 */
export const EVENT_TYPES = new Set([
  "agent_started",
  "agent_paused",
  "agent_resumed",
  "agent_exited",
  "turn_started",
  "turn_done",
  "turn_failed",
  "turn_cancelled",
  "injections_collected",
  "verb_started",
  "verb_result",
  "verb_done",
  "verb_failed",
  "output_started",
  "output_delta",
  "output_done",
  "output_cancelled",
  "user_input",
]);

export function isEnvelope(msg: unknown): msg is Envelope {
  if (typeof msg !== "object" || msg === null) return false;
  const m = msg as Partial<Envelope>;
  return (
    typeof m.t === "string" &&
    typeof m.seq === "number" &&
    typeof m.ts === "number" &&
    typeof m.content === "object" &&
    m.content !== null
  );
}

export function isHello(msg: unknown): msg is Hello {
  return (
    typeof msg === "object" &&
    msg !== null &&
    (msg as Hello).t === "hello" &&
    typeof (msg as Hello).seq === "number"
  );
}

/** 桌面是否消费该事件。 */
export function isLive(env: Envelope): boolean {
  return EVENT_TYPES.has(env.t);
}

// ------------------------------------------------------------------
// content 读取：字段缺失一律当 null，不抛
// ------------------------------------------------------------------

export function str(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

export function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

/** 事件的 agent 归属；没有就是 null（网关回声、模块生命周期）。 */
export function agentIdOf(env: Envelope): string | null {
  return str(env.content.agent_id);
}

export function textOf(env: Envelope): string {
  return str(env.content.text) ?? "";
}

export function verbOf(env: Envelope): string {
  return str(env.content.verb) ?? "unknown";
}

export function errorOf(
  env: Envelope,
): { type: string; message: string } | null {
  const e = obj(env.content.error);
  if (!e) return null;
  return {
    type: str(e.type) ?? "error",
    message: str(e.message) ?? "",
  };
}
