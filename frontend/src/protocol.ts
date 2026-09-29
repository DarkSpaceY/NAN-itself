// 协议 v2 — 后端事件信封与前端入站消息的 TS 契约。
// 权威来源：backend/nan_itself/events.py（信封）、agent/{core,engine,verbs}.py
// 与 app.py（content 字段）、gateway.py（hello / input / ping-pong）。
//
// 规则：后端只发结构化数据，所有渲染由前端完成；本文件只描述传输层
// 契约，不做任何解释（解释在 mapper / store 中进行）。

// ------------------------------------------------------------------
// 事件信封
// ------------------------------------------------------------------

/** 事件类型（信封 t 字段）。surface.* 为阶段4预留，回放时忽略。 */
export type EventType =
  | "user_input"
  | "status"
  | "output_started"
  | "output_delta"
  | "output_done"
  | "output_cancelled"
  | "record_started"
  | "record_detail"
  | "record_done"
  | "record_failed"
  | "record_void"
  | `surface.${string}`;

/** 每个事件的公共头部。 */
export interface EnvelopeBase {
  seq: number;
  ts: number; // unix seconds
  t: EventType;
  /** 同 id 事件配对（output_* 与 record_* 家族）；其余无 id。 */
  id?: string;
  /** 客户端提交的 user_input 事件携带客户端回执 id。 */
  boot_id?: string;
  content: Record<string, unknown>;
}

export type Envelope = EnvelopeBase;

// ------------------------------------------------------------------
// content 载荷（按事件类型）
// ------------------------------------------------------------------

/** user_input.content */
export interface UserInputContent {
  text: string;
  mid?: string;
}

/** status.content */
export interface StatusContent {
  state: "working" | "idle";
  agent_hash?: string;
  parent_hash?: string;
  depth?: number;
}

/** output_* 家族公共字段（来自发射点的 agent 身份）。 */
interface AgentIdentity {
  agent_hash?: string;
  parent_hash?: string;
  depth?: number;
}

export interface OutputStartedContent extends AgentIdentity {}

export interface OutputDeltaContent extends AgentIdentity {
  text: string;
}

export interface OutputDoneContent extends AgentIdentity {
  duration_s: number;
}

export interface OutputCancelledContent extends AgentIdentity {}

/** record_started.content 的 category（core._RECORD_CATEGORY + verbs）。 */
export type RecordCategory =
  | "tool_call"
  | "skill_invoke"
  | "channel_write"
  | "sleep"
  | "finish"
  | "subagent_spawn"
  | "subagent_report"
  | "module_query"
  | "unknown";

/**
 * record_started 的 payload 按 category 变形；前端按 category 解释，
 * 未识别的 category 按 unknown 兜底（不崩溃）。
 */
export interface RecordStartedContent extends AgentIdentity {
  category: RecordCategory | string;
  payload: Record<string, unknown>;
}

/** record_detail.content.entries 元素（core._result_entries 的两种 kind）。 */
export interface DetailEntry {
  kind: "field" | "text";
  label?: string;
  value?: unknown;
  text?: string;
}

export interface RecordDetailContent extends AgentIdentity {
  entries: DetailEntry[];
}

export interface RecordDoneContent extends AgentIdentity {
  duration_s?: number;
}

export interface RecordFailedContent extends AgentIdentity {
  error: { type: string; message: string };
}

export interface RecordVoidContent extends AgentIdentity {}

// ------------------------------------------------------------------
// hello（连接握手，非 bus 事件，无 ts/信封约束）
// ------------------------------------------------------------------

export interface HelloMessage {
  t: "hello";
  seq: number;
  content: {
    boot?: string;
    status?: { state: string };
    [key: string]: unknown;
  };
}

/** pong 心跳应答。 */
export interface PongMessage {
  t: "pong";
  content: Record<string, unknown>;
}

/** 服务端下行的所有消息形态。 */
export type ServerMessage = (Envelope & { t: EventType }) | HelloMessage | PongMessage;

// ------------------------------------------------------------------
// 前端入站消息
// ------------------------------------------------------------------

export interface InputMessage {
  t: "input";
  text: string;
  /** 回执去重 id；断线重连补发时网关按 mid 保证恰好一次。 */
  mid?: string;
}

export interface PingMessage {
  t: "ping";
}

export type ClientMessage = InputMessage | PingMessage;

// ------------------------------------------------------------------
// 类型守卫
// ------------------------------------------------------------------

const KNOWN_EVENT_TYPES = new Set([
  "user_input",
  "status",
  "output_started",
  "output_delta",
  "output_done",
  "output_cancelled",
  "record_started",
  "record_detail",
  "record_done",
  "record_failed",
  "record_void",
]);

/** 是否为 bus 事件信封（区别于 hello / pong）。 */
export function isEnvelope(msg: unknown): msg is Envelope {
  return (
    typeof msg === "object" &&
    msg !== null &&
    typeof (msg as Envelope).seq === "number" &&
    typeof (msg as Envelope).ts === "number" &&
    typeof (msg as Envelope).t === "string" &&
    typeof (msg as Envelope).content === "object" &&
    (msg as Envelope).content !== null
  );
}

/**
 * 事件分类：
 * - "live"：已知事件类型，进入 reducer
 * - "surface"：surface.* 预留事件，回放/直播均忽略（阶段4再消费）
 * - "unknown"：未知类型，保守忽略
 */
export function classifyEvent(env: Envelope): "live" | "surface" | "unknown" {
  if (env.t.startsWith("surface.")) return "surface";
  return KNOWN_EVENT_TYPES.has(env.t) ? "live" : "unknown";
}

// ------------------------------------------------------------------
// 工具
// ------------------------------------------------------------------

/** 客户端回执 id（输入去重用）。 */
export function newMid(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `mid-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}
