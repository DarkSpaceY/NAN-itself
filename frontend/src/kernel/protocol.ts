// Event contract: core process -> desktop, the single source of truth.
//
// The envelope is stamped uniformly by the sink (t / id / seq / ts /
// boot_id); every business field lives in content, dispatched by t.
// Identity fields are agent_id / parent_id / depth, written into
// content at the emission point -- see nan/agent/core.py and engine.py.
//
// The frontend is a renderer for this stream: it never infers,
// replays or caches turns. On reconnect the gateway replays the ring
// buffer after hello; seq is only for deduplication.

export interface Envelope {
  t: string;
  /** Correlation id: output_* events pair by id; others carry it too, ignorable. */
  id?: string;
  /** In-process monotonic sequence, used across reconnects for dedup and gap detection. */
  seq: number;
  /** unix seconds. */
  ts: number;
  /** Present only when the sink carries a boot id. */
  boot_id?: string;
  content: Record<string, unknown>;
}

/** Server downstream: events, handshake, heartbeat replies. */
export interface Hello {
  t: "hello";
  seq: number;
  content: { boot?: string; model?: string; paused?: boolean };
}

export interface Pong {
  t: "pong";
}

/** Frontend upstream. */
export type ClientMessage =
  | { t: "input"; text: string; mid?: string }
  | { t: "pause" }
  | { t: "resume" }
  | { t: "ping" };

// ------------------------------------------------------------------
// content payloads
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

/** One verb invocation: name + structured arguments. */
export interface VerbStarted extends AgentIdentity {
  verb: string;
  arguments: Record<string, unknown>;
}

/** The verbatim text the verb returned to the model; detail is a structured summary for the UI. */
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

export interface UserInput {
  text: string;
  mid?: string;
}

export interface AgentExited extends AgentIdentity {
  reason: string;
  report?: string | null;
}

/**
 * Surface registration (`surface_added`). Not yet implemented
 * server-side: `nan/injection/hub.py` registers context services,
 * not windows. The desktop reads the same shape from the mock
 * stream for now.
 */
export interface SurfaceSpec {
  service_id: string;
  url: string;
  title: string;
  icon?: string;
}

// ------------------------------------------------------------------
// Known type table
// ------------------------------------------------------------------

/**
 * Event types the desktop consumes. `surface.*` is a reserved
 * namespace for window registration, not consumed yet; any other
 * unknown type is ignored outright (later-added events must not
 * crash an older desktop).
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

/** Whether the desktop consumes this event. */
export function isLive(env: Envelope): boolean {
  return EVENT_TYPES.has(env.t);
}

// ------------------------------------------------------------------
// content readers: missing fields are null, never thrown
// ------------------------------------------------------------------

export function str(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

export function obj(v: unknown): Record<string, unknown> | null {
  return typeof v === "object" && v !== null && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

/** The agent this event belongs to; null when there is none (gateway echo, module lifecycle). */
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
