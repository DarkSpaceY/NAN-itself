// Mirror of frontend/PROTOCOL.md v2 — the backend sends typed data, the frontend renders.
export type StatusState = 'idle' | 'working' | 'error';

export interface Status {
  state: StatusState;
}

// Envelope: transport-layer fields stamped by the EventBus / sink. seq/ts are
// filled by the EventBus, id by the sink; boot_id is recorded at attach and the
// key is absent when unset.
export interface Envelope {
  seq?: number;
  ts?: number;
  id?: string;
  boot_id?: string;
}

// Identity keys: carried inside `content` by the emitting side; absent when the
// event has no agent context (the keys do not exist, they are not null).
export interface Identity {
  agent_hash?: string;
  parent_hash?: string | null;
  depth?: number;
}

// Payload event shell: `t` is the single discriminant, `id` is always present
// (the UI row key).
interface Payload<T extends string, C> extends Envelope {
  t: T;
  id: string;
  content: C & Identity;
}

// --- record categories -----------------------------------------------------

export type RecordCategory =
  | 'tool_call'
  | 'skill_invoke'
  | 'channel_write'
  | 'subagent_spawn'
  | 'subagent_report'
  | 'module_query'
  | 'sleep'
  | 'finish'
  | 'unknown';

// The `payload` union is discriminated on `category`: narrowing `category`
// narrows `payload` with it.
export type RecordStarted =
  | { category: 'tool_call';       payload: { provider: string; tool: string; arguments: Record<string, unknown> } }
  | { category: 'skill_invoke';    payload: { skill: string; resource?: string } }
  | { category: 'channel_write';   payload: { module: string; channel: string; payload?: unknown } }
  | { category: 'subagent_spawn';  payload: { agent_id: string; depth: number; task: string } }
  | { category: 'subagent_report'; payload: { agent_id: string; task: string; status: string; body: string } }
  | { category: 'module_query';    payload: Record<string, never> }
  | { category: 'sleep';           payload: { seconds: number } }
  | { category: 'finish';          payload: Record<string, never> }
  | { category: 'unknown';         payload: { verb: string } };

// --- detail entries --------------------------------------------------------

export type RecordEntry =
  | { kind: 'text';  text: string }
  | { kind: 'item';  text: string }
  | { kind: 'field'; label: string; value: string }
  | { kind: 'code';  text: string };

// --- events ----------------------------------------------------------------

// hello is the baseline handshake: it does not go through the sink (no `id`),
// and `seq` is mandatory.
export interface HelloEvent extends Envelope {
  t: 'hello';
  seq: number;
  content: {
    boot: string;
    model: string;
    base_url: string;
    status: Status;
  } & Identity;
}

export type ServerEvent =
  | HelloEvent
  | Payload<'status', { state: StatusState }>
  | Payload<'user_input', { text: string; mid?: string }>
  | Payload<'record_started', RecordStarted>
  | Payload<'record_detail', { entries: RecordEntry[] }>
  | Payload<'record_done', { result?: Record<string, unknown>; duration_s?: number }>
  | Payload<'record_failed', { error: { type: string; message: string } }>
  | Payload<'record_void', Identity>
  | Payload<'output_started', Identity>
  | Payload<'output_delta', { text: string }>
  | Payload<'output_done', { duration: number }>
  | Payload<'output_cancelled', Identity>;

// mid: the client message id, used for ack matching and server-side dedup
// (exactly-once under reconnect re-send).
export type ClientEvent = { t: 'input'; text: string; mid?: string } | { t: 'ping' };
