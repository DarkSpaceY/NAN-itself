// 与 frontend/PROTOCOL.md 一一对应的事件契约
export type StatusState = 'idle' | 'working' | 'error';

export interface Status {
  state: StatusState;
}

// 信封:EventBus/sink 统一加盖的传输层字段。seq/ts 由 EventBus 填,
// id 由 sink 生成,boot_id 在 attach 时记录——未设置则不出现该键。
export interface Envelope {
  seq?: number;
  ts?: number;
  id?: string;
  boot_id?: string;
}

// 身份三键:由发出方在 content 中携带;未携带即无 agent 上下文
// (此时这三个键不存在,不是 null)。
export interface Identity {
  agent_hash?: string;
  parent_hash?: string | null;
  depth?: number;
}

// 载荷事件的公共外壳:t 为唯一判别字段,id 恒在(UI 行关联键)。
interface Payload<T extends string, C> extends Envelope {
  t: T;
  id: string;
  content: C & Identity;
}

// hello 是基线握手:不经 sink,无 id;seq 由网关显式给出。
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
  | Payload<
      'record_started',
      {
        kind:
          | 'tool'
          | 'skill'
          | 'spawn'
          | 'sleep'
          | 'finish'
          | 'module'
          | 'agent'
          | 'target'
          | 'error';
        name: string;
        summary?: string;
      }
    >
  | Payload<'record_detail', { line: string }>
  | Payload<'record_done', { summary?: string; note?: string }>
  | Payload<'record_failed', { summary?: string }>
  | Payload<'record_void', Identity>
  | Payload<'output_started', Identity>
  | Payload<'output_delta', { text: string }>
  | Payload<'output_done', { duration: string }>
  | Payload<'output_cancelled', Identity>;

// mid: 客户端消息 id,用于回执匹配与服务端去重(防断线重发导致重复投递)
export type ClientEvent = { t: 'input'; text: string; mid?: string } | { t: 'ping' };

export const GLYPH_BY_KIND: Record<string, string> = {
  tool: '▸',
  skill: '✦',
  spawn: '⧉',
  sleep: '⏾',
  finish: '⏻',
  module: '◈',
  agent: '◈',
  target: '⌖',
  error: '✗',
};
