// 与 frontend/PROTOCOL.md 一一对应的事件契约
export type StatusState = 'idle' | 'working' | 'error';

export interface Status {
  state: StatusState;
  tools?: number;
  subagents?: number;
  next_hop?: string;
}

export type ServerEvent =
  | { t: 'hello'; seq: number; model: string; base_url: string; status: Status }
  | { t: 'status'; state: StatusState; tools?: number; subagents?: number; next_hop?: string }
  | { t: 'user_input'; id: string; text: string; mid?: string }
  | { t: 'divider'; label: string }
  | { t: 'record_started'; id: string; kind: 'tool' | 'skill' | 'spawn' | 'sleep' | 'finish' | 'module' | 'agent' | 'target' | 'error'; name: string; summary?: string }
  | { t: 'record_detail'; id: string; line: string }
  | { t: 'record_done'; id: string; summary?: string; note?: string }
  | { t: 'record_failed'; id: string; summary?: string }
  | { t: 'record_void'; id: string }
  | { t: 'output_started'; id: string }
  | { t: 'output_delta'; id: string; text: string }
  | { t: 'output_done'; id: string; ts: string; duration: string }
  | { t: 'output_cancelled'; id: string };

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