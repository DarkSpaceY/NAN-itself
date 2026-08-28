import type { ServerEvent, Status } from '../protocol';

// 流内条目 = 原语的运行时形态
// key: 客户端自分配的唯一标识(React key 用它)。服务端 id 只保证
//      单回合内唯一,不可直接作 React key——跨回合重复会引发
//      复用错位/历史行被改写。
export type Item =
  | { k: 'divider'; id: string; key: string; label: string }
  | { k: 'user'; id: string; key: string; text: string; queued?: boolean }
  | { k: 'nano'; id: string; key: string; text: string; ts?: string; duration?: string }
  | {
      k: 'record';
      id: string;
      key: string;
      kind: string;
      glyph: string;
      name: string;
      summary?: string;
      note?: string;
      state: 'running' | 'done' | 'failed';
      open: boolean;
      detail: string[];
    };

export interface Snapshot {
  items: Item[];
  status: Status;
  model: string;
  baseUrl: string;
  seq: number;
}

export const initialSnapshot: Snapshot = {
  items: [],
  status: { state: 'idle' },
  model: 'connecting…',
  baseUrl: '',
  seq: 0,
};

let counter = 0;
export const nextId = () => `c${++counter}`;

export function fold(s: Snapshot, e: ServerEvent): Snapshot {
  switch (e.t) {
    case 'hello':
      return { ...s, seq: e.seq, model: e.model, baseUrl: e.base_url ?? s.baseUrl, status: e.status ?? s.status };

    case 'status':
      return { ...s, status: { state: e.state, tools: e.tools, subagents: e.subagents, next_hop: e.next_hop } };

    case 'user_input':
      return { ...s, items: [...s.items, { k: 'user', id: e.id, key: nextId(), text: e.text }] };

    case 'divider': {
      // 同一日期只出现一次(重连重发/重放都不再重复)
      const dup = s.items.some(
        (it) => it.k === 'divider' && it.label === e.label,
      );
      if (dup) return s;

      return { ...s, items: [...s.items, { k: 'divider', id: e.label, key: nextId(), label: e.label }] };
    }

    case 'record_started': {
      const item: Item = {
        k: 'record',
        id: e.id,
        key: nextId(),
        kind: e.kind,
        glyph: e.kind === 'error' ? '✗' : e.kind === 'module' ? '◈' : e.kind === 'skill' ? '✦' : '▸',
        name: e.name,
        summary: e.summary,
        state: 'running',
        open: false,
        detail: [],
      };
      return { ...s, items: [...s.items, item] };
    }

    case 'record_detail':
      return mapRecord(s, e.id, (r) => ({
        ...r,
        detail: [...r.detail, ...e.line.split('\n')],
      }));

    case 'record_void': {
      // 只撤回最近一条同 id 记录:旧回合的同 id 行不许被动
      const idx = findLastIndex(s.items, (it) => it.id === e.id && it.k === 'record');
      if (idx === -1) return s;
      return { ...s, items: [...s.items.slice(0, idx), ...s.items.slice(idx + 1)] };
    }

    case 'record_done':
      return mapRecord(s, e.id, (r) => ({
        ...r,
        state: 'done',
        glyph: '✓',
        summary: e.summary ?? r.summary,
        note: e.note ?? r.note,
        open: false,
      }));

    case 'record_failed':
      return mapRecord(s, e.id, (r) => ({
        ...r,
        state: 'failed',
        glyph: '✗',
        summary: e.summary ?? r.summary,
        open: true,
      }));

    case 'output_started':
      return { ...s, items: [...s.items, { k: 'nano', id: e.id, key: nextId(), text: '' }] };

    case 'output_delta':
      return mapNano(s, e.id, (it) => ({ ...it, text: it.text + e.text }));

    case 'output_cancelled': {
      const idx = findLastIndex(s.items, (it) => it.id === e.id && it.k === 'nano');
      if (idx === -1) return s;
      return { ...s, items: [...s.items.slice(0, idx), ...s.items.slice(idx + 1)] };
    }

    case 'output_done':
      return mapNano(s, e.id, (it) => ({ ...it, ts: e.ts, duration: e.duration }));

    default:
      return s;
  }
}

function findLastIndex(items: Item[], pred: (it: Item) => boolean): number {
  for (let i = items.length - 1; i >= 0; i--) {
    if (pred(items[i])) return i;
  }
  return -1;
}

// 只匹配"最近一条"同 id 条目:同一回合内 id 唯一,最近的必然是
// 当前事件的目标;跨回合重复 id 时也不会误伤历史行。
function patchLast(s: Snapshot, id: string, pred: (it: Item) => boolean, f: (it: Item) => Item): Snapshot {
  const idx = findLastIndex(s.items, (it) => it.id === id && pred(it));
  if (idx === -1) return s;
  const items = s.items.slice();
  items[idx] = f(items[idx]);
  return { ...s, items };
}

function mapNano(s: Snapshot, id: string, f: (it: Extract<Item, { k: 'nano' }>) => Item): Snapshot {
  return patchLast(s, id, (it) => it.k === 'nano', f as (it: Item) => Item);
}

function mapRecord(s: Snapshot, id: string, f: (r: Extract<Item, { k: 'record' }>) => Item): Snapshot {
  return patchLast(s, id, (it) => it.k === 'record', f as (it: Item) => Item);
}
