import type { ServerEvent, Status } from '../protocol';

// 流内条目 = 原语的运行时形态
// key: 客户端自分配的唯一标识(React key 用它)。服务端 id 只保证
//      单回合内唯一,不可直接作 React key——跨回合重复会引发
//      复用错位/历史行被改写。
export type Item =
  | { k: 'divider'; id: string; key: string; label: string }
  | { k: 'user'; id: string; key: string; text: string; queued?: boolean }
  | { k: 'nano'; id: string; key: string; text: string; ts?: number; duration?: string }
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
  // 前端自行派生 divider 的依据:上一个事件所属的本地日期标签
  lastDate: string | null;
}

export const initialSnapshot: Snapshot = {
  items: [],
  status: { state: 'idle' },
  model: 'connecting…',
  baseUrl: '',
  seq: 0,
  lastDate: null,
};

let counter = 0;
export const nextId = () => `c${++counter}`;

// 渲染/折叠边缘才把数值时间戳变成日期标签(协议契约:传输层一律 epoch,
// 后端不产生 divider 事件,前端按事件 ts 自行插入)
const dateLabel = (ts: number) =>
  new Date(ts * 1000).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

export function fold(s: Snapshot, e: ServerEvent): Snapshot {
  // 日期分隔:任何带时间戳的事件跨天时先插一条 divider
  const ts = (e as { ts?: number }).ts;

  if (ts != null) {
    const label = dateLabel(ts);

    if (label !== s.lastDate) {
      s = {
        ...s,
        lastDate: label,
        items: [...s.items, { k: 'divider', id: label, key: nextId(), label }],
      };
    }
  }

  switch (e.t) {
    case 'hello':
      return { ...s, seq: e.seq, model: e.model, baseUrl: e.base_url ?? s.baseUrl, status: e.status ?? s.status };

    case 'status':
      return { ...s, status: { state: e.state } };

    case 'user_input':
      return { ...s, items: [...s.items, { k: 'user', id: e.id, key: nextId(), text: e.text }] };

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
