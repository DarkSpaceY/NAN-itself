import type { RecordCategory, RecordEntry, RecordStarted, ServerEvent, Status } from '../protocol';

// A stream row = the runtime shape of a primitive.
// key: client-assigned unique id (used as the React key). The server `id` is
//      only unique within a turn and must not be a React key directly — a
//      reused id across turns would cause a mis-keyed remount.
export type Item =
  | { k: 'divider'; id: string; key: string; label: string }
  | { k: 'user'; id: string; key: string; text: string; queued?: boolean }
  | { k: 'nano'; id: string; key: string; text: string; ts?: number; duration?: number }
  | {
      k: 'record';
      id: string;
      key: string;
      // The typed record payload, discriminated on `category`; the renderer
      // narrows `category` to reach the category-specific fields.
      started: RecordStarted;
      glyph: string;
      label: string;
      state: 'running' | 'done' | 'failed';
      // seconds (from record_done.duration_s); formatted only at the render edge
      duration?: number;
      error?: { type: string; message: string };
      open: boolean;
      detail: RecordEntry[];
    };

export interface Snapshot {
  items: Item[];
  status: Status;
  model: string;
  baseUrl: string;
  seq: number;
  // Frontend-derived divider basis: the local date label of the last event
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

// Presentation vocabulary owned by the frontend: category -> glyph.
const GLYPH_BY_CATEGORY: Record<RecordCategory, string> = {
  tool_call: '▸',
  skill_invoke: '✦',
  channel_write: '⌖',
  subagent_spawn: '⧉',
  subagent_report: '◈',
  module_query: '◈',
  sleep: '⏾',
  finish: '⏻',
  unknown: '▸',
};

// Human label derived from the typed payload (there is no backend `name` string).
function recordLabel(s: RecordStarted): string {
  switch (s.category) {
    case 'tool_call':
      return s.payload.tool;
    case 'skill_invoke':
      return s.payload.skill;
    case 'channel_write':
      return `${s.payload.module} · ${s.payload.channel}`;
    case 'subagent_spawn':
    case 'subagent_report':
      return s.payload.agent_id;
    case 'module_query':
      return 'module';
    case 'sleep':
      return 'sleep';
    case 'finish':
      return 'finish';
    case 'unknown':
      return s.payload.verb;
  }
}

// Render/fold edge only: turn a numeric timestamp into a date label (transport
// stays epoch; the backend emits no divider event, the frontend derives them
// from each event's numeric ts).
const dateLabel = (ts: number) =>
  new Date(ts * 1000).toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

export function fold(s: Snapshot, e: ServerEvent): Snapshot {
  // Date separator: insert a divider whenever an event with a timestamp crosses
  // into a new local day.
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
      return {
        ...s,
        seq: e.seq ?? s.seq,
        model: e.content.model,
        baseUrl: e.content.base_url ?? s.baseUrl,
        status: e.content.status ?? s.status,
      };

    case 'status':
      return { ...s, status: { state: e.content.state } };

    case 'user_input':
      return { ...s, items: [...s.items, { k: 'user', id: e.id, key: nextId(), text: e.content.text }] };

    case 'record_started': {
      const started = { category: e.content.category, payload: e.content.payload } as RecordStarted;
      const item: Item = {
        k: 'record',
        id: e.id,
        key: nextId(),
        started,
        glyph: GLYPH_BY_CATEGORY[started.category],
        label: recordLabel(started),
        state: 'running',
        open: false,
        detail: [],
      };
      return { ...s, items: [...s.items, item] };
    }

    case 'record_detail':
      // Entries arrive already typed; append them as-is.
      return mapRecord(s, e.id, (r) => ({
        ...r,
        detail: [...r.detail, ...e.content.entries],
      }));

    case 'record_void': {
      // Retract only the most recent same-id record: an older turn's same-id row
      // must stay untouched.
      const idx = findLastIndex(s.items, (it) => it.id === e.id && it.k === 'record');
      if (idx === -1) return s;
      return { ...s, items: [...s.items.slice(0, idx), ...s.items.slice(idx + 1)] };
    }

    case 'record_done':
      return mapRecord(s, e.id, (r) => ({
        ...r,
        state: 'done',
        glyph: '✓',
        duration: e.content.duration_s ?? r.duration,
        open: false,
      }));

    case 'record_failed':
      return mapRecord(s, e.id, (r) => ({
        ...r,
        state: 'failed',
        glyph: '✗',
        error: e.content.error,
        open: true,
      }));

    case 'output_started':
      return { ...s, items: [...s.items, { k: 'nano', id: e.id, key: nextId(), text: '' }] };

    case 'output_delta':
      return mapNano(s, e.id, (it) => ({ ...it, text: it.text + e.content.text }));

    case 'output_cancelled': {
      const idx = findLastIndex(s.items, (it) => it.id === e.id && it.k === 'nano');
      if (idx === -1) return s;
      return { ...s, items: [...s.items.slice(0, idx), ...s.items.slice(idx + 1)] };
    }

    case 'output_done':
      return mapNano(s, e.id, (it) => ({ ...it, ts: e.ts, duration: e.content.duration }));

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

// Patch only the *most recent* same-id row: within a turn an id is unique, so
// the last match is necessarily this event's target; a repeated id across turns
// never touches a historical row.
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
