import type { ServerEvent } from '../protocol';

// mock 驱动:与真实网关走同一事件协议,开发期替代 NAN 后端。
export interface Driver {
  start(onEvent: (e: ServerEvent) => void): void;
  send(text: string): void;
  stop(): void;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function createMockDriver(): Driver {
  let demoSeq = 0;
  let onEvent: ((e: ServerEvent) => void) | null = null;
  let timers: ReturnType<typeof setTimeout>[] = [];
  let alive = true;

  const later = (ms: number, fn: () => void) => {
    if (!alive) return;
    timers.push(setTimeout(fn, ms));
  };
  const emit = (e: ServerEvent) => {
    if (alive) onEvent?.(e);
  };
  const idOf = (() => {
    let i = 0;
    return () => `m${++i}`;
  })();

  const clearAll = () => {
    alive = false;
    timers.forEach(clearTimeout);
    timers = [];
  };

  async function runDemoTurn() {
    emit({ t: 'user_input', id: `u${++demoSeq}`, text: '帮我看看最近 workspace 改了哪些文件,顺便确认我的时区偏好还在记忆里。' });
    let id = idOf();
    emit({ t: 'record_started', id, kind: 'tool', name: 'workspace.scan()' });
    later(450, () => {
      emit({ t: 'record_detail', id, line: '103 files · changed 3 · deleted 0' });
      ['src/nan_itself/app.py', 'workspace/persona.md', 'docs/memory-design.md'].forEach((f, i) =>
        later(240 * (i + 1), () => emit({ t: 'record_detail', id, line: `· ${f}` })),
      );
    });
    await sleep(1500);
    emit({ t: 'record_done', id, summary: '103 files · +3 changed', note: '0.4s' });

    await sleep(300);
    id = idOf();
    emit({ t: 'record_started', id, kind: 'agent', name: 'dispatch_subagent', summary: 'memory-review' });
    emit({
      t: 'record_detail',
      id,
      line: '简报 · 核对记忆中「UTC+8」「极简回答」两条的有效性,无冲突即闭合。',
    });

    await sleep(400);
    id = idOf();
    emit({ t: 'record_started', id, kind: 'tool', name: 'files.read', summary: 'docs/memory-design.md' });
    await sleep(1200);
    emit({ t: 'record_detail', id, line: '✗ 超时 90s · 本地模型未响应' });
    emit({ t: 'record_failed', id });
    await sleep(600);

    id = idOf();
    emit({ t: 'record_started', id, kind: 'tool', name: 'files.read', summary: '重试' });
    await sleep(800);
    emit({ t: 'record_detail', id, line: 'ok · 9.8k chars · 设计文档 v1,待评审状态' });
    emit({ t: 'record_done', id, summary: '9.8k chars', note: '1.2s' });

    await sleep(400);
    id = idOf();
    emit({ t: 'record_started', id, kind: 'agent', name: 'report', summary: 'memory-review' });
    emit({ t: 'record_detail', id, line: 'm-000001 / m-000002 均有效,无冲突,未新增。' });
    await sleep(800);
    emit({ t: 'record_done', id, summary: '已消化', note: '1m12s' });

    await sleep(200);
    id = idOf();
    emit({ t: 'output_started', id });
    const text =
      '最近改动 3 个文件:app.py、persona.md、memory-design.md;其中设计文档读了一次超时,重试成功。\n记忆确认:时区 UTC+8、偏好极简回答,两条均有效,未新增。';
    for (let i = 0; i < text.length; i += 2) {
      emit({ t: 'output_delta', id, text: text.slice(i, i + 2) });
      await sleep(24);
    }
    await sleep(150);
    emit({ t: 'output_done', id, ts: '22:41:40', duration: '0.9s' });

    // 循环:sleep 是普通 verb
    await sleep(300);
    id = idOf();
    emit({ t: 'record_started', id, kind: 'verb', name: 'sleep', summary: '15s' });
    await sleep(700);
    emit({ t: 'record_done', id, summary: '下一跳 22:42:13' });
    emit({ t: 'status', state: 'idle', next_hop: '22:42:13' });
  }

  async function runAckTurn(text: string) {
    emit({ t: 'user_input', id: `u${++demoSeq}`, text });
    emit({ t: 'status', state: 'working', tools: 1 });
    let id = idOf();
    emit({ t: 'record_started', id, kind: 'verb', name: 'echo', summary: 'mock' });
    await sleep(500);
    emit({ t: 'record_done', id, summary: '已入队' });
    id = idOf();
    emit({ t: 'output_started', id });
    const reply = `mock 模式收到:「${text.slice(0, 24)}${text.length > 24 ? '…' : ''}」。接上 WS 网关后,这里就是真实的 NAN。`;
    for (let i = 0; i < reply.length; i += 2) {
      emit({ t: 'output_delta', id, text: reply.slice(i, i + 2) });
      await sleep(18);
    }
    emit({ t: 'output_done', id, ts: new Date().toTimeString().slice(0, 8), duration: '0.3s' });
    emit({ t: 'status', state: 'idle' });
  }

  return {
    start(evt) {
      onEvent = evt;
      emit({
        t: 'hello',
        seq: 42,
        model: 'local-model',
        base_url: '127.0.0.1:11434',
        status: { state: 'working', tools: 2, subagents: 1 },
      });
      emit({ t: 'divider', label: '今天 · 8月27日' });
      later(300, () => {
        void runDemoTurn();
      });
    },
    send(text) {
      void runAckTurn(text);
    },
    stop: clearAll,
  };
}