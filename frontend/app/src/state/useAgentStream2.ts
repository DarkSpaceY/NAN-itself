import { useEffect, useRef, useState } from 'react';
import type { ClientEvent, ServerEvent } from '../protocol';
import { fold, initialSnapshot, type Snapshot } from './store';
import { createMockDriver, type Driver } from '../mock/driver';

// 真实网关:ws://host/ws(vite proxy → 127.0.0.1:8765)
//
// 传输健壮性(连接层全权拥有 boot/seq 状态):
//   - 心跳检活:15s ping,4s 内无 pong 判死 → 重连
//   - 发送回执:每条 input 携带客户端 mid;回执按 mid(兼容按文本)
//     匹配。回显超时只撤销 pending,**不再踢线**——检活由心跳负责;
//     踢线会把"回显慢"误判成"消息丢失",重发造成同一条消息
//     重复投递(回合繁忙时必然发生)。
//   - 真正断线(onclose)时若 pending 仍在 → 重连后以同一 mid 补发,
//     服务端按 mid 去重,保证恰好一次。
//   - 历史重放按 seq 去重;hello.boot 变化 → 重置基线并通知上层清流
export const UI_VERSION = '2026-08-28.r5';

const PING_INTERVAL_MS = 15000;
const PONG_TIMEOUT_MS = 4000;
// 回执看门狗只撤销 pending(防重发),不再断连;检活由心跳负责。
const ACK_TIMEOUT_MS = 5000;

export type ConnState = 'connected' | 'reconnecting';

interface WsCallbacks {
  onEvent: (e: ServerEvent) => void;
  onReset: () => void;
  onConn: (state: ConnState) => void;
}

// 回执上下文:mid 用于匹配/去重,text 兼容旧网关(无 mid 时按文本)
interface PendingInput {
  mid: string;
  text: string;
}

let midCounter = 0;
const nextMid = () => `m${Date.now().toString(36)}-${(++midCounter).toString(36)}`;

function createWsDriver(url: string, cb: WsCallbacks): Driver {
  let ws: WebSocket | null = null;
  let alive = true;
  let lastSeq = 0;
  let boot: string | null = null;
  let pingTimer: ReturnType<typeof setInterval> | null = null;
  let pongTimer: ReturnType<typeof setTimeout> | null = null;
  let ackTimer: ReturnType<typeof setTimeout> | null = null;
  let pendingAck: PendingInput | null = null;
  let gotPong = false;

  const stopTimers = () => {
    if (pingTimer) clearInterval(pingTimer);
    if (pongTimer) clearTimeout(pongTimer);
    if (ackTimer) clearTimeout(ackTimer);
    pingTimer = pongTimer = ackTimer = null;
  };

  const connect = () => {
    if (!alive) return;

    ws = new WebSocket(url);

    ws.onopen = () => {
      cb.onConn('connected');
      startHeartbeat();

      if (pendingAck) {
        const p = pendingAck;
        pendingAck = null;
        try {
          ws?.send(JSON.stringify({ t: 'input', text: p.text, mid: p.mid } satisfies ClientEvent));
        } catch {
          /* 下一次 onclose 重连会再试 */
        }
      }
    };

    ws.onmessage = (m) => {
      let evt: ServerEvent;
      try {
        evt = JSON.parse(m.data) as ServerEvent;
      } catch {
        return;
      }

      const raw = evt as { t?: string; boot?: string; seq?: number; text?: string; mid?: string };

      if (raw.t === 'pong') {
        gotPong = true;
        if (pongTimer) clearTimeout(pongTimer);
        return;
      }

      if (raw.t === 'hello' && raw.boot) {
        if (boot && boot !== raw.boot) {
          // NAN 进程更换:基线清零 + 上层清流
          boot = raw.boot;
          lastSeq = 0;
          cb.onReset();
        } else if (!boot) {
          boot = raw.boot;
        }
        // hello 是基线握手,永远放行:新进程首个 hello 常携带
        // seq=0,无新事件的重连则 seq == lastSeq,按 seq 去重会
        // 把它吞掉,导致 model/status 永远停在 connecting…。
        cb.onEvent(evt);
        return;
      }

      if (raw.t === 'user_input' && pendingAck && (raw.mid === pendingAck.mid || raw.text === pendingAck.text)) {
        pendingAck = null;
        if (ackTimer) clearTimeout(ackTimer);
      }

      if (typeof raw.seq === 'number') {
        if (raw.seq <= lastSeq) return; // 重放去重
        lastSeq = raw.seq;
      }

      cb.onEvent(evt);
    };

    ws.onclose = () => {
      stopTimers();
      cb.onConn('reconnecting');
      if (alive) setTimeout(connect, 1200);
    };

    ws.onerror = () => {
      try {
        ws?.close();
      } catch {
        /* reconnect handles */
      }
    };
  };

  const startHeartbeat = () => {
    if (pingTimer) clearInterval(pingTimer);
    pingTimer = setInterval(() => {
      gotPong = false;
      try {
        ws?.send(JSON.stringify({ t: 'ping' } satisfies ClientEvent));
      } catch {
        return;
      }
      pongTimer = setTimeout(() => {
        if (!gotPong) {
          try {
            ws?.close(); // onclose → 重连
          } catch {
            /* noop */
          }
        }
      }, PONG_TIMEOUT_MS);
    }, PING_INTERVAL_MS);
  };

  return {
    start() {
      connect();
    },
    send(text) {
      const mid = nextMid();
      let ok = false;

      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ t: 'input', text, mid } satisfies ClientEvent));
          ok = true;
        } catch {
          ok = false;
        }
      }

      pendingAck = { mid, text };

      if (ok) {
        // 回执看门狗:只撤销 pending,绝不踢线。
        // 踢线会把"回显慢"(回合繁忙时必然发生)误判成"消息丢失",
        // 触发重发 → 同一条消息重复投递。检活是心跳(ping/pong)的职责。
        if (ackTimer) clearTimeout(ackTimer);
        ackTimer = setTimeout(() => {
          pendingAck = null;
        }, ACK_TIMEOUT_MS);
      } else {
        // 未连接:挂起,重连后以同一 mid 补发(服务端去重保证恰好一次)
      }
    },
    stop() {
      alive = false;
      stopTimers();
      ws?.close();
    },
  };
}

export function useAgentStream(): {
  snap: Snapshot;
  send: (text: string) => void;
  conn: ConnState;
} {
  const [snap, setSnap] = useState<Snapshot>(initialSnapshot);
  const [conn, setConn] = useState<ConnState>('connected');
  const driverRef = useRef<Driver | null>(null);

  useEffect(() => {
    const mode = new URLSearchParams(location.search).get('ws') !== null ? 'ws' : 'mock';

    const driver =
      mode === 'ws'
        ? createWsDriver(
            `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`,
            {
              onEvent: (e) => setSnap((s) => fold(s, e)),
              onReset: () => setSnap({ ...initialSnapshot, items: [] }),
              onConn: setConn,
            },
          )
        : createMockDriver();

    driverRef.current = driver;
    driver.start((e) => setSnap((s) => fold(s, e)));

    return () => driver.stop();
  }, []);

  const send = (text: string) => {
    const t = text.trim();
    if (!t) return;
    driverRef.current?.send(t);
  };

  return { snap, send, conn };
}
