// WebSocket 客户端：唯一的上行通道。
//
// 连接建立后网关先发 hello，再重放环形历史；之后是实时事件。桌面
// 只做三件事：hello 换 boot 时重建历史、事件交给 hub、断线重连。
//
// 套接字生命周期与 UI 无关：这里没有 React、没有状态存储。

import {
  isEnvelope,
  isHello,
  str,
  type ClientMessage,
  type Envelope,
  type SurfaceSpec,
} from "../kernel/protocol";
import type { Hub } from "../kernel/history/hub";

export type SurfaceHandler = (spec: SurfaceSpec) => void;

/** 桌面消费的传输接口；mock 提供同一形状。 */
export interface Transport {
  /** 状态行左侧的传输标签（ws / mock）。 */
  readonly label: string;
  start(onSurface: SurfaceHandler): void;
  stop(): void;
  send(msg: ClientMessage): void;
}

/** 心跳间隔：服务端只回 pong，不主动发；断线由 onclose 发现。 */
const PING_MS = 20_000;
const MAX_BACKOFF_MS = 8_000;

/** 同源 /ws；可用 ?ws=ws://host:port/ws 覆盖（dev 或反代）。 */
export function defaultWsUrl(): string {
  const override = new URLSearchParams(location.search).get("ws");
  if (override) return override;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function parseSurface(content: Record<string, unknown>): SurfaceSpec | null {
  const serviceId = str(content.service_id);
  const url = str(content.url);
  const title = str(content.title);
  if (!serviceId || !url || !title) return null;
  const icon = str(content.icon);
  return icon
    ? { service_id: serviceId, url, title, icon }
    : { service_id: serviceId, url, title };
}

export class WsTransport implements Transport {
  readonly label = "ws";

  private socket: WebSocket | null = null;
  private pingTimer: number | null = null;
  private retryTimer: number | null = null;
  private attempt = 0;
  private stopped = false;
  private onSurface: SurfaceHandler = () => {};

  constructor(
    private readonly hub: Hub,
    private readonly url: string = defaultWsUrl(),
  ) {}

  start(onSurface: SurfaceHandler): void {
    this.onSurface = onSurface;
    this.stopped = false;
    this.open();
  }

  stop(): void {
    this.stopped = true;
    this.clearTimers();
    this.socket?.close();
    this.socket = null;
    this.hub.global.setConnection("closed");
  }

  send(msg: ClientMessage): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(msg));
    }
  }

  // -- 连接 ----------------------------------------------------------

  private open(): void {
    if (this.stopped) return;
    this.hub.global.setConnection("connecting");

    let socket: WebSocket;
    try {
      socket = new WebSocket(this.url);
    } catch {
      this.scheduleRetry();
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      this.attempt = 0;
      this.hub.global.setConnection("open");
      this.startPing();
    };

    socket.onmessage = (event: MessageEvent) => {
      if (typeof event.data === "string") this.handle(event.data);
    };

    socket.onclose = () => {
      if (this.socket === socket) this.socket = null;
      this.clearTimers();
      this.hub.global.setConnection("closed");
      this.scheduleRetry();
    };

    // onclose 一定会跟着触发，这里只负责触发它。
    socket.onerror = () => socket.close();
  }

  private scheduleRetry(): void {
    if (this.stopped || this.retryTimer !== null) return;
    const delay = Math.min(500 * 2 ** this.attempt, MAX_BACKOFF_MS);
    this.attempt += 1;
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      this.open();
    }, delay);
  }

  private startPing(): void {
    this.stopPing();
    this.pingTimer = window.setInterval(
      () => this.send({ t: "ping" }),
      PING_MS,
    );
  }

  private stopPing(): void {
    if (this.pingTimer !== null) {
      window.clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private clearTimers(): void {
    this.stopPing();
    if (this.retryTimer !== null) {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  // -- 解析 ----------------------------------------------------------

  private handle(raw: string): void {
    let msg: unknown;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }

    if (isHello(msg)) {
      const previousBoot = this.hub.global.getState().bootId;
      const nextBoot =
        typeof msg.content?.boot === "string" ? msg.content.boot : null;
      this.hub.global.applyHello({
        seq: msg.seq,
        content: {
          boot: nextBoot ?? undefined,
          model:
            typeof msg.content?.model === "string"
              ? msg.content.model
              : undefined,
          paused:
            typeof msg.content?.paused === "boolean"
              ? msg.content.paused
              : undefined,
        },
      });
      // 后端重启过：历史不可拼接，重建树与时间线
      if (nextBoot && nextBoot !== previousBoot) this.hub.reset();
      return;
    }

    if (!isEnvelope(msg)) return;

    // 窗口注册不是历史，不进 hub
    if (msg.t === "surface_added") {
      const spec = parseSurface(msg.content as Record<string, unknown>);
      if (spec) this.onSurface(spec);
      return;
    }

    this.hub.ingest(msg as Envelope);
  }
}
