// WebSocket 连接层：
// - 连接即 hello + 历史重放（bus.history），之后进入直播转发
// - 断线指数退避重连（1s 起，10s 封顶），重连后重置本地模型由重放重建
// - 输入补发带 mid，后端网关按 mid 保证恰好一次
// - 30s 心跳 ping（后端回 pong），借 send 失败感知死连接

import {
  isEnvelope,
  newMid,
  type ClientMessage,
  type Envelope,
  type ServerMessage,
} from "./protocol";
import type { Store } from "./store";

/** 最小 socket 接口（测试可注入 mock）。 */
export interface MinimalSocket {
  readyState: number;
  send(data: string): void;
  close(): void;
  addEventListener(type: string, fn: (ev: MessageEvent | Event) => void): void;
}

export type SocketFactory = (url: string) => MinimalSocket;

export interface WsClientOptions {
  url: string;
  store: Store;
  /** 每个事件信封的旁路观察点（mapper 等并行消费者用）。 */
  onEnvelope?: (env: Envelope) => void;
  /** 握手：新 boot 的标识（缺省 null）。mapper 用它决定重放游标。 */
  onHello?: (bootId: string | null) => void;
  /** 重连初始退避 ms，默认 1000 */
  backoffBaseMs?: number;
  /** 重连退避封顶 ms，默认 10000 */
  backoffMaxMs?: number;
  /** 测试注入用：替换 WebSocket 构造器 */
  socketFactory?: SocketFactory;
}

export class WsClient {
  private readonly url: string;
  private readonly store: Store;
  private readonly onEnvelope: ((env: Envelope) => void) | null;
  private readonly onHello: ((bootId: string | null) => void) | null;
  private readonly backoffBaseMs: number;
  private readonly backoffMaxMs: number;
  private readonly makeSocket: SocketFactory;

  private socket: MinimalSocket | null = null;
  private attempt = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private closedByUser = false;

  constructor(opts: WsClientOptions) {
    this.url = opts.url;
    this.store = opts.store;
    this.onEnvelope = opts.onEnvelope ?? null;
    this.onHello = opts.onHello ?? null;
    this.backoffBaseMs = opts.backoffBaseMs ?? 1000;
    this.backoffMaxMs = opts.backoffMaxMs ?? 10000;
    this.makeSocket = opts.socketFactory ?? ((u) => new WebSocket(u));
  }

  // -- 生命周期 ------------------------------------------------------

  connect(): void {
    this.closedByUser = false;
    this.dial();
  }

  close(): void {
    this.closedByUser = true;
    this.clearTimers();
    this.socket?.close();
    this.socket = null;
    this.store.setConnection("closed");
  }

  /**
   * 提交用户输入；mid 由本地生成，后端按 mid 去重。
   *
   * 暂停态下先发 resume：暂停时发消息即自动恢复（消息进 inbox，
   * 下一轮模型看到）。这条策略属于交互约定，放在客户端；后端不
   * 关心暂停与输入的先后关系。
   */
  sendInput(text: string): string {
    if (this.store.getState().status === "paused") {
      this.send({ t: "resume" });
    }

    const mid = newMid();
    this.send({ t: "input", text, mid });
    return mid;
  }

  /** 请求暂停自主循环（协作式：后端在轮边界生效）。 */
  pause(): void {
    this.send({ t: "pause" });
  }

  /** 请求恢复自主循环。 */
  resume(): void {
    this.send({ t: "resume" });
  }

  // -- 内部 ----------------------------------------------------------

  private dial(): void {
    if (this.closedByUser) return;
    this.store.setConnection("connecting");
    const socket = this.makeSocket(this.url);
    this.socket = socket;

    socket.addEventListener("open", () => {
      // 只处理最新一次拨号的事件（旧 socket 的迟到事件直接忽略）
      if (this.socket !== socket) return;
      this.attempt = 0;
      this.store.setConnection("open");
      this.startHeartbeat();
    });

    socket.addEventListener("message", (ev) => {
      if (this.socket !== socket) return;
      let msg: unknown;
      try {
        msg = JSON.parse(String((ev as MessageEvent).data));
      } catch {
        return;
      }
      if (
        msg &&
        typeof msg === "object" &&
        (msg as ServerMessage).t === "hello"
      ) {
        // 新连接：重置本地模型，随后重放重建
        this.store.resetForReplay();
        const hello = msg as Extract<ServerMessage, { t: "hello" }>;
        this.store.applyHello(hello as never);
        // 必须在重放事件到达之前定下游标
        this.onHello?.(hello.content?.boot ?? null);
        return;
      }
      this.store.applyMessage(msg);
      if (this.onEnvelope && isEnvelope(msg)) this.onEnvelope(msg);
    });

    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.stopHeartbeat();
      if (this.closedByUser) {
        this.store.setConnection("closed");
        return;
      }
      this.scheduleReconnect();
    });

    // error 事件后必然紧跟 close，交给 close 分支处理
  }

  private scheduleReconnect(): void {
    const delay = Math.min(
      this.backoffBaseMs * 2 ** this.attempt,
      this.backoffMaxMs,
    );
    this.attempt += 1;
    this.store.setConnection("connecting");
    this.retryTimer = setTimeout(() => this.dial(), delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.send({ t: "ping" });
    }, 30_000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private clearTimers(): void {
    this.stopHeartbeat();
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  private send(msg: ClientMessage): void {
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    try {
      socket.send(JSON.stringify(msg));
    } catch {
      // 发送失败交给 close → 重连路径处理
    }
  }
}
