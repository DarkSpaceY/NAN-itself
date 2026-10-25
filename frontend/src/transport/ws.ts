// WebSocket client: the only upstream channel.
//
// After connecting, the gateway sends hello first, then replays the
// ring-buffered history; live events follow. The desktop does three
// things: rebuild history when hello carries a new boot, hand events
// to the hub, reconnect on disconnect.
//
// The socket lifecycle is independent of the UI: no React here, no
// state store.

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

/** A service withdrew its panel: the window disappears. */
export type SurfaceRemovedHandler = (serviceId: string) => void;

/** The transport interface the desktop consumes; the mock provides the same shape. */
export interface Transport {
  /** The transport label on the left of the status line (ws / mock). */
  readonly label: string;
  start(onSurface: SurfaceHandler, onSurfaceRemoved: SurfaceRemovedHandler): void;
  stop(): void;
  send(msg: ClientMessage): void;
}

/** Heartbeat interval: the server only replies with pong and never initiates; disconnects are discovered by onclose. */
const PING_MS = 20_000;
const MAX_BACKOFF_MS = 8_000;

/** Same-origin /ws; overridable with ?ws=ws://host:port/ws (dev or reverse proxy). */
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
  private onSurfaceRemoved: SurfaceRemovedHandler = () => {};

  constructor(
    private readonly hub: Hub,
    private readonly url: string = defaultWsUrl(),
  ) {}

  start(onSurface: SurfaceHandler, onSurfaceRemoved: SurfaceRemovedHandler): void {
    this.onSurface = onSurface;
    this.onSurfaceRemoved = onSurfaceRemoved;
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

  // -- connection ----------------------------------------------------

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

    // onclose is guaranteed to fire afterwards; this just triggers it.
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

  // -- parsing --------------------------------------------------------

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
      // The backend restarted: history cannot be stitched, rebuild tree and timeline
      if (nextBoot && nextBoot !== previousBoot) this.hub.reset();
      return;
    }

    if (!isEnvelope(msg)) return;

    // Surface registration is not history; it does not enter the hub
    if (msg.t === "surface_added") {
      const spec = parseSurface(msg.content as Record<string, unknown>);
      if (spec) this.onSurface(spec);
      return;
    }

    if (msg.t === "surface_removed") {
      const serviceId = str((msg.content as Record<string, unknown>).service_id);
      if (serviceId) this.onSurfaceRemoved(serviceId);
      return;
    }

    this.hub.ingest(msg as Envelope);
  }
}
