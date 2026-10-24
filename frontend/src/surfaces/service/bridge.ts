// Panel bridge: a one-time MessagePort handshake.
//
// Panels in the sandbox have no same-origin privileges, so there are
// only two channels: the postMessage sent after the iframe loads
// (carrying boot / service_id / theme), and the MessagePort the
// panel sets up once ready. Panel -> desktop traffic is attention
// only.

import type { SurfaceSpec } from "../../kernel/protocol";
import { themeTokens, type Theme } from "../../shell/theme";

export type AttentionHandler = (level: string) => void;

export interface Bridge {
  setTheme(theme: Theme): void;
  destroy(): void;
}

interface InitMessage {
  t: "init";
  content: {
    boot: string | null;
    service_id: string;
    theme: Theme;
    tokens: Record<string, string>;
  };
}

export function attachBridge(
  frame: HTMLIFrameElement,
  spec: SurfaceSpec,
  theme: Theme,
  bootId: string | null,
  onAttention?: AttentionHandler,
): Bridge {
  let themeNow = theme;
  let port: MessagePort | null = null;

  const init = (): InitMessage => ({
    t: "init",
    content: {
      boot: bootId,
      service_id: spec.service_id,
      theme: themeNow,
      tokens: themeTokens(themeNow),
    },
  });

  const post = () => {
    const target = frame.contentWindow;
    if (!target) return;

    if (port) port.postMessage(init());
    else target.postMessage(init(), "*");
  };

  const onLoad = () => post();

  const onMessage = (event: MessageEvent) => {
    if (event.source !== frame.contentWindow) return;
    const data = event.data as { t?: string } | null;
    if (!data || typeof data.t !== "string") return;

    if (data.t === "ready" && !port) {
      const channel = new MessageChannel();
      port = channel.port1;
      port.onmessage = (message) => {
        const payload = message.data as { t?: string; level?: unknown } | null;
        if (payload?.t === "attention" && typeof payload.level === "string") {
          onAttention?.(payload.level);
        }
      };
      frame.contentWindow?.postMessage({ t: "port" }, "*", [channel.port2]);
      post();
    }
  };

  frame.addEventListener("load", onLoad);
  window.addEventListener("message", onMessage);

  return {
    setTheme(next: Theme) {
      themeNow = next;
      if (port)
        port.postMessage({
          t: "theme",
          content: { theme: next, tokens: themeTokens(next) },
        });
      else post();
    },
    destroy() {
      frame.removeEventListener("load", onLoad);
      window.removeEventListener("message", onMessage);
      port?.close();
      port = null;
    },
  };
}
