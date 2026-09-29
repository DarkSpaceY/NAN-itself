// App：ExternalStoreRuntime 对接自持 mapper + 中央 dock。
// 布局：ThreadPrimitive.Root 包整个 dock（面板 + 把手 + 输入行）。

import { useCallback, useEffect, useMemo, useSyncExternalStore } from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  ThreadPrimitive,
  ComposerPrimitive,
} from "@assistant-ui/react";
import type { ThreadMessageLike } from "@assistant-ui/react";

import { Store, type NanState } from "../store";
import { Mapper, type HistoryRound } from "../history/mapper";
import type { RoundStore } from "../persistence/db";
import { roundsToMessages } from "../convert";
import { Stage } from "./Stage";
import { UserMessage, AssistantMessage } from "./Message";

export interface AppProps {
  store: Store;
  mapper: Mapper;
  db: RoundStore;
  sendInput: (text: string) => void;
}

export function App({ store, mapper, db, sendInput }: AppProps) {
  // mapper 轮次是原地变更的，用版本号做快照；messages 直接从
  // getRounds() 现取，不按数组引用缓存（引用恒定不会失效）。
  const roundsVersion = useSyncExternalStore(
    useCallback((cb) => mapper.subscribe(cb), [mapper]),
    () => mapper.getVersion(),
  );

  const nanState: NanState = useSyncExternalStore(
    useCallback((cb) => store.subscribe(cb), [store]),
    () => store.getState(),
  );
  const isRunning = nanState.status === "working";

  // 持久化：关闭轮立即落盘，活动轮节流
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let pending: HistoryRound | null = null;
    const unsub = mapper.subscribe((round) => {
      if (round.closed) {
        pending = null;
        void db.saveRound(round);
        return;
      }
      pending = round;
      timer ??= setTimeout(() => {
        timer = null;
        if (pending && !pending.closed) void db.saveRound(pending);
      }, 500);
    });
    return () => {
      unsub();
      if (timer) clearTimeout(timer);
    };
  }, [mapper, db]);

  const messages: ThreadMessageLike[] = useMemo(
    () => roundsToMessages(mapper.getRounds()),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [roundsVersion],
  );

  const runtime = useExternalStoreRuntime({
    messages,
    isRunning,
    convertMessage: (m) => m,
    onNew: async ({ content }) => {
      const text = content
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      if (text.trim()) sendInput(text);
    },
    onCancel: undefined, // 阶段4接 cancel
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Stage store={store} />
      <ThreadPrimitive.Root className="dock-root">
        <div id="dock-panel-wrap">
          <Handle />
          <div id="dock-panel" data-closed="true" style={{ height: 0 }}>
            <ThreadPrimitive.Viewport id="dock-history" autoScroll>
              <ThreadPrimitive.If empty>
                <div className="empty-hint">还没有对话。输入第一条消息开始。</div>
              </ThreadPrimitive.If>
              <ThreadPrimitive.Messages
                components={{ UserMessage, AssistantMessage }}
              />
            </ThreadPrimitive.Viewport>
          </div>
        </div>

        <ComposerPrimitive.Root id="dock-input-row">
          <span className="conn-dot" data-state={nanState.connection} />
          <ComposerPrimitive.Input id="dock-input" rows={1} autoFocus />
          <ComposerPrimitive.Send id="dock-send">↑</ComposerPrimitive.Send>
        </ComposerPrimitive.Root>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

/** 细线把手：拖拽调面板高度，轻点开关。 */
function Handle() {
  // 保留与 v1 相同的交互语义；React 实现直接操作 panel 高度
  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const panel = e.currentTarget.parentElement?.querySelector(
      "#dock-panel",
    ) as HTMLElement | null;
    if (!panel) return;
    panel.classList.add("dragging");
    const startY = e.clientY;
    const startH = parseInt(panel.style.height || "0", 10) || 0;
    const max = Math.round(window.innerHeight * 0.6);
    let moved = false;

    const apply = (h: number) => {
      const clamped = Math.max(0, Math.min(h, max));
      panel.style.height = `${clamped}px`;
      panel.dataset.closed = String(clamped === 0);
    };
    apply(startH);

    e.currentTarget.setPointerCapture(e.pointerId);

    const onMove = (ev: PointerEvent) => {
      const dy = startY - ev.clientY;
      if (Math.abs(dy) > 4) moved = true;
      if (moved) apply(startH + dy);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      panel.classList.remove("dragging");
      if (!moved) {
        // 轻点：开/关切换
        const cur = parseInt(panel.style.height || "0", 10) || 0;
        apply(cur > 0 ? 0 : Math.round(window.innerHeight * 0.4));
      }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  return (
    <div id="dock-handle" onPointerDown={onPointerDown} title="拖拽调整高度，轻点开/关">
      <div className="handle-line" />
    </div>
  );
}
