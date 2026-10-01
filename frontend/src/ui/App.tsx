// App：ExternalStoreRuntime + 官方 Thread 素材。
//
// 布局：单列对话流（往上滚即历史），composer 随 Thread 内置。
// Agent 无限运行 step——没有「轮」的边界，所以界面不设「等待回复」态：
// 用户任何时候都能发言，输入进 Inbox，下一个 step 的 module_query 会看到。
// 暂停按钮挂在 composer 操作行（经 pause context 下发，见 pause.tsx）。

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
  type PropsWithChildren,
} from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import type { ThreadMessageLike } from "@assistant-ui/react";

import { Store, type NanState } from "../store";
import { Mapper, type HistoryRound } from "../history/mapper";
import type { RoundStore } from "../persistence/db";
import { roundsToMessages } from "../convert";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { RecordTool } from "./RecordTool";
import { PauseControlContext, type PauseControl } from "./pause";

export interface AppProps {
  store: Store;
  mapper: Mapper;
  db: RoundStore;
  sendInput: (text: string) => void;
  setPaused: (paused: boolean) => void;
}

// shadcn 主题用 .dark class（不走媒体查询），这里同步系统偏好。
function useDarkMode() {
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => document.documentElement.classList.toggle("dark", mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);
}

// ToolGroup：连续的工具调用共用一条脊线，成为「一段活动」。
// 必须是模块级的稳定组件：若在 render 里内联定义，每次 App 重渲染都会换
// 组件类型，整段时间线被卸载重挂，展开状态全部丢失。
function ActivityGroup({ children }: PropsWithChildren) {
  return <div className="nan-activity">{children}</div>;
}

const threadComponents = {
  ToolFallback: RecordTool,
  ToolGroup: ActivityGroup,
};

export function App({ store, mapper, db, sendInput, setPaused }: AppProps) {
  useDarkMode();

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

  // ── 暂停 ───────────────────────────────────────────────────────
  // 状态由后端 status 事件回传（协作式，可能在当前 step 跑完后才生效）。
  // 点「暂停」后到后端确认之前是 pending；pending 时再点 = 撤回请求。
  const isPaused = nanState.status === "paused";
  const [pausePending, setPausePending] = useState(false);

  useEffect(() => {
    if (isPaused) setPausePending(false);
  }, [isPaused]);

  const pauseControl = useMemo<PauseControl>(() => {
    const pending = pausePending && !isPaused;
    return {
      state: isPaused ? "paused" : pending ? "pending" : "running",
      paused: isPaused,
      pending,
      toggle: () => {
        if (isPaused) {
          setPaused(false);
        } else if (pending) {
          setPausePending(false);
          setPaused(false);
        } else {
          setPausePending(true);
          setPaused(true);
        }
      },
    };
  }, [isPaused, pausePending, setPaused]);

  // ── 持久化：关闭轮立即落盘，活动轮节流；卸载时冲刷未落盘的活动轮 ──
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
        const r = pending;
        pending = null;
        if (r && !r.closed) void db.saveRound(r);
      }, 500);
    });
    return () => {
      unsub();
      if (timer) clearTimeout(timer);
      if (pending && !pending.closed) void db.saveRound(pending);
    };
  }, [mapper, db]);

  // ── 消息 ───────────────────────────────────────────────────────
  const messages: ThreadMessageLike[] = useMemo(
    () => roundsToMessages(mapper.getRounds()),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [roundsVersion],
  );

  // 排队中的输入（已发出、等当前轮结束才被看见）——输入框上方的等待卡
  const pendingInputs: string[] = useMemo(
    () => mapper.getPendingTexts(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [roundsVersion],
  );

  const runtime = useExternalStoreRuntime({
    messages,
    // 显式传 false：aui 在 isRunning === undefined 时会回退用最后一条
    // assistant 消息的 status 推断运行中并禁用发送；agent 常态就在跑，
    // 用户任何时候都能发言。
    isRunning: false,
    convertMessage: (m) => m,
    onNew: async ({ content }) => {
      const text = content
        .filter((c): c is { type: "text"; text: string } => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      if (text.trim()) sendInput(text);
    },
    onCancel: undefined,
  });

  return (
    <PauseControlContext.Provider value={pauseControl}>
      <AssistantRuntimeProvider runtime={runtime}>
        {/* 固定高度容器：composer 的 sticky 底部停靠依赖它。
            data-* 供 CSS 响应：暂停确认后，所有「进行中」的呼吸点静止。 */}
        <div
          className="relative h-dvh"
          data-agent-status={nanState.status}
          data-paused={isPaused}
        >
          <Thread
            components={threadComponents}
            autoFocus
            pendingInputs={pendingInputs}
          />
        </div>
      </AssistantRuntimeProvider>
    </PauseControlContext.Provider>
  );
}