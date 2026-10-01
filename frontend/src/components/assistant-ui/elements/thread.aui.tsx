"use client";

// NAN 的对话流。
//
// 与通用 assistant-ui Thread 的差异，都来自同一个事实：Agent 无限运行 step，
// 没有「一问一答」的回合。
//   - 一个 assistant 消息 = 一个 step，连续的 step 应当读起来是一条流，
//     而不是一摞独立的回复：step 之间只留小间距，没有逐条动作栏。
//   - 视口锚底（跟随最新），而不是把最新的用户消息顶到视口上沿。
//   - composer 永远停靠底部，永远可发送（输入进 Inbox，下一步看到）。
//   - 不做分支 / 编辑 / 重新生成 / 点赞 / 语音 / 附件 / 建议：NAN 不提供这些语义。

import { MarkdownText } from "@/components/assistant-ui/elements/markdown-text";
import {
  Reasoning,
  ReasoningContent,
  ReasoningRoot,
  ReasoningText,
  ReasoningTrigger,
} from "@/components/assistant-ui/elements/reasoning.aui";
import { ToolFallback } from "@/components/assistant-ui/elements/tool-fallback.aui";
import {
  ToolGroupContent,
  ToolGroupRoot,
  ToolGroupTrigger,
} from "@/components/assistant-ui/elements/tool-group.aui";
import { TooltipIconButton } from "@/components/assistant-ui/elements/tooltip-icon-button";
import { cn } from "@/lib/utils";
import {
  ActionBarPrimitive,
  AuiIf,
  type AssistantState,
  ComposerPrimitive,
  ErrorPrimitive,
  groupPartByType,
  MessagePrimitive,
  ThreadPrimitive,
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ClockIcon,
  CopyIcon,
} from "lucide-react";
import { PauseButton, usePauseControl } from "@/ui/pause";
import {
  createContext,
  useContext,
  type ComponentType,
  type FC,
  type PropsWithChildren,
} from "react";

export type ThreadGroupPart = MessagePrimitive.GroupedParts.GroupPart;

/**
 * Optional component overrides for the thread. `AssistantMessage` and
 * `Welcome` replace whole sections; the remaining slots override how the
 * assistant message renders tool calls and part groups. Tool UIs registered
 * by name (toolkit `render`, `useAssistantDataUI`) take precedence over
 * `ToolFallback`. When `TaskGroup` is set, tool calls that carry a nested
 * conversation and have no registered UI render through it instead of the
 * tool group; without it they render like any other tool call.
 */
export type ThreadComponents = {
  AssistantMessage?: ComponentType | undefined;
  Welcome?: ComponentType | undefined;
  ToolFallback?: ToolCallMessagePartComponent | undefined;
  ToolGroup?:
    | ComponentType<PropsWithChildren<{ group: ThreadGroupPart }>>
    | undefined;
  ReasoningGroup?:
    | ComponentType<PropsWithChildren<{ group: ThreadGroupPart }>>
    | undefined;
  TaskGroup?: ComponentType<{ group: ThreadGroupPart }> | undefined;
};

const messageGroupBy = groupPartByType({
  reasoning: ["group-chainOfThought", "group-reasoning"],
  "tool-call": ["group-chainOfThought", "group-tool"],
  "standalone-tool-call": [],
});

type ThreadGroupKey =
  | "group-chainOfThought"
  | "group-reasoning"
  | "group-tool"
  | "group-task";

const TASK_GROUP_PATH: readonly ThreadGroupKey[] = [
  "group-chainOfThought",
  "group-task",
];

const taskAwareGroupBy = (
  part: Parameters<typeof messageGroupBy>[0],
  context?: Parameters<typeof messageGroupBy>[1],
): readonly ThreadGroupKey[] => {
  const path = messageGroupBy(part, context);
  return part.type === "tool-call" &&
    part.messages !== undefined &&
    path.length > 0 &&
    !context?.toolUIs?.[part.toolName]?.length
    ? TASK_GROUP_PATH
    : path;
};

export type ThreadProps = {
  components?: ThreadComponents | undefined;
  autoFocus?: boolean | undefined;
  /** 已发送、但等当前轮结束才被模型看见的输入（输入框上方的等待卡）。 */
  pendingInputs?: readonly string[] | undefined;
};

const EMPTY_COMPONENTS: ThreadComponents = {};

const ThreadComponentsContext =
  createContext<ThreadComponents>(EMPTY_COMPONENTS);

const isEmptyView = (s: AssistantState) => s.thread.messages.length === 0;

export const Thread: FC<ThreadProps> = ({
  components = EMPTY_COMPONENTS,
  autoFocus = true,
  pendingInputs,
}) => {
  return (
    <ThreadComponentsContext.Provider value={components}>
      <ThreadRoot autoFocus={autoFocus} pendingInputs={pendingInputs} />
    </ThreadComponentsContext.Provider>
  );
};

const ThreadRoot: FC<{ autoFocus: boolean; pendingInputs?: readonly string[] }> = ({
  autoFocus,
  pendingInputs,
}) => {
  const { Welcome = ThreadWelcome } = useContext(ThreadComponentsContext);

  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root bg-background @container flex h-full flex-col"
      style={{
        // 宽屏下限制行长（中文约 45 字/行）；窄窗口仍是 100%
        ["--thread-max-width" as string]: "44rem",
        ["--composer-bg" as string]:
          "color-mix(in oklab, var(--color-muted) 30%, transparent)",
        ["--composer-radius" as string]: "1rem",
        ["--composer-padding" as string]: "8px",
      }}
    >
      {/* 锚底：Agent 持续输出，视口应当跟随最新内容 */}
      <ThreadPrimitive.Viewport
        turnAnchor="bottom"
        data-slot="aui_thread-viewport"
        className="relative flex flex-1 flex-col overflow-x-auto overflow-y-scroll"
      >
        <div className="mx-auto flex w-full max-w-(--thread-max-width) flex-1 flex-col px-4 pt-6">
          <AuiIf condition={isEmptyView}>
            <Welcome />
          </AuiIf>

          {/* 步骤间距由 .nan-stream 按相邻角色决定（见 style.css） */}
          <div
            data-slot="aui_message-group"
            className="nan-stream pb-8 empty:hidden"
          >
            <ThreadPrimitive.Messages>
              {() => <ThreadMessage />}
            </ThreadPrimitive.Messages>
          </div>

          {/* 页脚：内容从 composer 上方柔和淡出，而不是被硬切 */}
          <ThreadPrimitive.ViewportFooter
            className={cn(
              "aui-thread-viewport-footer bg-background sticky bottom-0 mt-auto flex flex-col gap-4 overflow-visible pb-4 md:pb-6",
              "before:pointer-events-none before:absolute before:inset-x-0 before:-top-6 before:h-6 before:bg-linear-to-t before:from-background before:to-transparent",
            )}
          >
            <ThreadScrollToBottom />
            {/* 等待卡：输入已发出，模型在下一轮才会看到 */}
            {pendingInputs?.map((text, i) => (
              <div
                key={i}
                data-slot="nan-pending"
                className="text-muted-foreground flex items-center gap-2 rounded-(--composer-radius) border bg-muted/40 px-3 py-2 text-sm"
              >
                <ClockIcon className="size-3.5 flex-none opacity-70" />
                <span className="min-w-0 flex-1 truncate">{text}</span>
                <span className="flex-none text-xs opacity-70">排队中</span>
              </div>
            ))}
            <Composer autoFocus={autoFocus} />
          </ThreadPrimitive.ViewportFooter>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const { AssistantMessage: AssistantMessageComponent = AssistantMessage } =
    useContext(ThreadComponentsContext);
  const role = useAuiState((s) => s.message.role);

  return role === "user" ? <UserMessage /> : <AssistantMessageComponent />;
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="回到最新"
        variant="outline"
        className="aui-thread-scroll-to-bottom dark:border-border dark:bg-background dark:hover:bg-accent absolute -top-12 z-10 self-center rounded-full p-4 disabled:invisible"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

/** 还没有任何 step：一个呼吸点 + 一句话。暂停时呼吸点由 CSS 自动静止。 */
const ThreadWelcome: FC = () => {
  return (
    <div className="aui-thread-welcome-root text-muted-foreground mb-6 flex items-center gap-1 px-2 text-sm">
      <span className="nan-dot" data-shape="dot" data-state="running" aria-hidden />
      <span>等待第一步</span>
    </div>
  );
};

// ------------------------------------------------------------------
// Composer
// ------------------------------------------------------------------

const Composer: FC<{ autoFocus: boolean }> = ({ autoFocus }) => {
  const pause = usePauseControl();
  // 暂停时告诉用户：话会被收下，但要继续后才会被看到
  const placeholder =
    pause?.state === "paused"
      ? "已暂停，输入的内容会在继续后被看到"
      : "说点什么，下一步就会看到";

  return (
    <ComposerPrimitive.Root className="aui-composer-root relative flex w-full flex-col">
      <div
        data-slot="aui_composer-shell"
        className="border-foreground/10 focus-within:border-foreground/25 flex w-full cursor-text flex-col gap-1 rounded-(--composer-radius) border bg-(--composer-bg) p-(--composer-padding) transition-[border-color] motion-reduce:transition-none"
      >
        <ComposerPrimitive.Input
          placeholder={placeholder}
          className="aui-composer-input caret-primary placeholder:text-muted-foreground/60 max-h-48 min-h-10 w-full resize-none bg-transparent px-2.5 py-1 text-base leading-6 outline-none"
          rows={1}
          autoFocus={autoFocus}
          enterKeyHint="send"
          aria-label="给 Agent 发消息"
        />
        <div className="aui-composer-action-wrapper relative flex items-center justify-between">
          <PauseButton />
          <ComposerPrimitive.Send asChild>
            <TooltipIconButton
              tooltip="发送"
              side="bottom"
              type="button"
              variant="default"
              size="icon"
              className="aui-composer-send ms-auto size-8 rounded-full"
              aria-label="发送"
            >
              <ArrowUpIcon className="size-4" />
            </TooltipIconButton>
          </ComposerPrimitive.Send>
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
};

// ------------------------------------------------------------------
// 消息
// ------------------------------------------------------------------

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root text-destructive mt-2 text-sm">
        <ErrorPrimitive.Message className="aui-message-error-message line-clamp-2" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  );
};

/** 一个 assistant 消息 = 一个 step：感知 → 言语 → 行动。 */
const AssistantMessage: FC = () => {
  const {
    ToolFallback: ToolFallbackComponent = ToolFallback,
    ToolGroup,
    ReasoningGroup,
    TaskGroup: TaskGroupComponent,
  } = useContext(ThreadComponentsContext);
  const groupBy = TaskGroupComponent ? taskAwareGroupBy : messageGroupBy;

  return (
    <MessagePrimitive.Root
      data-slot="aui_assistant-message-root"
      data-role="assistant"
      className="aui-assistant-step animate-in fade-in relative duration-200 [contain-intrinsic-size:auto_96px] [content-visibility:auto] motion-reduce:animate-none"
    >
      <div
        data-slot="aui_assistant-message-content"
        className="text-foreground px-2 leading-relaxed wrap-break-word"
      >
        <MessagePrimitive.GroupedParts groupBy={groupBy}>
          {({ part, children }) => {
            switch (part.type) {
              case "group-chainOfThought":
                return <div data-slot="aui_chain-of-thought">{children}</div>;
              case "group-task":
                return TaskGroupComponent ? (
                  <TaskGroupComponent group={part} />
                ) : null;
              case "group-tool":
                if (ToolGroup) {
                  return <ToolGroup group={part}>{children}</ToolGroup>;
                }
                return (
                  <ToolGroupRoot variant="ghost">
                    <ToolGroupTrigger
                      count={part.indices.length}
                      active={part.status.type === "running"}
                    />
                    <ToolGroupContent>{children}</ToolGroupContent>
                  </ToolGroupRoot>
                );
              case "group-reasoning": {
                if (ReasoningGroup) {
                  return (
                    <ReasoningGroup group={part}>{children}</ReasoningGroup>
                  );
                }
                const running = part.status.type === "running";
                return (
                  <ReasoningRoot streaming={running}>
                    <ReasoningTrigger active={running} />
                    <ReasoningContent aria-busy={running}>
                      <ReasoningText>{children}</ReasoningText>
                    </ReasoningContent>
                  </ReasoningRoot>
                );
              }
              case "text":
                return <MarkdownText />;
              case "reasoning":
                return <Reasoning {...part} />;
              case "tool-call":
                return part.toolUI ?? <ToolFallbackComponent {...part} />;
              case "data":
                return part.dataRendererUI;
              case "indicator":
                // 与时间线同一个「进行中」的点，而不是另一种符号
                return (
                  <span
                    data-slot="aui_assistant-message-indicator"
                    role="status"
                    aria-label="正在输出"
                    className="nan-dot"
                    data-shape="dot"
                    data-state="running"
                  />
                );
              default:
                return null;
            }
          }}
        </MessagePrimitive.GroupedParts>
        <MessageError />
      </div>

      <AssistantCopy />
    </MessagePrimitive.Root>
  );
};

/**
 * 唯一保留的消息动作：复制。悬停才出现、绝对定位不占版面——
 * 逐条 step 都带一排动作栏会把一条流切碎。没有正文的 step（纯感知/行动）不显示。
 */
const AssistantCopy: FC = () => {
  return (
    <AuiIf
      condition={(s) =>
        s.message.parts.some((p) => p.type === "text" && p.text.trim() !== "")
      }
    >
      <ActionBarPrimitive.Root
        autohide="always"
        className="aui-assistant-action-bar-root text-muted-foreground animate-in fade-in absolute end-0 top-0 duration-150"
      >
        <ActionBarPrimitive.Copy asChild>
          <TooltipIconButton tooltip="复制" className="size-7">
            <AuiIf condition={(s) => s.message.isCopied}>
              <CheckIcon className="animate-in zoom-in-50 fade-in duration-200 ease-out" />
            </AuiIf>
            <AuiIf condition={(s) => !s.message.isCopied}>
              <CopyIcon className="animate-in zoom-in-75 fade-in duration-150" />
            </AuiIf>
          </TooltipIconButton>
        </ActionBarPrimitive.Copy>
      </ActionBarPrimitive.Root>
    </AuiIf>
  );
};

/** 用户的话：靠右的浅灰气泡。输入进 Inbox，下一步被看到——没有编辑、没有分支。 */
const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      data-slot="aui_user-message-root"
      data-role="user"
      className="animate-in fade-in flex px-2 duration-200 [contain-intrinsic-size:auto_56px] [content-visibility:auto] motion-reduce:animate-none"
    >
      <div className="aui-user-message-content bg-muted text-foreground ms-auto max-w-[85%] min-w-0 rounded-(--composer-radius) px-4 py-2 wrap-break-word whitespace-pre-wrap empty:hidden">
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
};