// mapper 轮次 → assistant-ui ThreadMessageLike 转换。
// 轮 = 1 条 user 消息 + 1 条 assistant 消息（text + tool-call parts）。

import type { MessageStatus, ThreadMessageLike } from "@assistant-ui/react";
import type { HistoryRound, Part } from "./history/mapper";

export function roundToMessages(round: HistoryRound): ThreadMessageLike[] {
  const msgs: ThreadMessageLike[] = [];
  let assistantContent: Array<Record<string, unknown>> | null = null;

  // 运行状态由 mapper 自持：我们必须给出 message 级 status，因为 aui
  // 只在 message 级为 running 时才采信 part 级 status（否则光标被吞）。
  const assistantStatus: MessageStatus = round.parts.some(
    (p) => p.kind === "text" && p.cancelled,
  )
    ? { type: "incomplete", reason: "cancelled" }
    : round.parts.some((p) => p.kind === "text" && p.streaming)
      ? { type: "running" }
      : { type: "complete", reason: "stop" };

  const ensureAssistant = () => {
    if (!assistantContent) {
      assistantContent = [];
      msgs.push({
        role: "assistant",
        content: assistantContent as never,
        status: assistantStatus,
      });
    }
    return assistantContent;
  };

  for (const part of round.parts) {
    if (part.kind === "text" && part.role === "user") {
      msgs.push({ role: "user", content: [{ type: "text", text: part.text }] });
      continue;
    }
    const content = ensureAssistant();
    if (part.kind === "text") {
      content.push({
        type: "text",
        text: part.text,
        // 流式状态由 mapper 自持，不依赖 aui 的 isRunning 推断
        status: part.cancelled
          ? { type: "incomplete", reason: "cancelled" }
          : part.streaming
            ? { type: "running" }
            : { type: "complete" },
      });
    } else {
      content.push(toolCallPart(part));
    }
  }

  return msgs;
}

function toolCallPart(part: Extract<Part, { kind: "tool" }>): Record<string, unknown> {
  const done = part.state !== "running";
  return {
    type: "tool-call",
    toolCallId: part.id,
    toolName: part.category,
    args: {},
    argsText: "",
    // running → 无 result，aui 视为进行中
    ...(done
      ? {
          result: {
            state: part.state,
            title: part.title,
            entries: part.entries ?? [],
            error: part.error,
            durationS: part.durationS,
          },
        }
      : {}),
  };
}

/** 供 React 渲染的扁平消息流。 */
export function roundsToMessages(rounds: readonly HistoryRound[]): ThreadMessageLike[] {
  return rounds.flatMap(roundToMessages);
}
