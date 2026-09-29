// mapper 轮次 → assistant-ui ThreadMessageLike 转换。
// 轮 = 1 条 user 消息 + 1 条 assistant 消息（text + tool-call parts）。

import type { ThreadMessageLike } from "@assistant-ui/react";
import type { HistoryRound, Part } from "./history/mapper";

export function roundToMessages(round: HistoryRound): ThreadMessageLike[] {
  const msgs: ThreadMessageLike[] = [];
  let assistantContent: Array<Record<string, unknown>> | null = null;

  const ensureAssistant = () => {
    if (!assistantContent) {
      assistantContent = [];
      msgs.push({ role: "assistant", content: assistantContent as never });
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
      content.push({ type: "text", text: part.text });
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
