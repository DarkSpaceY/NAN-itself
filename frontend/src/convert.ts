// mapper 轮次 → assistant-ui ThreadMessageLike 转换。
//
// 一个 step 的解剖是「感知 → 言语 → 行动」：
//   感知 = 连续的 module_query（提示词组装，可能多个模块顺序给信息）
//   言语 = text
//   行动 = 连续的工具执行
// 连续的感知、连续的动作各自折叠成一个合成 part（__sense__ / __group__），
// 由 RecordTool 渲染成一行摘要；单个记录保持原样，不折叠。

import type { MessageStatus, ThreadMessageLike } from "@assistant-ui/react";
import type { HistoryRound, Part } from "./history/mapper";

type Content = Array<Record<string, unknown>>;

type Segment =
  | { kind: "user"; text: string }
  | { kind: "assistant"; parts: Part[] };

export function roundToMessages(round: HistoryRound): ThreadMessageLike[] {
  // 先按时间顺序切段：user 文本把 assistant 内容截成前后两条消息。
  // （旧实现把 user 之后的 assistant parts 追加回更早的那条 assistant 消息，
  //  用户在 step 中途插话时，顺序会错：后发生的内容出现在插话之前。）
  const segments: Segment[] = [];
  for (const part of round.parts) {
    if (part.kind === "text" && part.role === "user") {
      segments.push({ kind: "user", text: part.text });
      continue;
    }
    const last = segments[segments.length - 1];
    if (last?.kind === "assistant") last.parts.push(part);
    else segments.push({ kind: "assistant", parts: [part] });
  }

  return segments.map((seg): ThreadMessageLike => {
    if (seg.kind === "user") {
      return { role: "user", content: [{ type: "text", text: seg.text }] };
    }
    return {
      role: "assistant",
      // 轮进行中 → 合成组 part 标记 running：组的展开跟随轮生命周期，
      // 不跟着单条 record 的完成抖动（record 自身状态在 result 里）
      content: foldRecords(seg.parts.map(partToContent), !round.closed) as never,
      status: statusOf(seg.parts),
    };
  });
}

/**
 * 运行状态由 mapper 自持：必须给出 message 级 status，因为 aui 只在
 * message 级为 running 时才采信 part 级 status（否则光标被吞）。
 * 按每条消息自己的 parts 计算，避免多条 assistant 消息同时显示流式光标。
 */
function statusOf(parts: Part[]): MessageStatus {
  if (parts.some((p) => p.kind === "text" && p.cancelled)) {
    return { type: "incomplete", reason: "cancelled" };
  }
  if (parts.some((p) => p.kind === "text" && p.streaming)) {
    return { type: "running" };
  }
  return { type: "complete", reason: "stop" };
}

function partToContent(part: Part): Record<string, unknown> {
  if (part.kind === "text") {
    return {
      type: "text",
      text: part.text,
      // 流式状态由 mapper 自持，不依赖 aui 的 isRunning 推断
      status: part.cancelled
        ? { type: "incomplete", reason: "cancelled" }
        : part.streaming
          ? { type: "running" }
          : { type: "complete" },
    };
  }
  return toolCallPart(part);
}

function toolCallPart(part: Extract<Part, { kind: "tool" }>): Record<string, unknown> {
  const done = part.state !== "running";
  return {
    type: "tool-call",
    toolCallId: part.id,
    toolName: part.category,
    args: {},
    argsText: "",
    // running → 无 result。RecordTool 约定：没有 result = 正在进行
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

// ------------------------------------------------------------------
// 折叠
// ------------------------------------------------------------------

type RunKind = "sense" | "act";

const kindOf = (p: Record<string, unknown>): RunKind | null =>
  p.type !== "tool-call" ? null : p.toolName === "module_query" ? "sense" : "act";

/** 连续同类的 tool-call parts（≥2）合并成一个合成 part；其余原样保留。 */
function foldRecords(content: Content, roundOpen: boolean): Content {
  const out: Content = [];
  let run: Content = [];
  let runKind: RunKind | null = null;

  const flush = () => {
    if (run.length >= 2 && runKind) out.push(foldedPart(runKind, run, roundOpen));
    else out.push(...run);
    run = [];
    runKind = null;
  };

  for (const part of content) {
    const kind = kindOf(part);
    if (kind === null) {
      flush();
      out.push(part);
      continue;
    }
    if (kind !== runKind) flush();
    runKind = kind;
    run.push(part);
  }
  flush();
  return out;
}

function foldedPart(kind: RunKind, run: Content, roundOpen: boolean): Record<string, unknown> {
  return {
    type: "tool-call",
    // 以首条记录的 id 派生：比按序号计数稳定，历史重排时不会串号
    toolCallId: `${kind}-${String(run[0].toolCallId)}`,
    toolName: kind === "sense" ? "__sense__" : "__group__",
    args: {},
    argsText: "",
    // 组的「轮进行中」信号放 result 里（aui 不透传自定义 status 字段）
    result: {
      turnOpen: roundOpen,
      records: run.map((p) => ({ toolName: p.toolName, result: p.result })),
    },
  };
}

/** 供 React 渲染的扁平消息流。排队中的输入由输入框上方的等待卡呈现，不进消息流。 */
export function roundsToMessages(rounds: readonly HistoryRound[]): ThreadMessageLike[] {
  return rounds.flatMap(roundToMessages);
}