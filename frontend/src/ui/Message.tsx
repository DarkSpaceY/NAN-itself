// 消息渲染：复用 v1 的视觉类名（.round/.part/.tool），自建 CSS 不换。

import { MessagePrimitive } from "@assistant-ui/react";
import type {
  TextMessagePartProps,
  ToolCallMessagePartProps,
} from "@assistant-ui/react";
import { MarkdownText } from "./markdown";

export function UserMessage() {
  return (
    <div className="round">
      <div className="part text user">
        <div className="text-body">
          <MessagePrimitive.Parts />
        </div>
      </div>
    </div>
  );
}

export function AssistantMessage() {
  return (
    <div className="round">
      <MessagePrimitive.Parts
        components={{
          Text: TextPart,
          tools: { Fallback: ToolCard },
        }}
      />
    </div>
  );
}

// 自定义 Text 部件渲染器：只渲染「这个 part 自己」的内容。
// 注意不能用 MessagePrimitive.Parts（那是整个部件列表的渲染器，
// 在部件渲染器里再调用它会递归展开、把同一段内容重复渲染）。
// 流式光标由 markdown 容器的 data-status 驱动（见 dot.css）。
function TextPart({ status }: TextMessagePartProps) {
  const cancelled = status?.type === "incomplete" && status.reason === "cancelled";
  return (
    <div className={`part text assistant text-body${cancelled ? " cancelled" : ""}`}>
      <MarkdownText />
    </div>
  );
}

interface ToolResult {
  state: "running" | "done" | "failed" | "void";
  title: string;
  entries?: Array<Record<string, unknown>>;
  error?: { type: string; message: string };
  durationS?: number;
}

function ToolCard({ toolName, result }: ToolCallMessagePartProps) {
  const r = (result ?? undefined) as ToolResult | undefined;
  const state = r?.state ?? "running";
  const title = r?.title ?? toolName;

  return (
    <div className={`part tool state-${state}`} data-category={toolName}>
      <div className="tool-head">
        <span className="dot" />
        <span className="tool-title">{title}</span>
        <span className="tool-meta">{metaLabel(state, r?.durationS)}</span>
      </div>
      {r?.error && (
        <div className="tool-error">
          {r.error.type}: {r.error.message}
        </div>
      )}
      {r?.entries && r.entries.length > 0 && (
        <details className="tool-detail">
          <summary>详情</summary>
          {r.entries.map((entry, i) => (
            <div key={i} className={`entry kind-${entry.kind ?? "text"}`}>
              {entry.kind === "field" && typeof entry.label === "string" && (
                <span className="entry-label">{entry.label}</span>
              )}
              <span className="entry-value">
                {entry.kind === "text" ? String(entry.text ?? "") : String(entry.value ?? "")}
              </span>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

function metaLabel(state: string, durationS?: number): string {
  switch (state) {
    case "running":
      return "运行中";
    case "void":
      return "已作废";
    case "failed":
      return "失败";
    case "done":
      return durationS != null ? `${durationS.toFixed(1)}s` : "完成";
    default:
      return "";
  }
}
