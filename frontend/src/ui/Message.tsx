// 消息渲染：复用 v1 的视觉类名（.round/.part/.tool），自建 CSS 不换。

import { MessagePrimitive } from "@assistant-ui/react";
import type { ToolCallMessagePartProps } from "@assistant-ui/react";

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

function TextPart() {
  return <div className="part text assistant text-body"><MessagePrimitive.Parts /></div>;
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
