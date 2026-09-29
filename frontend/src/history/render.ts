// 历史流渲染：自建 CSS，kitn.ai/ui 视觉模式——
// 内联工具卡片（圆点状态、可展开 detail）、纯文字轮次。
// 只读：不绑定任何影响状态的行为，文本可选中复制。

import type { HistoryRound, Part, ToolPart, TextPart } from "./mapper";

export class HistoryView {
  private root: HTMLElement;
  private roundEls = new Map<string, HTMLElement>();

  constructor(root: HTMLElement) {
    this.root = root;
  }

  /** 全量重设（IndexedDB 种子恢复后）。 */
  setAll(rounds: readonly HistoryRound[]): void {
    this.root.replaceChildren();
    this.roundEls.clear();
    for (const r of rounds) this.upsert(r);
    this.scrollToBottom();
  }

  /** 单轮增量更新（mapper 回调）。 */
  upsert(round: HistoryRound): void {
    let el = this.roundEls.get(round.key);
    if (!el) {
      el = document.createElement("div");
      el.className = "round";
      el.dataset.key = round.key;
      this.roundEls.set(round.key, el);

      // 按 ts 找插入位置（种子乱序时保持时间序）
      const next = [...this.roundEls.entries()]
        .filter(([k]) => k !== round.key)
        .map(([k]) => this.roundEls.get(k)!)
        .find((sibling) => Number(sibling.dataset.ts ?? 0) > round.ts);
      this.root.insertBefore(el, next ?? null);
    }
    el.dataset.ts = String(round.ts);
    el.classList.toggle("closed", round.closed);

    el.replaceChildren(...round.parts.map((p) => renderPart(p)));
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    this.root.scrollTop = this.root.scrollHeight;
  }
}

// ------------------------------------------------------------------
// part → DOM
// ------------------------------------------------------------------

function renderPart(part: Part): HTMLElement {
  return part.kind === "text" ? renderText(part) : renderTool(part);
}

function renderText(part: TextPart): HTMLElement {
  const el = document.createElement("div");
  el.className = `part text ${part.role}`;
  if (part.cancelled) el.classList.add("cancelled");

  const body = document.createElement("div");
  body.className = "text-body";
  body.textContent = part.text || (part.streaming ? "" : "（空回复）");
  el.appendChild(body);

  if (part.streaming) {
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    cursor.textContent = "▍";
    el.appendChild(cursor);
  }
  return el;
}

function renderTool(part: ToolPart): HTMLElement {
  const el = document.createElement("div");
  el.className = `part tool state-${part.state}`;
  el.dataset.category = part.category;

  // 头行：圆点 + 标题 + 状态/耗时
  const head = document.createElement("div");
  head.className = "tool-head";

  const dot = document.createElement("span");
  dot.className = "dot";
  head.appendChild(dot);

  const title = document.createElement("span");
  title.className = "tool-title";
  title.textContent = part.title;
  head.appendChild(title);

  const meta = document.createElement("span");
  meta.className = "tool-meta";
  meta.textContent = metaLabel(part);
  head.appendChild(meta);

  el.appendChild(head);

  // 错误信息直接展示
  if (part.error) {
    const err = document.createElement("div");
    err.className = "tool-error";
    err.textContent = `${part.error.type}: ${part.error.message}`;
    el.appendChild(err);
  }

  // detail 可展开（有 entries 才渲染展开区）
  if (part.entries && part.entries.length > 0) {
    const toggle = document.createElement("button");
    toggle.className = "tool-toggle";
    toggle.type = "button";
    toggle.textContent = "详情";
    toggle.addEventListener("click", () => {
      const open = el.classList.toggle("open");
      toggle.textContent = open ? "收起" : "详情";
    });
    head.appendChild(toggle);

    const detail = document.createElement("div");
    detail.className = "tool-detail";
    for (const entry of part.entries) {
      const row = document.createElement("div");
      row.className = `entry kind-${entry.kind ?? "text"}`;
      if (entry.kind === "field" && typeof entry.label === "string") {
        const label = document.createElement("span");
        label.className = "entry-label";
        label.textContent = entry.label;
        row.appendChild(label);
      }
      const value = document.createElement("span");
      value.className = "entry-value";
      value.textContent =
        entry.kind === "text" ? String(entry.text ?? "") : String(entry.value ?? "");
      row.appendChild(value);
      detail.appendChild(row);
    }
    el.appendChild(detail);
  }

  return el;
}

function metaLabel(part: ToolPart): string {
  switch (part.state) {
    case "running":
      return "运行中";
    case "void":
      return "已作废";
    case "failed":
      return "失败";
    case "done":
      return part.durationS != null ? `${part.durationS.toFixed(1)}s` : "完成";
  }
}
