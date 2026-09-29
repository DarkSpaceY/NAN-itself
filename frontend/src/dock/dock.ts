// 中央 dock：输入框（唯一常驻组件）+ 上拉历史浅面板。
// - 面板高度由把手拖拽控制，0 = 关闭；轻点把手切换开/关
// - 输入框 Enter 发送（Shift+Enter 换行），自动增高
// - 居中悬浮块 max-width 720px，不通栏贴底

export interface DockOptions {
  onSend: (text: string) => void;
  /** 连接状态变化（输入框禁用/提示）。 */
  placeholder?: string;
}

export interface Dock {
  root: HTMLElement;
  /** 历史流渲染容器（HistoryView 挂载点）。 */
  historyEl: HTMLElement;
  setConnection(state: "connecting" | "open" | "closed"): void;
}

const PANEL_MAX_VH = 60;

export function createDock(opts: DockOptions): Dock {
  const root = document.createElement("div");
  root.id = "dock";

  // ---- 历史面板（上拉浅面板，只读） ----
  const panel = document.createElement("div");
  panel.id = "dock-panel";
  panel.style.height = "0px";

  const historyEl = document.createElement("div");
  historyEl.id = "dock-history";
  panel.appendChild(historyEl);

  // ---- 把手（细线，拖拽调高 / 轻点开关） ----
  const handle = document.createElement("div");
  handle.id = "dock-handle";
  handle.title = "拖拽调整高度，轻点开/关";
  const handleLine = document.createElement("div");
  handleLine.className = "handle-line";
  handle.appendChild(handleLine);

  // ---- 输入行 ----
  const inputRow = document.createElement("div");
  inputRow.id = "dock-input-row";

  const textarea = document.createElement("textarea");
  textarea.id = "dock-input";
  textarea.rows = 1;
  textarea.placeholder = opts.placeholder ?? "输入消息…";
  textarea.addEventListener("input", autoGrow);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape" && panelHeight() > 0) {
      setPanelHeight(0);
    }
  });

  const sendBtn = document.createElement("button");
  sendBtn.id = "dock-send";
  sendBtn.type = "button";
  sendBtn.textContent = "↑";
  sendBtn.addEventListener("click", submit);

  inputRow.append(textarea, sendBtn);
  root.append(panel, handle, inputRow);

  // ---- 发送 ----
  function submit() {
    const text = textarea.value.trim();
    if (!text) return;
    opts.onSend(text);
    textarea.value = "";
    autoGrow();
    // 发送后面板收起，聚焦舞台/输入
    setPanelHeight(0);
    textarea.focus();
  }

  function autoGrow() {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }

  // ---- 面板高度控制 ----
  function panelHeight(): number {
    return parseInt(panel.style.height, 10) || 0;
  }

  function setPanelHeight(px: number) {
    const max = Math.round(window.innerHeight * (PANEL_MAX_VH / 100));
    const clamped = Math.max(0, Math.min(px, max));
    panel.style.height = `${clamped}px`;
    root.classList.toggle("panel-open", clamped > 0);
  }

  // 把手交互：拖拽调高；位移 < 4px 视为轻点切换
  let dragging = false;
  let moved = false;
  let startY = 0;
  let startH = 0;

  handle.addEventListener("pointerdown", (e) => {
    dragging = true;
    moved = false;
    startY = e.clientY;
    startH = panelHeight();
    handle.setPointerCapture(e.pointerId);
  });

  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dy = startY - e.clientY; // 向上拖 = 展开
    if (Math.abs(dy) > 4) moved = true;
    if (moved) setPanelHeight(startH + dy);
  });

  handle.addEventListener("pointerup", (e) => {
    dragging = false;
    handle.releasePointerCapture(e.pointerId);
    if (!moved) {
      // 轻点：开/关切换
      setPanelHeight(panelHeight() > 0 ? 0 : Math.round(window.innerHeight * 0.4));
    }
    if (panelHeight() > 0) historyEl.focus();
  });

  // ---- 连接状态 ----
  function setConnection(state: "connecting" | "open" | "closed") {
    root.dataset.connection = state;
    textarea.disabled = state !== "open";
    textarea.placeholder =
      state === "open"
        ? (opts.placeholder ?? "输入消息…")
        : state === "connecting"
          ? "连接中…"
          : "连接已断开，等待重连…";
  }
  setConnection("connecting");

  return { root, historyEl, setConnection };
}
