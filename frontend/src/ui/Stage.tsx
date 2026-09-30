// 舞台渲染：常驻面（当前轮回复流）+ 临时面（工具/步骤记录）。
// portal 到 #stage；面由 stage/faces 从素材派生，布局由 stage/engine 计算，
// 淡出动画结束后由本组件移除。

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { computeLayout, type Face } from "../stage/engine";
import { deriveFaces } from "../stage/faces";
import { toolTitle } from "../history/mapper";
import type { Item, OutputItem, RecordItem, Store } from "../store";
import { Markdown } from "./markdown";

const FADE_MS = 400;

export function Stage({ store }: { store: Store }) {
  const state = useSyncExternalStore(
    (cb) => store.subscribe(cb),
    () => store.getState(),
  );

  // fading 面的宿主状态：动画结束后真正移除
  const [evicted, setEvicted] = useState<Set<string>>(new Set());

  const faces: Face[] = useMemo(
    () => deriveFaces(Object.values(state.items)),
    [state.items],
  );

  const layout = useMemo(() => computeLayout(faces), [faces]);

  useEffect(() => {
    const fresh = layout.fadingIds.filter((id) => !evicted.has(id));
    if (fresh.length === 0) return;
    const t = setTimeout(() => {
      setEvicted((prev) => {
        const next = new Set(prev);
        for (const id of layout.fadingIds) next.add(id);
        return next;
      });
    }, FADE_MS);
    return () => clearTimeout(t);
  }, [layout, evicted]);

  const byId = new Map(faces.map((f) => [f.id, f]));

  return createPortal(
    <div className="stage-grid">
      {layout.slots.map((slot) => {
        const face = byId.get(slot.id)!;
        return (
          <div
            key={slot.id}
            className={`face ${slot.primary ? "face-primary" : "face-secondary"} kind-${face.kind}`}
          >
            <FaceBody item={state.items[slot.id]} />
          </div>
        );
      })}
      {/* fading 面先播动画再移除 */}
      {layout.fadingIds
        .filter((id) => !evicted.has(id))
        .map((id) => {
          const item = state.items[id];
          return item ? (
            <div key={id} className="face face-fading kind-temp">
              <FaceBody item={item} />
            </div>
          ) : null;
        })}
    </div>,
    document.getElementById("stage")!,
  );
}

function FaceBody({ item }: { item: Item | undefined }) {
  if (!item) return null;
  return item.kind === "output" ? (
    <OutputFace item={item} />
  ) : (
    <RecordFace item={item} />
  );
}

function OutputFace({ item }: { item: OutputItem }) {
  return (
    <div className="face-body face-output">
      <div className="face-head">
        <span className="face-label">回复</span>
        <span className="face-meta">{stateLabel(item.state)}</span>
      </div>
      <div className="face-text">
        <Markdown text={item.text} />
        {!item.text && item.state === "streaming" && (
          <span className="face-placeholder">思考中…</span>
        )}
      </div>
    </div>
  );
}

function RecordFace({ item }: { item: RecordItem }) {
  const entries = item.entries ?? [];
  return (
    <div className={`face-body face-record state-${item.state}`}>
      <div className="face-head">
        <span className="dot" />
        <span className="face-title">
          {toolTitle(item.category, item.payload)}
        </span>
        <span className="face-meta">{stateLabel(item.state, item.durationS)}</span>
      </div>
      <div className="face-label">{item.category}</div>
      {item.error && (
        <div className="tool-error">
          {item.error.type}: {item.error.message}
        </div>
      )}
      {entries.length > 0 && (
        <details className="tool-detail">
          <summary>详情</summary>
          {entries.map((entry, i) => (
            <div key={i} className={`entry kind-${entry.kind ?? "text"}`}>
              {entry.kind === "field" && typeof entry.label === "string" && (
                <span className="entry-label">{entry.label}</span>
              )}
              <span className="entry-value">
                {entry.kind === "text"
                  ? String(entry.text ?? "")
                  : String(entry.value ?? "")}
              </span>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

function stateLabel(state: string, durationS?: number): string {
  switch (state) {
    case "running":
    case "streaming":
      return "进行中";
    case "void":
      return "已作废";
    case "cancelled":
      return "已取消";
    case "failed":
      return "失败";
    case "done":
      return durationS != null ? `${durationS.toFixed(1)}s` : "完成";
    default:
      return "";
  }
}
