// 舞台渲染：常驻面（当前轮流式输出）+ 临时面（工具/产出记录）。
// portal 到 #stage；布局由 stage/engine 计算；面淡出后由本组件移除。

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { useSyncExternalStore } from "react";
import { computeLayout, DEFAULT_WEIGHTS, type Face } from "../stage/engine";
import { toolTitle } from "../history/mapper";
import type { Store, Item } from "../store";

const FADE_MS = 400;

export function Stage({ store }: { store: Store }) {
  const state = useSyncExternalStore(
    (cb) => store.subscribe(cb),
    () => store.getState(),
  );

  // fading 面的宿主状态：动画结束后真正移除
  const [evicted, setEvicted] = useState<Set<string>>(new Set());

  const items: Item[] = useMemo(
    () => Object.values(state.items).sort((a, b) => a.ts - b.ts),
    [state.items],
  );

  const faces: Face[] = useMemo(() => {
    const list: Face[] = [];
    // 常驻面：当前轮 = 最新的 output 流
    const outputs = items.filter((i) => i.kind === "output");
    const current = outputs.at(-1);
    if (current) {
      list.push({
        id: current.id,
        kind: "resident",
        weight: DEFAULT_WEIGHTS.resident,
        focus: false,
        createdAt: current.ts,
      });
    }
    // 临时面：工具记录（产出流已并入常驻面时跳过）
    for (const item of items) {
      if (item.kind !== "record") continue;
      list.push({
        id: item.id,
        kind: "temp",
        weight: DEFAULT_WEIGHTS.record,
        focus: false,
        createdAt: item.ts,
      });
    }
    return list;
  }, [items]);

  // 引擎计算 + fading 超时回收
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
        const item = state.items[slot.id];
        return (
          <div
            key={slot.id}
            className={`face ${slot.primary ? "face-primary" : "face-secondary"} kind-${face.kind}`}
          >
            <FaceBody item={item} />
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
  if (item.kind === "output") {
    return (
      <div className="face-body face-output">
        <div className="face-label">回复</div>
        <div className="face-text">{item.text || "…"}</div>
      </div>
    );
  }
  return (
    <div className={`face-body face-record state-${item.state}`}>
      <div className="face-label">
        <span className="dot" /> {item.category}
      </div>
      <div className="face-text">
        {toolTitle(item.category, item.payload)}
      </div>
    </div>
  );
}
