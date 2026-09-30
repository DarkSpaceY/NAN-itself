// 面派生：从事件流聚合出的素材推断舞台面。
//
// 后端只发事件，不参与渲染决策——「哪些算面、算哪一类」是前端的语义，
// 在这里集中定义（可单测），不要在组件里散落判断。
//
// 规则（方案B）：
// - 常驻面：最新的 output（当前轮的回复流）
// - 临时面：每条 record（工具/步骤记录），规模由容量与淡出控制

import type { Item } from "../store";
import { DEFAULT_WEIGHTS, type Face } from "./engine";

/** 素材 → 面。输入顺序无关（内部按 ts 判定最新）。 */
export function deriveFaces(items: readonly Item[]): Face[] {
  const faces: Face[] = [];

  let latestOutput: Item | undefined;
  for (const item of items) {
    if (item.kind !== "output") continue;
    if (!latestOutput || item.ts >= latestOutput.ts) latestOutput = item;
  }

  if (latestOutput) {
    faces.push({
      id: latestOutput.id,
      kind: "resident",
      weight: DEFAULT_WEIGHTS.resident,
      createdAt: latestOutput.ts,
    });
  }

  for (const item of items) {
    if (item.kind !== "record") continue;
    faces.push({
      id: item.id,
      kind: "temp",
      weight: DEFAULT_WEIGHTS.record,
      createdAt: item.ts,
    });
  }

  return faces;
}
