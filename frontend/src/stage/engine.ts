// 舞台布局引擎：纯函数，语义与几何分离。
//
// 设计约束（方案B）：
// - 面分两类：常驻面（当前对话轮）与临时面（机器活动/工作产物）
// - 排序与占位：聚焦者占主位；无聚焦时按 类型默认权重 > 同权重最新优先
// - LLM focus 标记（surface.* 事件）> 类型默认权重 > 最新优先
// - 临时面容量：舞台 ≥5 个面时最旧临时面进入 fading，动画后移除
// - 面与历史互不关联：淡出不在历史留痕

// ------------------------------------------------------------------
// 模型
// ------------------------------------------------------------------

export type FaceKind = "resident" | "temp";

export interface Face {
  id: string;
  kind: FaceKind;
  /** 类型默认权重：越大越靠前。 */
  weight: number;
  /** LLM focus 标记（阶段4由 surface.* 驱动）。 */
  focus: boolean;
  /** 同权重内最新优先的时序依据。 */
  createdAt: number;
  /** 淡出中的面：仍在舞台，动画结束后由宿主移除。 */
  fading?: boolean;
}

export interface Slot {
  id: string;
  /** 主位（聚焦者）或次位。 */
  primary: boolean;
}

export interface LayoutResult {
  slots: Slot[];
  /** 本轮被判淡出的面 id（宿主负责播放动画后移除）。 */
  fadingIds: string[];
}

/** 舞台临时面容量（含常驻面共 5）。 */
export const STAGE_CAPACITY = 5;

/** 类型默认权重表：当前轮 > 进行中工具 > 产出流 > 其他记录。 */
export const DEFAULT_WEIGHTS: Record<string, number> = {
  resident: 100,
  record: 60,
  output: 50,
};

// ------------------------------------------------------------------
// 引擎
// ------------------------------------------------------------------

/**
 * 计算舞台布局。
 * - 聚焦面（focus=true，多个时取最新）占主位
 * - 无聚焦时排序第一者占主位
 * - 超容量时最旧临时面标记 fading（常驻面永不淡出）
 * - fading 面不参与排序占位，但保留在 fadingIds 中供宿主动画
 */
export function computeLayout(faces: readonly Face[]): LayoutResult {
  const fadingIds = faces
    .filter((f) => f.fading)
    .map((f) => f.id);

  const active = faces.filter((f) => !f.fading);

  // 超容量：最旧临时面淡出（fading 中的不再重复计入）
  const overflow = active.length - STAGE_CAPACITY;
  let toFade = new Set<string>();
  if (overflow > 0) {
    const temps = active
      .filter((f) => f.kind === "temp")
      .sort((a, b) => a.createdAt - b.createdAt);
    toFade = new Set(temps.slice(0, overflow).map((f) => f.id));
  }

  const staged = active.filter((f) => !toFade.has(f.id));

  // 排序：focus > weight > createdAt 降序
  const sorted = [...staged].sort((a, b) => {
    if (a.focus !== b.focus) return a.focus ? -1 : 1;
    if (a.weight !== b.weight) return b.weight - a.weight;
    return b.createdAt - a.createdAt;
  });

  const focusTarget = sorted.find((f) => f.focus) ?? sorted[0];

  const slots = sorted.map((f) => ({
    id: f.id,
    primary: f === focusTarget,
  }));

  return {
    slots,
    fadingIds: [...fadingIds, ...toFade],
  };
}
