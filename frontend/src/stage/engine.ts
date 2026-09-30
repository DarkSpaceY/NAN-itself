// 舞台布局引擎：纯函数，语义与几何分离。
//
// 设计约束（方案B）：
// - 面分两类：常驻面（当前对话轮）与临时面（工具/步骤记录）
// - 面的来源是前端对事件流的推断（后端只发事件，不参与渲染决策），
//   见 stage/faces.ts
// - 排序与占位：类型默认权重 > 同权重最新优先；首位占主位
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
  /** 同权重内最新优先的时序依据。 */
  createdAt: number;
  /** 淡出中的面：仍在舞台，动画结束后由宿主移除。 */
  fading?: boolean;
}

export interface Slot {
  id: string;
  /** 主位（排序首位）或次位。 */
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
 * - 排序第一者占主位
 * - 超容量时最旧临时面标记 fading（常驻面永不淡出）
 * - fading 面不参与排序占位，但保留在 fadingIds 中供宿主动画
 */
export function computeLayout(faces: readonly Face[]): LayoutResult {
  const fadingIds = faces.filter((f) => f.fading).map((f) => f.id);

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

  // 排序：weight 降序 > createdAt 降序
  const sorted = [...staged].sort((a, b) => {
    if (a.weight !== b.weight) return b.weight - a.weight;
    return b.createdAt - a.createdAt;
  });

  const slots = sorted.map((f, i) => ({
    id: f.id,
    primary: i === 0,
  }));

  return {
    slots,
    fadingIds: [...fadingIds, ...toFade],
  };
}
