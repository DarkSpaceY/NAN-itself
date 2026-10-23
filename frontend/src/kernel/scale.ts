// 纯缩放：所有面板渲染同一份固定宽度设计，父层 transform 缩放。
//
// 面板里永远是 1024px 宽的上下文，媒体查询永远命中同一个分支 ——
// 布局错落在结构上不可能发生。桥不传 scale、不传密度、不传断点。

/** 面板内部设计宽度。面板作者只能假设这一个宽度。 */
export const DESIGN_W = 1024;

/** 主区在超高窗口下的缩放上限，超过则加黑边而不是继续放大。 */
export const MAX_SCALE = 1.5;

export interface ScaledFrame {
  width: number;
  height: number;
  scale: number;
  transformOrigin: "top left";
}

/**
 * 把 DESIGN_W 宽的设计塞进 paneW × paneH 的视口。
 * 宽度不足时整体等比缩小；缩放超上限时停止放大（letterbox）。
 */
export function fitFrame(paneW: number, paneH: number): ScaledFrame {
  const raw = paneW / DESIGN_W;
  const scale = Math.min(raw, MAX_SCALE);
  return {
    width: DESIGN_W,
    // 先按 1:1 撑高内容，再由 transform 压缩到视口高度
    height: scale > 0 ? paneH / scale : paneH,
    scale,
    transformOrigin: "top left",
  };
}
