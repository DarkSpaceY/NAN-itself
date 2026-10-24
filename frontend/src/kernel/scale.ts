// Pure scaling: every pane renders the same fixed-width design; the
// parent scales it with a transform.
//
// Inside a pane the context is always 1024px wide and media queries
// always hit the same branch -- broken layout is structurally
// impossible. The bridge passes no scale, no density, no breakpoints.

/** Design width inside a pane. Pane authors may assume only this width. */
export const DESIGN_W = 1024;

/** Scale cap for the main area on very tall windows; beyond it, letterbox instead of enlarging further. */
export const MAX_SCALE = 1.5;

export interface ScaledFrame {
  width: number;
  height: number;
  scale: number;
  transformOrigin: "top left";
}

/**
 * Fit the DESIGN_W-wide design into a paneW x paneH viewport.
 * Scale down proportionally when width runs short; stop enlarging past
 * the cap (letterbox).
 */
export function fitFrame(paneW: number, paneH: number): ScaledFrame {
  const raw = paneW / DESIGN_W;
  const scale = Math.min(raw, MAX_SCALE);
  return {
    width: DESIGN_W,
    // Size the content at 1:1 height first, then compress to the viewport height via transform
    height: scale > 0 ? paneH / scale : paneH,
    scale,
    transformOrigin: "top left",
  };
}
