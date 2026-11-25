// Window catalog: one built-in agent pane + one window per registered
// service.
//
// Registration only produces WindowRef; queue order is managed by
// kernel/queue, the two never mix. A service without a declared icon
// degrades to the service_id initial; a blank chip never appears.

import type { SurfaceSpec } from "../kernel/protocol";
import type { WindowRef } from "../kernel/queue";

export const AGENT_ID = "agent";

/** One service, one window id; registration and removal share it. */
export function serviceWindowId(serviceId: string): string {
  return `svc:${serviceId}`;
}

/** Hexagon mark (the product glyph): root and subagents share it. */
export const AGENT_ICON =
  "M12 2l8.66 5v10L12 22l-8.66-5V7zM12 5.52 6.39 8.76v6.48L12 18.48l5.61-3.24V8.76z";

export function builtinWindow(): WindowRef {
  return { id: AGENT_ID, kind: "agent", title: "agent", icon: AGENT_ICON };
}

export function serviceWindow(spec: SurfaceSpec): WindowRef {
  return {
    id: serviceWindowId(spec.service_id),
    kind: "service",
    title: spec.title,
    icon: spec.icon,
    serviceId: spec.service_id,
    url: spec.url,
  };
}

/** Single-letter mark used when there is no icon. */
export function monogram(win: WindowRef): string {
  const source = win.serviceId ?? win.title;
  return source.slice(0, 1).toUpperCase();
}

function isAsset(icon: string): boolean {
  return (
    icon.startsWith("/") ||
    icon.startsWith("http://") ||
    icon.startsWith("https://")
  );
}

export function WindowIcon({
  win,
  size = 16,
}: {
  win: WindowRef;
  size?: number;
}) {
  if (!win.icon) {
    return (
      <span
        className="monogram"
        style={{ fontSize: Math.round(size * 0.6) }}
        aria-hidden="true"
      >
        {monogram(win)}
      </span>
    );
  }

  if (isAsset(win.icon)) {
    return (
      <img
        className="asset-icon"
        src={win.icon}
        width={size}
        height={size}
        alt=""
      />
    );
  }

  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d={win.icon} fill="currentColor" />
    </svg>
  );
}
