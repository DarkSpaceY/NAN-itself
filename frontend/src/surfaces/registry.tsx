// 窗口目录：一个内建 agent 面板 + 每个注册的服务窗口。
//
// 注册只产生 WindowRef；队列顺序由 kernel/queue 管，二者不混。
// 服务未声明 icon 时退化为 service_id 首字母，绝不出现空白 chip。

import type { SurfaceSpec } from "../kernel/protocol";
import type { WindowRef } from "../kernel/queue";

export const AGENT_ID = "agent";

/** 四角星：root 与子 agent 共用一个 glyph。 */
export const AGENT_ICON =
  "M12 2.5l2.35 7.15L21.5 12l-7.15 2.35L12 21.5l-2.35-7.15L2.5 12l7.15-2.35z";

export function builtinWindow(): WindowRef {
  return { id: AGENT_ID, kind: "agent", title: "agent", icon: AGENT_ICON };
}

export function serviceWindow(spec: SurfaceSpec): WindowRef {
  return {
    id: `svc:${spec.service_id}`,
    kind: "service",
    title: spec.title,
    icon: spec.icon,
    serviceId: spec.service_id,
    url: spec.url,
  };
}

/** 无 icon 时的单字标记。 */
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
