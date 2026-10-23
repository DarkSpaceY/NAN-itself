// 服务窗口：iframe + sandbox，没有任何同源权限。
//
// sandbox 只给脚本与表单；不要 allow-same-origin，不要弹窗，不要
// 顶层导航。权限策略逐项拒绝，媒体能力由面板自己申请也没用。

import { useEffect, useRef } from "react";

import type { WindowRef } from "../../kernel/queue";
import type { Theme } from "../../shell/theme";
import { attachBridge } from "./bridge";

const SANDBOX = "allow-scripts allow-forms";
const ALLOW =
  "camera 'none'; microphone 'none'; clipboard-read 'none'; clipboard-write 'none'; fullscreen 'none'";

interface ServiceSurfaceProps {
  win: WindowRef;
  theme: Theme;
  bootId: string | null;
}

export function ServiceSurface({ win, theme, bootId }: ServiceSurfaceProps) {
  const frame = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    const el = frame.current;
    if (!el || !win.url) return;

    const bridge = attachBridge(
      el,
      {
        service_id: win.serviceId ?? win.id,
        url: win.url,
        title: win.title,
        icon: win.icon,
      },
      theme,
      bootId,
    );

    return () => bridge.destroy();
  }, [win.id, win.url, win.title, win.icon, win.serviceId, theme, bootId]);

  if (!win.url) return <div className="service-surface empty">no url</div>;

  return (
    <div className="service-surface">
      <iframe
        ref={frame}
        src={win.url}
        title={win.title}
        sandbox={SANDBOX}
        allow={ALLOW}
        referrerPolicy="no-referrer"
      />
    </div>
  );
}
