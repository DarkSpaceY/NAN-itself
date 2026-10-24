// Service window: iframe + sandbox, no same-origin privileges at all.
//
// The sandbox grants scripts and forms only: no allow-same-origin, no
// popups, no top-level navigation. The permissions policy denies every
// capability; media access requested by the pane itself is futile.

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
