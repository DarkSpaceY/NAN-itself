// Service window: iframe + sandbox, no same-origin privileges at all.
//
// The sandbox grants scripts and forms only: no allow-same-origin, no
// popups, no top-level navigation. The permissions policy denies every
// capability; media access requested by the pane itself is futile.
//
// The panel is a plain loopback page that talks to its own backend
// directly (fetch / SSE / WebSocket), and the backend allows that
// origin. The desktop hands it a url and the theme, nothing else.

import { useCallback, useEffect, useRef } from "react";

import type { WindowRef } from "../../kernel/queue";
import type { Theme } from "../../shell/theme";

const SANDBOX = "allow-scripts allow-forms";
const ALLOW =
  "camera 'none'; microphone 'none'; clipboard-read 'none'; clipboard-write 'none'; fullscreen 'none'";

interface ServiceSurfaceProps {
  win: WindowRef;
  theme: Theme;
}

export function ServiceSurface({ win, theme }: ServiceSurfaceProps) {
  const frame = useRef<HTMLIFrameElement>(null);

  // The panel is cross-origin (opaque origin), so postMessage is the
  // only channel; it carries no other state.
  const pushTheme = useCallback(() => {
    frame.current?.contentWindow?.postMessage({ type: "nan:theme", theme }, "*");
  }, [theme]);

  useEffect(() => {
    pushTheme();
  }, [pushTheme]);

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
        onLoad={pushTheme}
      />
    </div>
  );
}
