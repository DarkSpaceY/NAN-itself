// Service window: iframe + sandbox, no same-origin privileges at all.
//
// The sandbox grants scripts and forms only: no allow-same-origin, no
// popups, no top-level navigation. The permissions policy denies every
// capability; media access requested by the pane itself is futile.
//
// No desktop bridge: the panel is a plain loopback page that talks to
// its own backend directly (fetch / SSE / WebSocket), and the backend
// allows that origin. The desktop only hands it a url.

import type { WindowRef } from "../../kernel/queue";

const SANDBOX = "allow-scripts allow-forms";
const ALLOW =
  "camera 'none'; microphone 'none'; clipboard-read 'none'; clipboard-write 'none'; fullscreen 'none'";

interface ServiceSurfaceProps {
  win: WindowRef;
}

export function ServiceSurface({ win }: ServiceSurfaceProps) {
  if (!win.url) return <div className="service-surface empty">no url</div>;

  return (
    <div className="service-surface">
      <iframe
        src={win.url}
        title={win.title}
        sandbox={SANDBOX}
        allow={ALLOW}
        referrerPolicy="no-referrer"
      />
    </div>
  );
}
