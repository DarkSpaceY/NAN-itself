// The single window in the content area: native chrome + scaled
// viewport.
//
// The chrome does not scale (breadcrumbs stay readable on narrow
// windows); inside, the context is always DESIGN_W wide, scaled down
// to the available size by the parent's transform.

import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

import type { AgentStatus } from "../kernel/agents";
import type { WindowRef } from "../kernel/queue";
import { DESIGN_W, fitFrame } from "../kernel/scale";
import { WindowIcon } from "../surfaces/registry";

export interface Crumb {
  id: string;
  label: string;
  status: AgentStatus;
}

interface PaneProps {
  win: WindowRef;
  status: AgentStatus | null;
  crumbs?: Crumb[];
  onCrumb?: (id: string) => void;
  onClose?: () => void;
  children: ReactNode;
}

export function Pane({
  win,
  status,
  crumbs,
  onCrumb,
  onClose,
  children,
}: PaneProps) {
  const viewport = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const el = viewport.current;
    if (!el) return;

    const measure = () => setBox({ w: el.clientWidth, h: el.clientHeight });
    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const frame = fitFrame(box.w, box.h);
  const gutter = Math.max(0, (box.w - DESIGN_W * frame.scale) / 2);

  return (
    <section className="pane" data-status={status ?? undefined}>
      <header className="pane-head">
        <span
          className="dot"
          data-status={status ?? "idle"}
          aria-hidden="true"
        />
        <span className="pane-icon">
          <WindowIcon win={win} />
        </span>
        <span className="pane-title">{win.title}</span>

        {crumbs && crumbs.length > 1 && (
          <nav className="crumbs" aria-label="agent path">
            {crumbs.map((crumb, i) => (
              <span key={crumb.id} className="crumb">
                {i > 0 && <span className="crumb-sep">/</span>}
                <button
                  type="button"
                  className="crumb-btn"
                  data-current={i === crumbs.length - 1 ? "true" : undefined}
                  onClick={() => onCrumb?.(crumb.id)}
                >
                  <span
                    className="dot"
                    data-status={crumb.status}
                    aria-hidden="true"
                  />
                  {crumb.label}
                </button>
              </span>
            ))}
          </nav>
        )}

        {onClose && (
          <button
            type="button"
            className="pane-close"
            aria-label={`close ${win.title}`}
            onClick={onClose}
          >
            ×
          </button>
        )}
      </header>

      <div className="viewport" ref={viewport}>
        {box.w > 0 && (
          <div
            className="frame"
            style={{
              width: frame.width,
              height: frame.height,
              transform: `translateX(${gutter}px) scale(${frame.scale})`,
              transformOrigin: frame.transformOrigin,
            }}
          >
            {children}
          </div>
        )}
      </div>
    </section>
  );
}
