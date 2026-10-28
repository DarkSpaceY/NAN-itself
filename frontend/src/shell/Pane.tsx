// The single window in the content area: native chrome over a
// viewport that hands the surface the size the pane actually has.

import type { ReactNode } from "react";

import type { AgentStatus } from "../kernel/agents";
import type { WindowRef } from "../kernel/queue";
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

      <div className="viewport">
        <div className="frame">{children}</div>
      </div>
    </section>
  );
}
