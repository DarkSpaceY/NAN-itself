// rail: the only active-window switcher, one chip per window.
//
// 48px cannot fit a title, so an icon is required; services without an
// icon degrade to their initial. Chip order is fixed (registration
// order) and never re-sorted by switching.

import type { WindowRef } from "../kernel/queue";
import { WindowIcon } from "../surfaces/registry";

interface RailProps {
  ids: string[];
  windows: Map<string, WindowRef>;
  activeId: string | null;
  onFocus: (id: string) => void;
}

export function Rail({ ids, windows, activeId, onFocus }: RailProps) {
  return (
    <nav className="rail" aria-label="windows">
      {ids.map((id) => {
        const win = windows.get(id);
        if (!win) return null;

        return (
          <button
            key={id}
            type="button"
            className="chip"
            data-role={id === activeId ? "active" : "idle"}
            title={win.title}
            aria-label={win.title}
            aria-current={id === activeId ? "true" : undefined}
            onClick={() => onFocus(id)}
          >
            <WindowIcon win={win} />
            {win.unread ? <span className="chip-badge">{win.unread}</span> : null}
          </button>
        );
      })}
    </nav>
  );
}
