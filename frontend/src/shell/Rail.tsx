// rail：唯一的活动窗口切换器，每个窗口一个 chip。
//
// 48px 放不下标题，所以图标是必需的；没有 icon 的服务退化为首字母。
// chip 顺序固定（注册序），不会因为切换而重排。

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
