// footer: the composer plus one line of global status.
//
// The composer appears only while the main area is the agent pane; the
// status line is always present and never scrolls. No derived metrics
// here -- every number comes from the event stream or hello.

import type { Connection } from "../kernel/store";
import type { InputPart } from "../kernel/history/parts";
import type { Transport } from "../transport/ws";
import { Composer } from "../surfaces/agent/Composer";
import type { Theme } from "./theme";

interface FooterProps {
  transport: Transport;
  paused: boolean;
  targetLabel: string | null;
  queued: readonly InputPart[];
  connection: Connection;
  seq: number;
  panes: number;
  running: number;
  turn: number;
  model: string | null;
  theme: Theme;
  onToggleTheme: () => void;
}

export function Footer({
  transport,
  paused,
  targetLabel,
  queued,
  connection,
  seq,
  panes,
  running,
  turn,
  model,
  theme,
  onToggleTheme,
}: FooterProps) {
  return (
    <footer className="foot">
      <Composer
        transport={transport}
        paused={paused}
        targetLabel={targetLabel}
        queued={queued}
      />

      <div className="statusline">
        <span className="status-left">
          <span className="conn" data-conn={connection} aria-hidden="true" />
          {transport.label}
          <span className="mono">seq {seq}</span>
          <span className="sep">·</span>
          {panes} pane
          <span className="sep">·</span>
          {running} running
          <span className="sep">·</span>
          turn {turn}
          {paused && <span className="paused">paused</span>}
        </span>

        <span className="status-right">
          {model && <span className="mono">{model}</span>}
          <button
            type="button"
            className="ghost theme-btn"
            onClick={onToggleTheme}
          >
            {theme}
          </button>
        </span>
      </div>
    </footer>
  );
}
