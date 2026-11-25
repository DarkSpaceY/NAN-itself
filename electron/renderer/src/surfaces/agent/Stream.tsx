// The visible timeline of one agent.
//
// Follows the output tail: scrolling up stops following; coming back
// near the bottom resumes it. version comes from the hub and
// increments once per applied event.

import { useCallback, useLayoutEffect, useRef } from "react";

import type { Round } from "../../kernel/history/parts";
import { PartView } from "./parts";

const FOLLOW_SLACK = 48;

interface StreamProps {
  rounds: readonly Round[];
  version: number;
  onDrill?: (agentId: string) => void;
  emptyHint?: string;
}

export function Stream({ rounds, version, onDrill, emptyHint }: StreamProps) {
  const scroller = useRef<HTMLDivElement>(null);
  const follow = useRef(true);

  const pinToTail = useCallback(() => {
    const el = scroller.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, []);

  // New output: only follow while the user is parked at the bottom.
  useLayoutEffect(() => {
    if (follow.current) pinToTail();
  }, [version, pinToTail]);

  // The pane can change height without a new event (window resize, the
  // composer growing). Re-pin then too, otherwise the tail silently
  // drops below the fold and the last scroll position looks stuck.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el || typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver(() => {
      if (follow.current) pinToTail();
    });

    observer.observe(el);

    return () => observer.disconnect();
  }, [pinToTail]);

  return (
    <div
      className="stream"
      ref={scroller}
      onScroll={(event) => {
        const el = event.currentTarget;
        follow.current =
          el.scrollHeight - el.scrollTop - el.clientHeight < FOLLOW_SLACK;
      }}
    >
      {rounds.length === 0 && (
        <div className="stream-empty">{emptyHint ?? "waiting for output…"}</div>
      )}

      {rounds.map((round) => (
        <section className="round" key={`${round.index}-${round.ts}`}>
          {round.parts.map((part) => (
            <PartView key={part.id} part={part} onDrill={onDrill} />
          ))}

          {/* Whole round failed: a red error line follows the partial output; once the retry succeeds the round is rolled back */}
          {round.outcome === "failed" && round.detail && (
            <div className="round-error">
              <span className="mark">×</span> {round.detail}
            </div>
          )}

          {/* Parked, not lost: the partial content stays until the turn continues or the agent exits */}
          {round.outcome === "interrupted" && round.detail && (
            <div className="round-interrupted">
              <span className="mark">‖</span> {round.detail}
            </div>
          )}
        </section>
      ))}
    </div>
  );
}
