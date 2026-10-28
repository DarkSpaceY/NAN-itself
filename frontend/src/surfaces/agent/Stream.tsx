// The visible timeline of one agent.
//
// Follows the output tail: scrolling up stops following; coming back
// near the bottom resumes it. version comes from the hub and
// increments once per applied event.

import { useLayoutEffect, useRef } from "react";

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

  useLayoutEffect(() => {
    const el = scroller.current;
    if (!el || !follow.current) return;
    el.scrollTop = el.scrollHeight;
  }, [version]);

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
        </section>
      ))}
    </div>
  );
}
