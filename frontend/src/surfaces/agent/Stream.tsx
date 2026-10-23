// 一个 agent 的可见时间线。
//
// 跟随输出尾部：用户向上滚动就停止跟随，回到底部附近自动恢复。
// version 来自 hub，事件每落地一次就递增。

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

          {/* 整轮失败：红色错误行跟在半截输出后面，重试成功后整轮被回退 */}
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
