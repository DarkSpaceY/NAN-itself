// 内建 agent 面板：一次只显示一条流，导航靠面包屑钻入。
//
// 这里不拥有焦点状态：焦点栈在 Shell（每个标签页独立），surface
// 只按传入的 agentId 渲染。composer 也不在这里 —— 它属于 footer。

import type { Hub } from "../../kernel/history/hub";
import { Stream } from "./Stream";

interface AgentSurfaceProps {
  hub: Hub;
  agentId: string | null;
  version: number;
  onDrill: (agentId: string) => void;
}

export function AgentSurface({
  hub,
  agentId,
  version,
  onDrill,
}: AgentSurfaceProps) {
  const rounds = agentId ? (hub.mapper(agentId)?.getRounds() ?? []) : [];
  const evicted = rounds.length === 0 && hub.node(agentId)?.status === "exited";

  return (
    <div className="agent-surface">
      <Stream
        rounds={rounds}
        version={version}
        onDrill={onDrill}
        emptyHint={evicted ? "history evicted" : undefined}
      />
    </div>
  );
}
