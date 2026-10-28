// Built-in agent pane: shows one stream at a time; navigation is
// breadcrumb drill-in.
//
// Focus state is not owned here: the focus stack lives in Shell (one
// per tab) and the surface only renders the agentId passed in. The
// composer is not here either -- it belongs to the footer.

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
