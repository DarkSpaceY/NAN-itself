// Desktop shell: layout and window switching.
//
// One content slot + one rail (kernel/queue). This only holds the
// window set, the currently active window, and the breadcrumb focus
// inside the agent pane. Surfaces know nothing of the queue.

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

import type { AgentNode } from "../kernel/agents";
import type { Hub } from "../kernel/history/hub";
import {
  close as closeWindow,
  emptyQueue,
  focus as focusWindow,
  register as registerWindow,
  unregister as unregisterWindow,
  type QueueState,
  type WindowRef,
} from "../kernel/queue";
import type { SurfaceSpec } from "../kernel/protocol";
import { AgentSurface } from "../surfaces/agent/AgentSurface";
import {
  AGENT_ID,
  builtinWindow,
  serviceWindow,
  serviceWindowId,
} from "../surfaces/registry";
import { ServiceSurface } from "../surfaces/service/ServiceSurface";
import type { Transport } from "../transport/ws";
import { Footer } from "./Footer";
import { Pane, type Crumb } from "./Pane";
import { Rail } from "./Rail";
import { applyTheme, loadTheme, saveTheme, toggleTheme, type Theme } from "./theme";

function initialQueue(): QueueState {
  return focusWindow(registerWindow(emptyQueue(), AGENT_ID, true), AGENT_ID);
}

function crumbLabel(node: AgentNode): string {
  return node.depth === 0 ? "root" : node.agent_id.slice(-4);
}

export function Shell({ hub, transport }: { hub: Hub; transport: Transport }) {
  const [queue, setQueue] = useState<QueueState>(initialQueue);
  const [windows, setWindows] = useState<Map<string, WindowRef>>(
    () => new Map([[AGENT_ID, builtinWindow()]]),
  );
  const [focusAgentId, setFocusAgentId] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>(loadTheme);

  // -- subscription ---------------------------------------------------

  const subscribeHub = useCallback((fn: () => void) => hub.subscribe(fn), [hub]);
  const hubVersion = useCallback(() => hub.getVersion(), [hub]);
  const version = useSyncExternalStore(subscribeHub, hubVersion);

  const subscribeGlobal = useCallback((fn: () => void) => hub.global.subscribe(fn), [hub]);
  const globalSnapshot = useCallback(() => hub.global.getState(), [hub]);
  const global = useSyncExternalStore(subscribeGlobal, globalSnapshot);

  useEffect(() => {
    applyTheme(theme);
    saveTheme(theme);
  }, [theme]);

  // -- service registration -------------------------------------------

  const addSurface = useCallback((spec: SurfaceSpec) => {
    const win = serviceWindow(spec);
    setWindows((previous) => {
      if (previous.has(win.id)) return previous;
      const next = new Map(previous);
      next.set(win.id, win);
      return next;
    });
    setQueue((previous) => registerWindow(previous, win.id));
  }, []);

  const removeSurface = useCallback((serviceId: string) => {
    const id = serviceWindowId(serviceId);
    setWindows((previous) => {
      if (!previous.has(id)) return previous;
      const next = new Map(previous);
      next.delete(id);
      return next;
    });
    setQueue((previous) => unregisterWindow(previous, id));
  }, []);

  useEffect(() => {
    transport.start(addSurface, removeSurface);
    return () => transport.stop();
  }, [transport, addSurface, removeSurface]);

  // -- derived --------------------------------------------------------

  const rootId = hub.rootId;
  const activeAgent = focusAgentId && hub.node(focusAgentId) ? focusAgentId : rootId;

  // The agent being viewed is excluded from timeline eviction
  useEffect(() => {
    hub.pin(activeAgent);
  }, [hub, activeAgent]);

  const crumbs: Crumb[] = hub.path(activeAgent).map((node) => ({
    id: node.agent_id,
    label: crumbLabel(node),
    status: node.status,
  }));

  let running = 0;
  for (const node of hub.agents.nodes.values()) {
    if (node.status === "working") running += 1;
  }

  const rootNode = rootId ? hub.node(rootId) : undefined;
  const activeWin = queue.active ? windows.get(queue.active) : undefined;

  const targetLabel =
    activeAgent && rootId && activeAgent !== rootId
      ? `root · viewing ${hub.node(activeAgent)?.agent_id.slice(-4) ?? ""}`
      : null;

  const surfaceFor = (win: WindowRef) =>
    win.kind === "agent" ? (
      <AgentSurface
        hub={hub}
        agentId={activeAgent}
        version={version}
        onDrill={(agentId) => setFocusAgentId(agentId)}
      />
    ) : (
      <ServiceSurface win={win} />
    );

  return (
    <div className="app">
      <div className="stage">
        <div className="content">
          {activeWin ? (
            <Pane
              win={activeWin}
              status={activeWin.kind === "agent" ? (hub.node(activeAgent)?.status ?? null) : null}
              crumbs={activeWin.kind === "agent" ? crumbs : undefined}
              onCrumb={(id) => setFocusAgentId(id)}
              onClose={() => setQueue((previous) => closeWindow(previous, activeWin.id))}
            >
              {surfaceFor(activeWin)}
            </Pane>
          ) : (
            <div className="workspace-empty">
              no open window<span>pick one from the rail</span>
            </div>
          )}
        </div>

        <Rail
          ids={queue.order}
          windows={windows}
          activeId={queue.active}
          onFocus={(id) => setQueue((previous) => focusWindow(previous, id))}
        />
      </div>

      <Footer
        transport={transport}
        paused={global.paused === true}
        targetLabel={targetLabel}
        queued={hub.queued(rootId)}
        connection={global.connection}
        seq={global.lastSeq}
        panes={windows.size}
        running={running}
        turn={rootNode?.turns ?? 0}
        model={global.model}
        theme={theme}
        onToggleTheme={() => setTheme((current) => toggleTheme(current))}
      />
    </div>
  );
}
