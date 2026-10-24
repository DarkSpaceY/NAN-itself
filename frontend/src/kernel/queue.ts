// Window set + the single active window: the desktop's entire
// navigation state.
//
// Two fields: order (the rail's fixed, registration-ordered
// sequence) and active (the one window occupying the content
// area). No secondary area -- one content slot; switching is
// switching, clicking twice always returns to the same place.
// Position is state; roles are never stored.
//
// Pure functions: state in, new state out, directly unit-testable.

export interface WindowRef {
  id: string;
  /** The built-in agent panel, cannot be unregistered; service windows come from registration. */
  kind: "agent" | "service";
  title: string;
  /** SVG path contents for a 16px icon; falls back to the id's initial. */
  icon?: string;
  /** The service's declared primary color, for optional accents beyond the status dot; empty = follow the theme. */
  serviceId?: string;
  url?: string;
  /** Unread badge on the rail; cleared on activation. */
  unread?: number;
}

export interface QueueState {
  /** The rail's render order. Fixed after registration; chips never jump. */
  order: string[];
  /** The window currently shown in the content area; null = empty workspace. */
  active: string | null;
}

export function emptyQueue(): QueueState {
  return { order: [], active: null };
}

function without(list: string[], id: string): string[] {
  return list.filter((x) => x !== id);
}

/** The built-in agent panel is always registered; service windows append in registration order. */
export function register(
  state: QueueState,
  id: string,
  first = false,
): QueueState {
  if (state.order.includes(id)) return state;
  return {
    order: first ? [id, ...state.order] : [...state.order, id],
    // Registration never steals the content slot: new windows wait on the rail
    active: state.active,
  };
}

/** Service unregistration: the window truly disappears. If the active window goes, back to the empty workspace. */
export function unregister(state: QueueState, id: string): QueueState {
  return {
    order: without(state.order, id),
    active: state.active === id ? null : state.active,
  };
}

/** Focus: put this window in the single content slot. */
export function focus(state: QueueState, id: string): QueueState {
  if (!state.order.includes(id) || state.active === id) return state;
  return { ...state, active: id };
}

/** Close: fall back to the rail, leaving the workspace empty (the window is not destroyed). */
export function close(state: QueueState, id: string): QueueState {
  if (state.active !== id) return state;
  return { ...state, active: null };
}

export function isActive(state: QueueState, id: string): boolean {
  return state.active === id;
}
