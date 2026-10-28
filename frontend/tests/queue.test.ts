import { describe, expect, it } from "vitest";

import {
  close,
  emptyQueue,
  focus,
  isActive,
  register,
  unregister,
  type QueueState,
} from "../src/kernel/queue";

/** 三个窗口都已注册（agent 在表中第一）。 */
const base = (): QueueState => ({ order: ["agent", "diff", "tests"], active: "agent" });

describe("queue", () => {
  it("registers new windows without stealing the content slot", () => {
    const state = register(base(), "plan");
    expect(state.order).toEqual(["agent", "diff", "tests", "plan"]);
    expect(state.active).toBe("agent");
  });

  it("registers the builtin panel first", () => {
    const state = register(emptyQueue(), "agent", true);
    expect(state.order).toEqual(["agent"]);
    expect(state.active).toBeNull();
  });

  it("focus swaps the content slot and never reorders the rail", () => {
    const state = focus(base(), "tests");
    expect(state.order).toEqual(["agent", "diff", "tests"]);
    expect(state.active).toBe("tests");
  });

  it("focus is idempotent, so clicking twice lands where you started", () => {
    const once = focus(base(), "diff");
    const twice = focus(once, "diff");
    expect(twice).toEqual(once);
    expect(isActive(twice, "diff")).toBe(true);
  });

  it("switching A -> B -> A restores the original content", () => {
    let state = focus(base(), "diff");
    state = focus(state, "tests");
    state = focus(state, "agent");
    expect(state.active).toBe("agent");
    expect(state.order).toEqual(["agent", "diff", "tests"]);
  });

  it("close only empties the content slot, never the rail", () => {
    const state = close(base(), "agent");
    expect(state.order).toEqual(["agent", "diff", "tests"]);
    expect(state.active).toBeNull();
  });

  it("close is a no-op for a window that is not active", () => {
    expect(close(base(), "diff")).toEqual(base());
  });

  it("unregister removes the window and clears it if active", () => {
    const state = unregister(base(), "agent");
    expect(state.order).toEqual(["diff", "tests"]);
    expect(state.active).toBeNull();
  });

  it("unregistering an inactive window keeps the content slot", () => {
    const state = unregister(base(), "tests");
    expect(state.order).toEqual(["agent", "diff"]);
    expect(state.active).toBe("agent");
  });

  it("reopening a closed window puts it back in the slot", () => {
    const state = focus(close(base(), "agent"), "agent");
    expect(state.active).toBe("agent");
  });
});
