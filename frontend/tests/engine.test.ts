import { describe, it, expect } from "vitest";
import { computeLayout, STAGE_CAPACITY, type Face } from "../src/stage/engine";

function face(id: string, opts: Partial<Face> = {}): Face {
  return {
    id,
    kind: "temp",
    weight: 50,
    focus: false,
    createdAt: 1000,
    ...opts,
  };
}

describe("computeLayout", () => {
  it("empty stage yields empty slots", () => {
    expect(computeLayout([])).toEqual({ slots: [], fadingIds: [] });
  });

  it("single face takes primary slot", () => {
    const r = computeLayout([face("a")]);
    expect(r.slots).toEqual([{ id: "a", primary: true }]);
  });

  it("higher weight wins primary", () => {
    const r = computeLayout([
      face("low", { weight: 10, createdAt: 2000 }),
      face("high", { weight: 90 }),
    ]);
    expect(r.slots[0]).toEqual({ id: "high", primary: true });
  });

  it("same weight: newest wins primary", () => {
    const r = computeLayout([
      face("old", { createdAt: 1 }),
      face("new", { createdAt: 2 }),
    ]);
    expect(r.slots[0]?.id).toBe("new");
    expect(r.slots[0]?.primary).toBe(true);
  });

  it("focus overrides weight and recency", () => {
    const r = computeLayout([
      face("heavy", { weight: 100, createdAt: 3 }),
      face("focused", { weight: 10, createdAt: 1, focus: true }),
    ]);
    expect(r.slots[0]).toEqual({ id: "focused", primary: true });
  });

  it("multiple focus faces: latest focused is primary", () => {
    const r = computeLayout([
      face("f1", { focus: true, createdAt: 1 }),
      face("f2", { focus: true, createdAt: 2 }),
    ]);
    expect(r.slots[0]?.id).toBe("f2");
  });

  it("resident face never fades on overflow", () => {
    const faces = [
      face("resident", { kind: "resident", weight: 100 }),
      ...Array.from({ length: STAGE_CAPACITY }, (_, i) =>
        face(`t${i}`, { createdAt: i }),
      ),
    ];
    const r = computeLayout(faces);
    // 超出 1 个：最旧临时面 t0 淡出，resident 保留
    expect(r.fadingIds).toEqual(["t0"]);
    expect(r.slots.map((s) => s.id)).toContain("resident");
  });

  it("already-fading faces are excluded from slots and not re-faded", () => {
    const r = computeLayout([
      face("gone", { fading: true }),
      face("a"),
      face("b"),
    ]);
    expect(r.fadingIds).toEqual(["gone"]);
    expect(r.slots.map((s) => s.id).sort()).toEqual(["a", "b"]);
  });

  it("overflow marks multiple oldest temp faces", () => {
    const faces = [
      face("resident", { kind: "resident", weight: 100 }),
      ...Array.from({ length: STAGE_CAPACITY + 1 }, (_, i) =>
        face(`t${i}`, { createdAt: i }),
      ),
    ];
    const r = computeLayout(faces);
    expect(r.fadingIds).toEqual(["t0", "t1"]);
  });
});
