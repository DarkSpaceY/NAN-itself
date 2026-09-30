import { describe, it, expect } from "vitest";
import { deriveFaces } from "../src/stage/faces";
import type { Item, OutputItem, RecordItem } from "../src/store";

function output(id: string, ts: number, state: OutputItem["state"] = "done"): OutputItem {
  return { kind: "output", id, seq: ts, ts, state, text: id };
}

function record(id: string, ts: number, state: RecordItem["state"] = "done"): RecordItem {
  return {
    kind: "record",
    id,
    seq: ts,
    ts,
    category: "tool_call",
    payload: {},
    state,
  };
}

describe("deriveFaces", () => {
  it("no material yields no faces", () => {
    expect(deriveFaces([])).toEqual([]);
  });

  it("only the latest output becomes the single resident face", () => {
    const faces = deriveFaces([output("o1", 1), output("o2", 2)]);
    expect(faces).toHaveLength(1);
    expect(faces[0]).toMatchObject({ id: "o2", kind: "resident" });
  });

  it("latest output is picked by ts, not by array order", () => {
    const faces = deriveFaces([output("new", 9), output("old", 1)]);
    expect(faces[0]).toMatchObject({ id: "new", kind: "resident" });
  });

  it("every record becomes a temp face", () => {
    const faces = deriveFaces([record("r1", 1), record("r2", 2)]);
    expect(faces).toHaveLength(2);
    expect(faces.map((f) => f.kind)).toEqual(["temp", "temp"]);
  });

  it("resident outweighs temp faces so it takes the primary slot", () => {
    const faces = deriveFaces([record("r1", 5), output("o1", 6)]);
    const resident = faces.find((f) => f.kind === "resident")!;
    const temp = faces.find((f) => f.kind === "temp")!;
    expect(resident.weight).toBeGreaterThan(temp.weight);
  });

  it("fades out only temp faces when over capacity", () => {
    const items: Item[] = [
      output("o1", 100),
      ...Array.from({ length: 6 }, (_, i) => record(`r${i}`, i)),
    ];
    const faces = deriveFaces(items);
    // 常驻面 + 6 个临时面，容量 5 → 至少有一个临时面会被判淡出
    expect(faces.filter((f) => f.kind === "resident")).toHaveLength(1);
    expect(faces.filter((f) => f.kind === "temp")).toHaveLength(6);
  });
});
