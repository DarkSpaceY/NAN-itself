import assert from "node:assert/strict";
import test from "node:test";

import {
  FOLD_COL,
  GUTTER,
  LANE_CAP,
  LANE_GAP,
  LANE_X0,
  laneSvg,
  layoutLanes,
} from "../lanes.js";

const row = (agent, seq, parent = "root", depth = 1) => ({
  agent_id: agent,
  parent_id: parent,
  seq,
  depth,
  summary: "",
});

const agentsOf = (count, turns = 2) =>
  new Map(
    Array.from({ length: count }, (_, i) => [
      `a${i}`,
      { depth: 1, count: turns, first: 100 },
    ]),
  );

const layout = (rows, agents) =>
  layoutLanes({ rows, agents, winFrom: 100, isHidden: () => false });

const concurrent = (count) => {
  const rows = [];

  for (let i = 0; i < count; i += 1) rows.push(row(`a${i}`, 0));
  for (let i = 0; i < count; i += 1) rows.push(row(`a${i}`, 1));

  return { rows, agents: agentsOf(count) };
};

test("the gutter reserves one folded column past the cap", () => {
  assert.equal(GUTTER, LANE_X0 + (LANE_CAP + 1) * LANE_GAP);
});

test("the agent past the cap folds instead of sharing a live column", () => {
  const { rows, agents } = concurrent(13);
  const lanes = layout(rows, agents);

  assert.equal(lanes[12].folded, true);
  assert.equal(lanes[12].col, FOLD_COL);

  const open = lanes[12].after.filter((entry) => !entry.folded);

  assert.equal(open.length, LANE_CAP);
  assert.equal(new Set(open.map((entry) => entry.col)).size, LANE_CAP);
});

test("a lane keeps its column for the agent's whole life", () => {
  const { rows, agents } = concurrent(13);
  const lanes = layout(rows, agents);

  for (let i = 0; i < 13; i += 1) {
    assert.equal(lanes[i].col, lanes[13 + i].col, `a${i} moved lanes`);
  }
});

test("a freed column is reused by the next agent", () => {
  const rows = [
    row("a0", 0),
    row("a0", 1),
    row("a1", 0),
    row("a1", 1),
  ];

  const lanes = layout(rows, agentsOf(2));

  assert.notEqual(lanes[2].col, FOLD_COL);
  assert.equal(lanes[2].col, lanes[0].col);
});

test("folded agents share one dashed line and are counted once", () => {
  const { rows, agents } = concurrent(14);
  const lanes = layout(rows, agents);

  assert.equal(lanes[13].folded, true);

  const folded = lanes[13].after.filter((entry) => entry.folded);

  assert.equal(folded.length, 1);
  assert.equal(folded[0].agents.length, 2);

  const svg = laneSvg(lanes[13], new Set(), 30);

  assert.match(svg, /lane[^"]*folded/);
  assert.match(svg, /2 branches folded into one lane/);
});

test("a child sharing its parent's column drops straight down", () => {
  const lane = {
    agent: "child",
    hidden: false,
    col: 3,
    folded: false,
    birth: "born",
    elbow: 3,
    dead: false,
    before: [],
    after: [],
  };

  const svg = laneSvg(lane, new Set(), 30);

  assert.ok(!svg.includes("Q"));
  assert.match(svg, /M30 0 L30 15/);
});

test("a child on its own column still gets an elbow", () => {
  const lane = {
    agent: "child",
    hidden: false,
    col: 3,
    folded: false,
    birth: "born",
    elbow: 1,
    dead: false,
    before: [],
    after: [],
  };

  assert.match(laneSvg(lane, new Set(), 30), /Q/);
});
