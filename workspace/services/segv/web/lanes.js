// The lineage gutter: lane layout and its SVG, with no DOM access, so the
// geometry can be tested with `node --test`.
//
// Rows are in global time order, so this is a git graph, not a tree: a
// branch stays open from the row that first shows the agent until the row
// that carries its last recorded turn. An agent born before the window is
// drawn as a lane already in flight; one whose last turn is still ahead
// simply runs off the bottom -- nothing is marked dead until the archive
// says so.
//
// Capacity is fixed. Past it, agents share one folded lane drawn dashed:
// the graph never reuses a live column, so it never draws a branch to the
// wrong place. A folded agent keeps its lane until it closes, which keeps
// the lines continuous.

export const LANE_GAP = 8;
export const LANE_X0 = 6;
export const LANE_CAP = 12; // solid lanes, one per agent
export const FOLD_COL = LANE_CAP; // everything past the cap shares this one
export const GUTTER = LANE_X0 + (LANE_CAP + 1) * LANE_GAP;

// Open lanes collapsed to one entry per column: folded agents share a
// column, and a shared dashed line is meaningless unless the column is
// drawn once.
function snapshot(open) {
  const columns = new Map();

  for (const entry of open) {
    const current = columns.get(entry.col);

    if (current) current.agents.push(entry.agent);
    else columns.set(entry.col, { col: entry.col, folded: entry.folded, agents: [entry.agent] });
  }

  return [...columns.values()].sort((a, b) => a.col - b.col);
}

export function layoutLanes({ rows, agents, winFrom, isHidden, capacity = LANE_CAP }) {
  const open = []; // [{ agent, col, folded }]
  const lanes = new Array(rows.length);

  const free = (col) => !open.some((entry) => entry.col === col);

  const firstFree = () => {
    for (let col = 0; col < capacity; col += 1) if (free(col)) return col;

    return -1;
  };

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const hidden = isHidden(row);
    const meta = agents.get(row.agent_id);
    const before = snapshot(open);

    let entry = open.find((candidate) => candidate.agent === row.agent_id);
    let birth = null;
    let elbow = null;

    if (!entry) {
      const parent = open.find((candidate) => candidate.agent === row.parent_id) || null;
      const depth = meta ? meta.depth || 0 : row.depth || 0;

      // A child prefers the lane right after its parent; a folded parent
      // has no column to follow, so depth decides.
      const preferred = parent && !parent.folded
        ? parent.col + 1
        : Math.min(depth, capacity - 1);

      let col = free(preferred) ? preferred : firstFree();
      let folded = false;

      if (col < 0) {
        col = FOLD_COL;
        folded = true;
      }

      entry = { agent: row.agent_id, col, folded };
      open.push(entry);
      open.sort((a, b) => a.col - b.col);

      // Born inside the window: branch off the parent's lane. Born before
      // it: the lane was already running when this window opened, so it
      // enters from the top edge instead.
      const older = meta && meta.first != null && meta.first < winFrom - 0.5;
      birth = older ? "carried" : "born";
      elbow = parent ? parent.col : null;
    }

    const count = meta ? meta.count : row.seq + 1;
    const closes = row.seq >= count - 1;
    const dead = closes && (!meta || meta.last_death);
    const col = entry.col;

    if (closes) open.splice(open.indexOf(entry), 1);

    lanes[i] = {
      agent: row.agent_id,
      hidden,
      col,
      folded: entry.folded,
      birth,
      elbow,
      dead,
      before,
      after: snapshot(open),
    };
  }

  return lanes;
}

const laneX = (col) => LANE_X0 + col * LANE_GAP;

export function laneSvg(lane, path, rowHeight) {
  if (!lane) return "";

  const mid = rowHeight / 2;
  const parts = [];

  const line = (col, y1, y2, agents) => {
    const hot = agents.some((agent) => path.has(agent)) ? " hot" : "";
    const folded = col === FOLD_COL ? " folded" : "";

    parts.push(`<path class="lane${hot}${folded}" d="M${laneX(col)} ${y1} L${laneX(col)} ${y2}"/>`);
  };

  for (const entry of lane.before) line(entry.col, 0, mid, entry.agents);

  if (lane.birth === "carried") {
    line(lane.col, 0, mid, [lane.agent]);
  } else if (lane.birth === "born") {
    if (lane.elbow == null || lane.elbow === lane.col) {
      // No parent in view, or the child shares the parent's lane: a
      // straight drop, never a curve that doubles back on itself.
      line(lane.col, 0, mid, [lane.agent]);
    } else {
      const px = laneX(lane.elbow);
      const nx = laneX(lane.col);
      const hot = path.has(lane.agent) ? " hot" : "";
      const folded = lane.col === FOLD_COL || lane.elbow === FOLD_COL ? " folded" : "";

      parts.push(
        `<path class="lane${hot}${folded}" d="M${px} 0 L${px} ${mid - 5} Q${px} ${mid} ${px + 5} ${mid} L${nx - 4} ${mid}"/>`,
      );
    }
  }

  const nodeClass = (lane.dead ? "node dead" : "node") + (lane.folded ? " folded" : "");

  parts.push(`<circle class="${nodeClass}" cx="${laneX(lane.col)}" cy="${mid}" r="${lane.dead ? 3.2 : 3.4}"/>`);

  for (const entry of lane.after) line(entry.col, mid, rowHeight, entry.agents);

  const folded = lane.after
    .filter((entry) => entry.folded)
    .reduce((total, entry) => total + entry.agents.length, 0);

  const title = folded
    ? `<title>${folded} branch${folded === 1 ? "" : "es"} folded into one lane</title>`
    : "";

  return (
    `<svg width="${GUTTER}" height="${rowHeight}" viewBox="0 0 ${GUTTER} ${rowHeight}" aria-hidden="true">` +
    `${title}${parts.join("")}</svg>`
  );
}
