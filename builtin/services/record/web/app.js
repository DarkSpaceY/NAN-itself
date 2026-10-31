// The record ledger: one global time order over every agent, plus a
// two-level time navigator. Reads the archive; never writes.
//
// Two scales, because the archive spans minutes to years, and each
// one does exactly one job:
//
//   archive strip  position, and nothing else. Dragging moves the
//                  window along the archive; a click outside the
//                  band jumps there; double-click shows everything.
//                  Size belongs to the window strip, which zooms on
//                  the wheel -- so this strip answers one question.
//   window strip   a time cursor. It stands for a time, not for the
//                  ledger's top row, so every turn in the window is
//                  reachable -- the last screen included. The nearest
//                  turn is marked; dragging the cursor pulls the ledger
//                  along, and scrolling the ledger leaves the cursor
//                  be until its marked turn would leave the view,
//                  then the view follows it.

const $ = (sel, root = document) => root.querySelector(sel);

// A scroll the page caused (reload, reveal) is not a scroll the reader
// made: for a moment after one, the cursor must not be re-anchored to
// whatever row happens to be at the edge.
let settleUntil = 0;
const settle = (ms = 120) => {
  settleUntil = performance.now() + ms;
};

const ROW_H = 30;
const GAP_H = 26;
const GAP_SECONDS = 300; // idle longer than this earns a divider
const LANE_GAP = 10;
const LANE_COLS = 8;
const LANE_X0 = 6;
const GUTTER = LANE_X0 + LANE_COLS * LANE_GAP;
const OVERSCAN = 10; // rows rendered beyond the viewport, each way
const PAGE = 300;
const MIN_SPAN = 4; // seconds; tighter than this is not worth showing
const UNIT_SECONDS = { m: 60, h: 3600, d: 86400, w: 604800 };

const state = {
  agents: new Map(),
  axis: { from: 0, to: 1 }, // the whole archive
  win: { from: 0, to: 1 }, // what the ledger is showing
  follow: true, // the window rides the live edge
  whole: true, // the window is the whole archive, so it keeps growing with it
  strip: [], // archive histogram
  stripUnit: "m",
  series: [], // window histogram
  unit: "m",
  rows: [],
  index: new Map(),
  cursor: null,
  done: true,
  total: null,
  loading: false,
  items: [],
  rowItem: [],
  lanes: [],
  at: null, // cursor time; null means "the window start"
  mark: null, // row index the cursor points at
  rendered: { from: 0, to: 0 },
  height: 0,
  filter: "",
  sel: null, // selected ledger key
  path: new Set(), // agents on that row's ancestor path
  drawer: null,
  tab: "summary",
  theme: "light",
};

// ---------------------------------------------------------------- helpers

const pad = (n) => String(n).padStart(2, "0");

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const esc = (text) =>
  String(text ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const clock = (ts) => {
  const d = new Date(ts * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};

const hhmm = (ts) => {
  const d = new Date(ts * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

// The archive is measured in years, so a day is only unambiguous with
// its year: "Jun 28" says nothing about which Jun 28 it was.
const dayLabel = (ts) => {
  const d = new Date(ts * 1000);
  return `${MONTHS[d.getMonth()]} ${d.getDate()}`;
};

const dayStamp = (ts) => `${dayLabel(ts)} ${new Date(ts * 1000).getFullYear()}`;

const dayKey = (ts) => new Date(ts * 1000).toDateString();

const sameYear = (a, b) =>
  new Date(a * 1000).getFullYear() === new Date(b * 1000).getFullYear();

const stamp = (ts) => (ts ? `${dayStamp(ts)} ${hhmm(ts)}` : "—");

const dur = (seconds) => {
  if (seconds == null || Number.isNaN(seconds)) return "";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
};

const spanText = (seconds) => {
  if (seconds < 1) return "1s";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) {
    const h = Math.floor(seconds / 3600);
    return `${h}h ${Math.round((seconds % 3600) / 60)}m`;
  }
  return `${Math.floor(seconds / 86400)}d ${Math.round((seconds % 86400) / 3600)}h`;
};

const compact = (n) => {
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1000).toFixed(n < 1e4 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(1)}M`;
};

const bytes = (n) => {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
};

// A stable, low-chroma hue per agent: enough to tell branches apart
// at a glance, quiet enough to keep eight lanes from turning into
// confetti.
const hueOf = (agentId) => {
  let h = 0;
  for (let i = 0; i < agentId.length; i += 1) h = (h * 31 + agentId.charCodeAt(i)) % 360;
  return h;
};

const keyOf = (row) => `${row.agent_id}:${row.seq}`;

// A tool call is a verb and an object; both belong on one line, not in
// a JSON dump the eye has to parse.
const argLine = (args) => {
  if (args == null) return "";

  // A provider-shaped call carries its arguments as a JSON string.
  if (typeof args === "string") {
    const text = args.trim();

    if (text.startsWith("{")) {
      try {
        return argLine(JSON.parse(text));
      } catch {
        // Not JSON after all; show it as it came.
      }
    }

    return args.length > 64 ? `${args.slice(0, 64)}…` : args;
  }

  if (typeof args !== "object") return String(args);

  return Object.entries(args)
    .slice(0, 3)
    .map(([key, value]) => {
      const text = typeof value === "string" ? value : JSON.stringify(value);
      return `${key}=${text.length > 24 ? `${text.slice(0, 24)}…` : text}`;
    })
    .join(" · ");
};

// A result carries no failure flag: how it went is in its text, and
// shell is the one verb that leads with its status line. Two shell
// failures carry no usable code there -- a command killed at the
// timeout is replaced by a single sentence, and a signal death comes
// back as a negative code -- so they are read from the text as well.
const exitCode = (out) => {
  const match = /^exit code: (-?\d+)/m.exec(out || "");
  return match ? Number(match[1]) : null;
};

const failed = (out) => {
  const code = exitCode(out);
  if (code !== null && code !== 0) return true;
  return /^Timed out after /m.test(out || "");
};

// A call and what came back are one object in the archive: calls are
// positional with results, and `call_id` pairs them when both carry
// ids. One card each, closed until asked, so a turn with six calls
// still reads as a list.
function toolsBlock(turn) {
  const calls = turn.calls || [];
  const results = turn.results || [];

  if (!calls.length && !results.length) return "";

  const byId = new Map();

  for (const result of results) {
    if (result.call_id != null) byId.set(result.call_id, result);
  }

  const cards = calls.map((call, index) => {
    const result = (call.id != null && byId.get(call.id)) || results[index] || null;

    return toolCard(call, result);
  });

  return `<div class="tools"><div class="msg-role">TOOLS</div>${cards.join("")}</div>`;
}

function toolCard(call, result) {
  const name = call.name || (result && result.name) || "call";
  const args = argLine(call.arguments ?? call.args);
  const out = (result && result.output) || "";
  const ok = !failed(out);

  const took =
    result && result.started_at != null && result.ended_at != null
      ? dur(result.ended_at - result.started_at)
      : "";

  const meta = [...(ok ? [] : ["failed"]), took].filter(Boolean).join(" · ");

  // One line of the output before it is opened: enough to tell a
  // result from an error without a click.
  const peek = (out.split("\n").find((line) => line.trim()) || "").trim();

  return (
    `<details class="tool${ok ? "" : " bad"}">` +
    `<summary><span class="tool-name">${esc(name)}</span>` +
    (args ? `<span class="tool-args">${esc(args)}</span>` : "") +
    (peek ? `<span class="tool-peek">${esc(peek.length > 96 ? `${peek.slice(0, 96)}…` : peek)}</span>` : "") +
    (meta ? `<span class="tool-meta">${esc(meta)}</span>` : "") +
    `</summary>` +
    (out ? `<pre class="tool-out">${esc(out)}</pre>` : "") +
    `</details>`
  );
}

const api = async (path) => {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${path}`);
  return response.json();
};

// ------------------------------------------------------------ histogram

// Bars for one range, in whichever unit the server picked. Dense
// units get rectangles on a real axis; "turn" gets a tick per turn,
// which is a barcode of exactly what was recorded.
function barsPath(series, unit, from, to, width, height) {
  if (!series.length || width <= 1) return "";

  const span = Math.max(to - from, 0.001);
  const x = (t) => ((t - from) / span) * width;
  const parts = [];

  if (unit === "turn") {
    const top = height * 0.42;
    for (const point of series) {
      const px = x(point.t);
      if (px < -2 || px > width + 2) continue;
      parts.push(`M${px.toFixed(2)} ${height}h1.4v${-top.toFixed(2)}h-1.4z`);
    }
    return parts.join("");
  }

  const max = Math.max(1, ...series.map((p) => p.n));
  const width0 = Math.max(1, (UNIT_SECONDS[unit] / span) * width);

  for (const point of series) {
    if (!point.n) continue;
    const px = x(point.t);
    if (px + width0 < -2 || px > width + 2) continue;
    const bar = Math.max(1.5, height * (point.n / max));
    parts.push(
      `M${px.toFixed(2)} ${height}h${Math.min(width0, 12).toFixed(2)}v${(-bar).toFixed(2)}h${(-Math.min(width0, 12)).toFixed(2)}z`,
    );
  }

  return parts.join("");
}

function paintStrip() {
  const el = $("#hist-all");
  const width = el.clientWidth || 600;
  $("#bars-all").setAttribute("d", barsPath(state.strip, state.stripUnit, state.axis.from, state.axis.to, width, 19));
  el.setAttribute("viewBox", `0 0 ${width} 19`);
}

function paintSeries() {
  const el = $("#hist-win");
  const width = el.clientWidth || 600;
  $("#bars-win").setAttribute("d", barsPath(state.series, state.unit, state.win.from, state.win.to, width, 44));
  el.setAttribute("viewBox", `0 0 ${width} 44`);

  // Labels every so often, never a crowd.
  const ticks = $("#ticks");
  if (state.unit === "turn" || !state.series.length) {
    ticks.textContent = "";
    return;
  }

  const span = Math.max(state.win.to - state.win.from, 0.001);
  const stride = Math.max(1, Math.ceil(state.series.length / 7));
  const out = [];

  state.series.forEach((point, i) => {
    if (i % stride) return;
    const x = ((point.t - state.win.from) / span) * width;
    if (x < 30 || x > width - 30) return;
    // Days only need the year when the window itself crosses one.
    const label =
      state.unit === "m" || state.unit === "h"
        ? hhmm(point.t)
        : sameYear(state.win.from, state.win.to)
          ? dayLabel(point.t)
          : dayStamp(point.t);
    out.push(`<span style="left:${x.toFixed(1)}px">${label}</span>`);
  });

  ticks.innerHTML = out.join("");
}

// Where the window sits inside the archive -- and the handle for
// changing it.
function paintWindowMark() {
  const el = $("#overview");
  const width = el.clientWidth || 600;
  const span = Math.max(state.axis.to - state.axis.from, 0.001);
  const left = Math.max(0, Math.min(width, ((state.win.from - state.axis.from) / span) * width));
  const right = Math.max(0, Math.min(width, ((state.win.to - state.axis.from) / span) * width));

  // The width is the window's real share of the archive, however
  // thin that gets: the strip only moves the window, and the hit
  // area lives in the pointer code, not in the paint.
  const mark = $("#winmark");
  mark.style.left = `${left.toFixed(1)}px`;
  mark.style.width = `${Math.max(0, right - left).toFixed(1)}px`;

  const whole = state.win.from <= state.axis.from + 0.5 && state.win.to >= state.axis.to - 0.5;
  mark.classList.toggle("full", whole);

  $("#range-from").textContent = stamp(state.win.from);
  $("#range-to").textContent = stamp(state.win.to);
  $("#range-span").textContent = whole
    ? `${spanText(span)} · the whole archive`
    : `${spanText(state.win.to - state.win.from)} of ${spanText(span)}`;
}

// ------------------------------------------------------------- fetching

async function fetchStrip() {
  const width = $("#overview").clientWidth || 600;
  const data = await api(
    `/api/activity?start=${state.axis.from}&end=${state.axis.to}&buckets=${Math.max(60, Math.min(2000, width))}`,
  );
  state.strip = data.series;
  state.stripUnit = data.unit;
  paintStrip();
}

async function fetchSeries() {
  const width = $("#focus").clientWidth || 600;
  const data = await api(
    `/api/activity?start=${state.win.from}&end=${state.win.to}&buckets=${Math.max(60, Math.min(2000, width))}`,
  );
  state.series = data.series;
  state.unit = data.unit;
  paintSeries();
}

// ---------------------------------------------------------------- ledger

async function loadPage(reset) {
  if (state.loading) return;

  if (reset) {
    state.rows = [];
    state.cursor = null;
    state.done = false;
    state.total = null;
    state.rendered = { from: 0, to: 0 };
  }

  if (state.done) return;

  state.loading = true;
  paintFoot();

  try {
    let query = `/api/ledger?start=${state.win.from}&end=${state.win.to}&limit=${PAGE}`;
    if (state.cursor) query += `&cursor=${state.cursor}`;

    const data = await api(query);

    state.rows = state.rows.concat(data.rows);
    state.cursor = data.cursor;
    state.done = data.rows.length < PAGE;
    if (data.total != null) state.total = data.total;

    rebuild();
  } finally {
    state.loading = false;
    paintFoot();
  }
}

// Everything between the loaded rows and the DOM: seconds to offset,
// idle stretches, lane columns. Cheap enough to redo on any change,
// which is what keeps the gutters honest when new turns arrive.
function rebuild() {
  const items = [];
  const rowItem = [];
  let top = 0;
  let prev = null;
  let day = null;

  state.index = new Map();

  for (let i = 0; i < state.rows.length; i += 1) {
    const row = state.rows[i];
    state.index.set(keyOf(row), row);

    if (state.filter && !matchFilter(row)) continue;

    const rowDay = dayKey(row.started_at);

    if (rowDay !== day) {
      items.push({ kind: "day", h: GAP_H, top, label: dayStamp(row.started_at) });
      top += GAP_H;
      day = rowDay;
      prev = null;
    } else if (prev && row.started_at - (prev.ended_at || prev.started_at) > GAP_SECONDS) {
      items.push({ kind: "idle", h: GAP_H, top, label: spanText(row.started_at - prev.ended_at) });
      top += GAP_H;
    }

    // `ri` keeps the row's own index so the lane graph (built per
    // row, not per item) can be looked up without a scan. The same
    // index keys rowItem, which is how a time becomes a scroll offset.
    rowItem[i] = items.length;
    items.push({ kind: "row", h: ROW_H, top, row, ri: i });
    top += ROW_H;
    prev = row;
  }

  state.items = items;
  state.rowItem = rowItem;
  state.height = top;
  state.lanes = layoutLanes();

  settle();

  const rows = $("#rows");
  rows.style.height = `${top}px`;

  state.rendered = { from: 0, to: 0 };

  // Either nothing is recorded here, or the filter emptied the list;
  // both have to clear whatever slice was on screen.
  if (!items.length) {
    rows.style.height = "auto";
    rows.innerHTML = `<div class="empty">${
      state.filter ? "no loaded row matches the filter" : "nothing recorded in this window"
    }</div>`;
    state.mark = null;
    placeCursor(cursorTime());
    return;
  }

  // A fresh window starts at its first turn: the cursor follows the
  // ledger from there.
  if (state.at == null) state.at = state.rows[0].started_at;

  // Marked rows are drawn by rowHtml, so this has to happen before
  // the slice is painted.
  state.mark = markIndexFor(cursorTime());
  placeCursor(cursorTime());

  paintRows();
}

function matchFilter(row) {
  const needle = state.filter;
  return (
    row.agent_id.includes(needle) ||
    String(row.seq) === needle ||
    row.summary.toLowerCase().includes(needle)
  );
}

// The lineage gutter.
//
// Rows are in global time order, so this is a git graph, not a tree:
// a branch is open from the row that first shows the agent until the
// row that carries its last recorded turn. An agent that was born
// before the window is drawn as a lane already in flight, and one
// whose last turn is still ahead simply runs off the bottom --
// nothing is marked dead until the archive says so.
function layoutLanes() {
  const open = []; // [{ agent, col }] keeping column order
  const lanes = new Array(state.rows.length);

  const free = (col) => !open.some((entry) => entry.col === col);

  const firstFree = (from) => {
    for (let col = Math.max(0, from); col < LANE_COLS; col += 1) if (free(col)) return col;
    for (let col = 0; col < LANE_COLS; col += 1) if (free(col)) return col;
    return -1;
  };

  for (let i = 0; i < state.rows.length; i += 1) {
    // Filtered-out rows still hold their lane: the branch is real
    // whether or not the current filter shows it.
    const hidden = state.filter && !matchFilter(state.rows[i]);
    const row = state.rows[i];
    const meta = state.agents.get(row.agent_id);
    const before = open.map((entry) => ({ ...entry }));

    let entry = open.find((c) => c.agent === row.agent_id);
    let birth = null;
    let elbow = null;

    if (!entry) {
      const parent = open.find((c) => c.agent === row.parent_id) || null;
      const depth = meta ? meta.depth || 0 : row.depth || 0;
      const preferred = parent ? parent.col + 1 : Math.min(depth, LANE_COLS - 1);
      const col = free(preferred) ? preferred : Math.max(0, firstFree(0));

      entry = { agent: row.agent_id, col };
      open.push(entry);
      open.sort((a, b) => a.col - b.col);

      // Born inside the window: branch off the parent's lane. Born
      // before it: the lane was already running when this window
      // opened, so it enters from the top edge instead.
      const older = meta && meta.first != null && meta.first < state.win.from - 0.5;
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
      birth,
      elbow,
      dead,
      before,
      after: open.map((c) => ({ ...c })),
    };
  }

  return lanes;
}

function laneSvg(lane) {
  if (!lane) return "";

  const mid = ROW_H / 2;
  const x = (col) => LANE_X0 + col * LANE_GAP;
  const parts = [];

  const line = (col, y1, y2, agent) => {
    const hot = state.path.has(agent) ? " hot" : "";
    parts.push(`<path class="lane${hot}" d="M${x(col)} ${y1} L${x(col)} ${y2}"/>`);
  };

  for (const entry of lane.before) line(entry.col, 0, mid, entry.agent);

  if (lane.birth === "carried") {
    line(lane.col, 0, mid, lane.agent);
  } else if (lane.birth === "born") {
    if (lane.elbow == null) {
      line(lane.col, 0, mid, lane.agent);
    } else {
      const px = x(lane.elbow);
      const nx = x(lane.col);
      const hot = state.path.has(lane.agent) ? " hot" : "";
      parts.push(
        `<path class="lane${hot}" d="M${px} 0 L${px} ${mid - 5} Q${px} ${mid} ${px + 5} ${mid} L${nx - 4} ${mid}"/>`,
      );
    }
  }

  const nodeClass = lane.dead ? "node dead" : "node";
  parts.push(`<circle class="${nodeClass}" cx="${x(lane.col)}" cy="${mid}" r="${lane.dead ? 3.2 : 3.4}"/>`);

  for (const entry of lane.after) line(entry.col, mid, ROW_H, entry.agent);

  return `<svg width="${GUTTER}" height="${ROW_H}" viewBox="0 0 ${GUTTER} ${ROW_H}" aria-hidden="true">${parts.join("")}</svg>`;
}

// The row's own outcome, kept tiny: the ledger is a log, so only the
// unusual characteristics earn a mark and the ordinary run of turns
// stays silent.
function stateMark(row) {
  if (row.errors) return `<span class="mk err" title="${row.errors} failed injection(s)">✕</span>`;
  if (row.retries) return `<span class="mk warn" title="${row.retries} retries">↻${row.retries}</span>`;
  if (row.death) return `<span class="mk dim" title="agent exited: ${esc(row.death)}">○</span>`;
  if (row.origin === "context_cleared") return `<span class="mk dim" title="context cleared">↻</span>`;
  if (row.origin === "first") return `<span class="mk faint" title="first turn">◦</span>`;
  return `<span class="mk"></span>`;
}

function rowHtml(item) {
  const { row } = item;
  const key = keyOf(row);
  const lane = state.lanes[item.ri];
  const selected = state.sel === key ? " selected" : "";
  const marked = state.mark === item.ri ? " marked" : "";
  const danger = row.errors ? " err" : "";
  const seconds = (row.ended_at || row.started_at) - row.started_at;

  return (
    `<div class="row${selected}${marked}${danger}" data-key="${esc(key)}" data-ri="${item.ri}" ` +
    `style="top:${item.top}px;--h:${hueOf(row.agent_id)}">` +
    `<span class="c-gut">${laneSvg(lane)}</span>` +
    `<span class="t">${clock(row.started_at)}</span>` +
    `<span class="a">${stateMark(row)}<span class="nm">${esc(row.agent_id)}</span><span class="sq">${row.seq}</span></span>` +
    `<span class="sum" title="${esc(row.summary)}">${esc(row.summary)}</span>` +
    `<span class="meta"><span class="tok">${row.tokens ? compact(row.tokens) : ""}</span>` +
    `<span class="dur">${dur(seconds)}</span></span>` +
    `</div>`
  );
}

// First item whose bottom edge is below `y`, or items.length when
// every item sits above it. The default matters: scrolling past the
// end must answer "the whole tail", not "the beginning".
function itemIndexAt(y) {
  const items = state.items;
  let lo = 0;
  let hi = items.length - 1;
  let found = items.length;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;

    if (items[mid].top + items[mid].h > y) {
      found = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  return found;
}

function paintRows() {
  if (!state.items.length) return;

  const scroll = $("#scroll");
  const top = Math.max(0, scroll.scrollTop - OVERSCAN * ROW_H);
  const bottom = scroll.scrollTop + scroll.clientHeight + OVERSCAN * ROW_H;
  const from = Math.min(itemIndexAt(top), state.items.length - 1);
  const to = Math.min(state.items.length, itemIndexAt(bottom) + 1);

  if (from === state.rendered.from && to === state.rendered.to) return;

  state.rendered = { from, to };

  // Positions are absolute, so the slice can be replaced wholesale
  // without disturbing the scroll offset.
  const html = state.items
    .slice(from, to)
    .map((item) =>
      item.kind === "row"
        ? rowHtml(item)
        : `<div class="gap ${item.kind}" style="top:${item.top}px"><span>${esc(item.label)}</span></div>`,
    )
    .join("");

  $("#rows").innerHTML = html;
}

function paintFoot() {
  const parts = [];

  if (state.total != null) {
    parts.push(`${state.rows.length} of ≈${state.total} rows`);
  } else {
    parts.push(`${state.rows.length} rows`);
  }

  const agents = new Set(state.rows.map((row) => row.agent_id)).size;
  if (agents) parts.push(`${agents} agents`);

  if (state.filter) parts.push(`filter: ${state.rows.filter(matchFilter).length} match`);

  $("#foot").innerHTML =
    `<span>${esc(parts.join(" · "))}</span><span class="spacer"></span>` +
    `<span>${state.loading ? "loading…" : state.done ? "end of window" : "scroll for more"}</span>`;
}

// ---------------------------------------------------------------- window

const axisTo = () => state.axis.to;

// A drag repaints on every pointer move; refetching the histogram as
// often would be one request per pixel. One deferred fetch keeps the
// bars honest without that traffic.
let seriesTimer = 0;

function paintWindow() {
  paintWindowMark();
  paintSeries();

  // Place the rule by its time, not by a remembered pixel: sliding
  // the window under a fixed time is exactly the motion that should
  // move it along the strip.
  placeCursor(cursorTime());

  clearTimeout(seriesTimer);
  seriesTimer = setTimeout(() => fetchSeries().catch(() => {}), 70);
}

function setWindow(from, to, { reload = true, carry = false } = {}) {
  let start = Math.max(state.axis.from, from);
  let end = Math.min(state.axis.to, to);

  if (end - start < MIN_SPAN) end = start + MIN_SPAN;
  if (end > state.axis.to) {
    end = state.axis.to;
    start = Math.max(state.axis.from, end - MIN_SPAN);
  }

  state.win = { from: start, to: end };

  // The whole archive is not "near the live edge": if it rode the
  // edge it would creep forward and quietly drop its first days.
  const whole = start <= state.axis.from + 0.5 && end >= state.axis.to - 0.5;
  state.whole = whole;
  state.follow = !whole && state.axis.to - end < Math.max(30, (end - start) * 0.05);

  // A slide moves the window, not the cursor: the rule is a fixed
  // time, and the window is what travels around it, so the same turn
  // stays marked. Only a resize (zoom, reset) has no time to keep and
  // re-anchors to its first row.
  state.at = carry && state.at != null
    ? Math.min(end, Math.max(start, state.at))
    : null;

  paintWindow();
  if (reload) loadPage(true);
}

// Sliding the window keeps its size: the travel is clamped, so
// running into the edge stops the window there.
function shiftWindow(origin, shift, options) {
  const low = state.axis.from - origin.from;
  const high = state.axis.to - origin.to;
  const by = Math.min(high, Math.max(low, shift));

  setWindow(origin.from + by, origin.to + by, { ...options, carry: true });
}

// A window change reloads the ledger from its new start, which would
// otherwise land the list at an arbitrary height. The marked turn is
// put back where it was on screen, so the archive moves and the list
// does not. `anchor` is the mark's height above the viewport's top.
async function reloadAround(anchor) {
  await loadPage(true);

  // The cursor kept its time through the change, so the ledger has
  // to reach it: otherwise the mark falls on the last loaded row.
  if (state.at == null) return;
  await follow(state.at);

  const now = markedItem();
  if (anchor == null || !now) return;

  settle();
  $("#scroll").scrollTop = Math.max(0, now.top - anchor);
}

// ---------------------------------------------------------------- cursor
//
// The cursor is a time, and a time is all it is: `state.at` is the one
// piece of state, everything else is derived from it.
//
// It points at a turn -- the nearest row to that time, which gets
// marked in the list. Dragging it moves the mark; the list only
// scrolls when the mark would leave the viewport (edge auto-scroll),
// and only by the minimum distance. That is what keeps the last screen
// reachable: a cursor cannot position "the top row", but it can always
// position "this row", and the list can always make that row visible.
//
// Scrolling by hand leaves the cursor alone; it is only re-anchored
// once its row has actually gone off screen.

// Positioned in pixels so the rule stays inside the strip: at ratio 0
// a negative margin would put half of it under the clip.
function placeCursor(time) {
  const strip = $("#focus");
  const cursor = $("#cursor");
  const width = strip.clientWidth || 600;
  const span = Math.max(0.001, state.win.to - state.win.from);
  const at = Math.min(1, Math.max(0, (time - state.win.from) / span));

  const label = $("b", cursor);
  label.textContent = stamp(time);

  // The label grew a year, so flip by its measured width and keep it
  // inside the strip near the right end.
  const room = (label.offsetWidth || 90) + 12;
  cursor.classList.toggle("flip", at * width > width - room);

  cursor.style.left = `${Math.min(width - 2, Math.max(1, at * width)).toFixed(1)}px`;
}

function cursorTime() {
  return state.at == null ? state.win.from : state.at;
}

// First row at or after `time`; rows.length means "all of them are
// earlier".
function lowerBoundRow(time) {
  const rows = state.rows;
  let lo = 0;
  let hi = rows.length - 1;
  let found = rows.length;

  while (lo <= hi) {
    const mid = (lo + hi) >> 1;

    if (rows[mid].started_at >= time) {
      found = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  return found;
}

function paintMarks() {
  for (const node of $("#rows").children) {
    if (node.dataset.ri == null) continue;
    node.classList.toggle("marked", Number(node.dataset.ri) === state.mark);
  }
}

// The row a time points at: whichever turn is nearer, not simply the
// next one. Skipping hidden rows keeps the mark on something visible.
function markIndexFor(time) {
  const rows = state.rows;
  if (!rows.length) return null;

  const index = lowerBoundRow(time);

  let before = index - 1;
  while (before >= 0 && state.rowItem[before] == null) before -= 1;

  let after = index;
  while (after < rows.length && state.rowItem[after] == null) after += 1;

  if (before < 0) return after < rows.length ? after : null;
  if (after >= rows.length) return before;

  const dBefore = time - rows[before].started_at;
  const dAfter = rows[after].started_at - time;

  return dBefore <= dAfter ? before : after;
}

// Keep the cursor, the label and the marked row as one fact.
function markCursor() {
  if (state.at != null) state.at = Math.min(state.win.to, Math.max(state.win.from, state.at));

  const mark = markIndexFor(cursorTime());
  const changed = mark !== state.mark;

  state.mark = mark;
  placeCursor(cursorTime());
  if (changed) paintMarks();
}

function markedItem() {
  if (state.mark == null) return null;

  const item = state.rowItem[state.mark];
  return item == null ? null : state.items[item];
}

// The smallest scroll that brings the marked row into view, stopping
// a row and a half short of the edge: the cursor should not sit on
// the last visible pixel, where the next step hides it.
function revealMark() {
  const item = markedItem();
  if (!item) return;

  const scroll = $("#scroll");
  const view = scroll.clientHeight;
  const top = scroll.scrollTop;
  const air = ROW_H * 1.5;

  // This is our scroll, not the reader's: the event it fires must not
  // be mistaken for a hand scroll and re-anchor the cursor.
  settle();

  if (item.top < top + air) scroll.scrollTop = Math.max(0, item.top - air);
  else if (item.top + item.h > top + view - air) {
    scroll.scrollTop = item.top + item.h + air - view;
  }
}

// Once the drag is over, the cursor snaps to the nearest real turn.
function snapCursor() {
  if (drag) return;

  const row = state.mark == null ? null : state.rows[state.mark];
  if (!row) return;

  state.at = row.started_at;
  placeCursor(state.at);
}

// Rows arrive oldest first, so a target past the tail means more
// pages -- which is exactly what "drag into the far end" asks for.
let followToken = 0;

async function follow(time) {
  const token = ++followToken;

  for (let page = 0; page < 80; page += 1) {
    const last = state.rows[state.rows.length - 1];

    if (state.done || !last || time <= last.started_at) break;

    const seen = state.rows.length;
    await loadPage(false);

    if (token !== followToken) return;
    if (state.rows.length === seen) break;
  }

  if (token !== followToken) return;

  markCursor();
  revealMark();
}

// Scrolling by hand: the cursor gives way before its row touches an
// edge, and lands on the row that entered there -- a row and a half
// in, so a slow trackpad scroll does not flip the mark on every pixel.
function keepCursorInView() {
  // Scrolls we caused ourselves (reload, reveal) say nothing about
  // where the reader is looking.
  if (performance.now() < settleUntil) return;

  const item = markedItem();
  if (!item) return;

  const scroll = $("#scroll");
  const view = scroll.clientHeight;
  const top = scroll.scrollTop;
  const air = ROW_H;
  const roomy = ROW_H * 1.5; // adopt halfway deeper, so the next flip is a row away

  if (item.top >= top + air && item.top + item.h <= top + view - air) return;

  let row = null;

  if (item.top < top + air) {
    for (let i = itemIndexAt(top + roomy); i < state.items.length; i += 1) {
      if (state.items[i].kind === "row") {
        row = state.items[i];
        break;
      }
    }
  } else {
    for (let i = Math.min(itemIndexAt(top + view - roomy), state.items.length - 1); i >= 0; i -= 1) {
      if (state.items[i].kind === "row" && state.items[i].top <= top + view - roomy) {
        row = state.items[i];
        break;
      }
    }
  }

  if (!row) return;

  state.at = row.row.started_at;
  state.mark = row.ri;
  placeCursor(state.at);
  paintMarks();
}

// The window strip is the cursor's surface, and the only place the
// window's size is set: drag to seek, wheel to zoom.
let drag = null;
let zoomTimer = null;

function bindFocus() {
  const strip = $("#focus");
  const cursor = $("#cursor");

  const ratioAt = (event) => {
    const rect = strip.getBoundingClientRect();
    return Math.min(1, Math.max(0, (event.clientX - rect.left) / (rect.width || 1)));
  };

  const timeAt = (ratio) => state.win.from + (state.win.to - state.win.from) * ratio;

  // Moving the cursor is all it takes: the mark is derived, and
  // revealMark() only scrolls when that mark would leave the view.
  const seek = (ratio) => {
    state.at = timeAt(ratio);
    markCursor();
    follow(state.at).catch(() => {});
  };

  strip.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;

    drag = true;
    strip.setPointerCapture(event.pointerId);
    cursor.classList.add("drag");
    event.preventDefault();
    seek(ratioAt(event));
  });

  strip.addEventListener("pointermove", (event) => {
    if (!drag) return;
    seek(ratioAt(event));
  });

  // Releasing runs one uninterrupted follow, so a flick to the far end
  // is not left short by a page fetch that was still in flight. The
  // cursor then sits on the row it actually landed on.
  const finish = (event) => {
    if (!drag) return;

    drag = null;
    cursor.classList.remove("drag");

    state.at = timeAt(ratioAt(event));
    markCursor();
    follow(state.at)
      .then(snapCursor)
      .catch(() => {});
  };

  strip.addEventListener("pointerup", finish);
  strip.addEventListener("pointercancel", finish);

  strip.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();

      // One pixel of delta must do the same thing on a wheel and on a
      // trackpad, where a single gesture emits dozens of small events:
      // so the zoom is exponential in the delta itself, not a fixed
      // step per event.
      const step = Math.max(-240, Math.min(240, event.deltaY * (event.deltaMode === 1 ? 16 : 1)));
      const span = state.win.to - state.win.from;
      const at = ratioAt(event);

      if (event.shiftKey) {
        shiftWindow(state.win, span * 0.12 * Math.sign(step));
        return;
      }

      const rate = Math.exp(step * 0.0015); // ~16% per wheel notch
      const next = Math.min(
        state.axis.to - state.axis.from,
        Math.max(MIN_SPAN, span * rate),
      );
      const centre = state.win.from + span * at;

      // A zoom is a window change too: the cursor keeps its time and
      // the ledger keeps the marked turn at the same height.
      const item = markedItem();
      const scroll = $("#scroll");
      const anchor = item ? item.top - scroll.scrollTop : null;

      // The band and both histograms follow every event; the ledger
      // reloads once the hand stops.
      setWindow(centre - next * at, centre + next * (1 - at), {
        reload: false,
        carry: true,
      });

      // A page may still be in flight; loadPage would drop the reset,
      // so wait for it before reloading.
      clearTimeout(zoomTimer);
      zoomTimer = setTimeout(function later() {
        if (state.loading) {
          zoomTimer = setTimeout(later, 80);
          return;
        }
        reloadAround(anchor).catch(() => {});
      }, 140);
    },
    { passive: false },
  );
}

// The archive strip moves the window and never resizes it: size is
// the window strip's business (the wheel there). One gesture, one
// meaning -- drag the band to slide, click off it to jump.
let brush = null;

function bindOverview() {
  const strip = $("#overview");
  const mark = $("#winmark");

  const ratioAt = (event) => {
    const rect = strip.getBoundingClientRect();
    return Math.min(1, Math.max(0, (event.clientX - rect.left) / (rect.width || 1)));
  };

  const timeAt = (ratio) => state.axis.from + (state.axis.to - state.axis.from) * ratio;

  // Centre the window on a time, keeping its span. Only the start
  // moves, and `setWindow` clamps it against the ends.
  const centreOn = (ratio, span) => {
    const from = Math.min(
      state.axis.to - span,
      Math.max(state.axis.from, timeAt(ratio) - span / 2),
    );
    // A jump slides the window too, so it carries the cursor the same
    // way: pressing off the band must not throw the cursor's time away.
    setWindow(from, from + span, { reload: false, carry: true });
  };

  strip.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;

    const rect = strip.getBoundingClientRect();
    const band = mark.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const ratio = ratioAt(event);

    // The band is the handle, so grabbing it must not also recentre
    // it under the hand: pressing it slides, pressing elsewhere jumps.
    const inside = x >= band.left - rect.left - 6 && x <= band.right - rect.left + 6;

    brush = { mode: inside ? "move" : "jump", start: ratio, origin: { ...state.win } };

    // A jump lands at once, so a click alone already moves the window;
    // the drag that may follow keeps it under the pointer.
    if (!inside) centreOn(ratio, state.win.to - state.win.from);

    strip.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  strip.addEventListener("pointermove", (event) => {
    if (!brush) return;

    const ratio = ratioAt(event);
    const { mode, origin } = brush;

    if (mode === "move") {
      shiftWindow(origin, (ratio - brush.start) * (state.axis.to - state.axis.from), {
        reload: false,
      });
      return;
    }

    centreOn(ratio, origin.to - origin.from);
  });

  strip.addEventListener("pointerup", () => {
    if (!brush) return;

    brush = null;

    // Where the marked turn sat on screen. The window slid, but the
    // ledger should not: after reloading, the same turn goes back to
    // the same height, which is what makes the slide feel like the
    // archive moving and not the list jumping.
    const item = markedItem();
    const scroll = $("#scroll");
    const anchor = item ? item.top - scroll.scrollTop : null;

    reloadAround(anchor).catch(() => {});
  });

  strip.addEventListener("pointercancel", () => {
    brush = null;
  });

  // Double-click here, not on the window strip: this is the scale
  // that knows what "everything" means.
  strip.addEventListener("dblclick", (event) => {
    event.preventDefault();
    showAll();
  });
}

function showAll() {
  setWindow(state.axis.from, state.axis.to);
}

function showLatest() {
  const span = Math.min(state.axis.to - state.axis.from, Math.max(600, state.win.to - state.win.from));
  setWindow(state.axis.to - span, state.axis.to);
}

// ---------------------------------------------------------------- drawer

function ancestorsOf(agentId) {
  const path = new Set();
  let cursor = agentId;
  let guard = 0;

  while (cursor && guard < 64) {
    path.add(cursor);
    const meta = state.agents.get(cursor);
    cursor = meta ? meta.parent_id : null;
    guard += 1;
  }

  return path;
}

async function openRow(row) {
  state.sel = keyOf(row);
  state.path = ancestorsOf(row.agent_id);
  state.drawer = row;
  state.tab = "summary";

  state.rendered = { from: 0, to: 0 };
  paintRows();
  renderDrawer();

  try {
    state.drawer = await api(`/api/turns/${encodeURIComponent(row.agent_id)}/${row.seq}`);
    renderDrawer();
  } catch {
    state.drawer = row;
  }
}

function closeDrawer() {
  state.drawer = null;
  state.sel = null;
  state.path = new Set();
  renderDrawer();
  state.rendered = { from: 0, to: 0 };
  paintRows();
}

function renderDrawer() {
  const drawer = $("#drawer");
  const turn = state.drawer;

  if (!turn) {
    drawer.hidden = true;
    drawer.textContent = "";
    return;
  }

  drawer.hidden = false;

  const kv = [
    ["agent", turn.agent_id],
    ["parent", turn.parent_id || "—"],
    ["depth", String(turn.depth ?? 0)],
    ["origin", turn.origin || "—"],
    ["started", clock(turn.started_at)],
    ["took", dur((turn.ended_at || turn.started_at) - turn.started_at)],
    ["model", turn.llm ? turn.llm.model : "—"],
    ["tokens", turn.llm && turn.llm.usage
      ? `${turn.llm.usage.prompt_tokens || 0} + ${turn.llm.usage.completion_tokens || 0}`
      : "—"],
    ["retries", String((turn.llm && turn.llm.retries) || 0)],
    ["inbox", String((turn.inbox || []).length)],
    ["injections", String((turn.injections || []).length)],
    ["delta", String((turn.delta || []).length)],
    ["calls", String((turn.calls || []).length)],
    ["thinking", String((turn.thinking || []).reduce((n, block) => n + (block.text || "").length, 0))],
  ];

  const head =
    `<div class="d-head"><span class="d-id"><span class="d-kicker">turn</span>` +
    `<span class="d-title">${esc(String(turn.seq))}</span></span>` +
    `<span class="d-agent">${esc(turn.agent_id)}</span>` +
    `<span class="spacer"></span><button class="ghost" id="close">close</button></div>`;

  const tabs =
    `<div class="tabs">` +
    ["summary", "preview", "raw"]
      .map((id) => `<button class="tab${state.tab === id ? " on" : ""}" data-tab="${id}">${id}</button>`)
      .join("") +
    `</div>`;

  let body = "";

  if (state.tab === "summary") {
    body =
      `<dl class="kv">` +
      kv.map(([k, v]) =>
        `<dt>${k}</dt><dd${k === "parent" && turn.parent_id ? ' class="link" data-agent="' + esc(turn.parent_id) + '"' : ""}>${esc(v)}</dd>`,
      ).join("") +
      `</dl>`;

    if (turn.persona != null) {
      // Not "the persona": this turn is the epoch marker, the text is
      // only here because the persona file changed since the last turn.
      body += `<div class="msg"><div class="msg-role">PERSONA CHANGE</div><div class="msg-body">${esc(turn.persona)}</div></div>`;
    }

    // `reply` is the field the collector records; `summary` only exists
    // on ledger rows, where the store derives it from this one.
    if (turn.reply) {
      body += `<div class="msg"><div class="msg-role">REPLY</div><div class="msg-body">${esc(turn.reply)}</div></div>`;
    }

    body += toolsBlock(turn);
  } else if (state.tab === "preview") {
    const rendered = turn.rendered || [];
    body = rendered.length
      ? rendered
          .map((message) => {
            const thinkText = (message.thinking || [])
              .filter((block) => block && block.text)
              .map((block) => block.text)
              .join("\n\n");

            const calls = (message.tool_calls || []).map((call) => {
              const name = call.name || (call.function && call.function.name) || "call";
              const args = call.arguments ?? call.args ?? (call.function && call.function.arguments);
              const line = argLine(args);

              return (
                `<span class="call"${line ? ` title="${esc(line)}"` : ""}>` +
                `<span class="call-name">${esc(name)}</span>` +
                (line ? `<span class="call-args">${esc(line)}</span>` : "") +
                `</span>`
              );
            }).join("");

            return (
              `<div class="msg"><div class="msg-role">${esc(String(message.role || "").toUpperCase())}</div>` +
              (thinkText
                ? `<details class="think"><summary>thinking<span class="think-chars">${thinkText.length}</span></summary>` +
                  `<div class="think-body">${esc(thinkText)}</div></details>`
                : "") +
              (message.content ? `<div class="msg-body">${esc(message.content)}</div>` : "") +
              (calls ? `<div class="calls">${calls}</div>` : "") +
              `</div>`
            );
          })
          .join("")
      : `<div class="empty">no rendered messages</div>`;
  } else {
    body = `<pre class="raw">${esc(JSON.stringify(turn, null, 2))}</pre>`;
  }

  drawer.innerHTML = `${head}${tabs}<div class="d-body">${body}</div>`;

  $("#close", drawer).addEventListener("click", closeDrawer);

  for (const tab of drawer.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      state.tab = tab.dataset.tab;
      renderDrawer();
    });
  }

  for (const link of drawer.querySelectorAll(".link")) {
    link.addEventListener("click", () => goToAgent(link.dataset.agent));
  }
}

// Select the newest row that agent owns, and scroll to it.
function goToAgent(agentId) {
  const row = [...state.rows].reverse().find((r) => r.agent_id === agentId);
  if (row) openRow(row);
}

// ------------------------------------------------------------------ live

async function refreshStats() {
  try {
    const stats = await api("/api/stats");
    const cap = stats.max_bytes ? ` / ${bytes(stats.max_bytes)}` : "";

    $("#stats").textContent =
      `${stats.turns} turns · ${stats.agents} agents · ${bytes(stats.bytes)}${cap}` +
      (stats.first ? ` · ${stamp(stats.first)} → ${stamp(stats.last)}` : "");

    if (stats.first) {
      // A sliver of room past the last turn so it is not glued to
      // the edge; capped, or a year-long archive would leave days of
      // empty space to the right.
      const head = Math.max(30, Math.min((stats.last - stats.first) * 0.02, 6 * 3600));
      state.axis = { from: state.axis.from && state.axis.from < stats.first ? state.axis.from : stats.first, to: stats.last + head };
    }

    return stats;
  } catch {
    return null;
  }
}

// New turns: extend the axis, ride the edge if we were riding it, and
// append whatever landed after the last row we hold -- so the gutter
// keeps telling the truth about branches that were still open.
async function advance() {
  const before = state.axis.to;

  await refreshStats();
  paintWindowMark();
  fetchStrip().catch(() => {});

  if (!state.axis.to) return;

  if (state.whole && state.win.to < state.axis.to - 0.5) {
    // Looking at everything means the new turns are in view too.
    state.win = { from: state.axis.from, to: state.axis.to };
    paintWindowMark();
    paintSeries();
    fetchSeries().catch(() => {});
  } else if (state.follow) {
    const span = state.win.to - state.win.from;
    const end = state.axis.to;
    const from = Math.max(state.axis.from, end - span);
    const shift = from - state.win.from;

    state.win = { from, to: end };

    // The window slid, the content did not: the cursor keeps its
    // place over the same rows.
    if (state.at != null) state.at += shift;

    paintWindowMark();
    paintSeries();
    fetchSeries().catch(() => {});
  }

  if (state.done && state.rows.length && before !== state.axis.to) {
    const last = state.rows[state.rows.length - 1];
    const from = Math.max(last.started_at, state.win.from);

    try {
      const data = await api(`/api/ledger?start=${from}&end=${state.win.to}&limit=${PAGE}`);
      const known = state.index;
      const fresh = data.rows.filter((row) => !known.has(keyOf(row)));

      if (fresh.length) {
        state.rows = state.rows.concat(fresh);
        rebuild();
      }
    } catch {
      /* the service may be restarting */
    }
  } else if (!state.rows.length) {
    loadPage(true);
  }

  markCursor();
}

let pending = null;

function scheduleAdvance() {
  if (pending) return;

  pending = setTimeout(() => {
    pending = null;
    advance().catch(() => {});
  }, 500);
}

function connect() {
  const source = new EventSource("/api/stream");

  source.addEventListener("turn", scheduleAdvance);
  source.onerror = () => {
    /* EventSource retries on its own */
  };
}

// ------------------------------------------------------------------ boot

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
}

// The desktop owns the theme and pushes it here; opened on its own the
// page reads the OS preference and offers no switch of its own.
window.addEventListener("message", (event) => {
  const data = event.data;

  if (!data || data.type !== "nan:theme") return;
  if (data.theme !== "light" && data.theme !== "dark") return;

  state.theme = data.theme;
  applyTheme();
});

function bind() {
  $("#fit").addEventListener("click", showAll);
  $("#latest").addEventListener("click", showLatest);

  $("#refresh").addEventListener("click", () => {
    refreshStats().then(() => {
      fetchStrip().catch(() => {});
      fetchSeries().catch(() => {});
      loadPage(true);
    });
  });

  $("#filter").addEventListener("input", (event) => {
    state.filter = event.target.value.trim().toLowerCase();
    rebuild();
  });

  $("#scroll").addEventListener(
    "scroll",
    () => {
      paintRows();
      keepCursorInView();

      const scroll = $("#scroll");
      const near = scroll.scrollTop + scroll.clientHeight > state.height - 600;
      if (near && !state.done) loadPage(false);
    },
    { passive: true },
  );

  $("#rows").addEventListener("click", (event) => {
    const node = event.target.closest(".row");
    if (!node) return;

    const row = state.index.get(node.dataset.key);
    if (row) openRow(row);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.drawer) closeDrawer();
  });

  let resize = null;
  window.addEventListener("resize", () => {
    clearTimeout(resize);
    resize = setTimeout(() => {
      paintStrip();
      paintSeries();
      paintWindowMark();
      fetchStrip().catch(() => {});
    }, 160);
  });
}

async function boot() {
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    state.theme = "dark";
  }

  applyTheme();
  bind();
  bindFocus();
  bindOverview();

  const stats = await refreshStats();
  const agents = await api("/api/agents");
  state.agents = new Map(agents.agents.map((agent) => [agent.agent_id, agent]));

  if (!stats || !stats.first) {
    $("#rows").innerHTML = `<div class="empty">the archive is empty</div>`;
    $("#foot").textContent = "waiting for the first turn";
    connect();
    return;
  }

  // The record opens whole: its shape is the first thing worth
  // seeing, and the two strips are how it gets narrowed down.
  setWindow(state.axis.from, state.axis.to);

  fetchStrip().catch(() => {});
  connect();

  setInterval(refreshStats, 10_000);
}

boot().catch((error) => {
  console.error(error);
  $("#rows").innerHTML = `<div class="empty">failed to load: ${esc(error.message)}</div>`;
});
