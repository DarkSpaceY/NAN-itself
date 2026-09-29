// NAN frontend — bootstrap
// 数据层（store/ws/mapper/db）+ dock（输入框/上拉历史面板）。
// 舞台与布局引擎在阶段3接入。

import "./style.css";

import { Store } from "./store";
import { WsClient } from "./ws";
import { Mapper, type HistoryRound } from "./history/mapper";
import { HistoryView } from "./history/render";
import { createRoundStore } from "./persistence/db";
import { createDock } from "./dock/dock";

const store = new Store();
const mapper = new Mapper();
const db = createRoundStore();

// ---- DOM 骨架 ----

const stage = document.createElement("div");
stage.id = "stage";
document.body.appendChild(stage);

// 连接建立后由 boot() 注入；此前输入框是禁用的，不会触发。
let sendInput: (text: string) => void = () => {};

const dock = createDock({
  onSend: (text) => sendInput(text),
});
document.body.appendChild(dock.root);

const historyView = new HistoryView(dock.historyEl);

// ---- 持久化：轮更新写 IndexedDB（流式更新节流 500ms） ----

let saveTimer: ReturnType<typeof setTimeout> | null = null;
let pendingRound: HistoryRound | null = null;

mapper.subscribe((round) => {
  historyView.upsert(round);

  // 关闭的轮立即落盘；活动轮节流落盘
  if (round.closed) {
    pendingRound = null;
    void db.saveRound(round);
    return;
  }
  pendingRound = round;
  saveTimer ??= setTimeout(() => {
    saveTimer = null;
    if (pendingRound && !pendingRound.closed) void db.saveRound(pendingRound);
  }, 500);
});

// ---- 启动：先恢复本地历史，再连接（重放会按 mid 覆盖对齐） ----

async function boot() {
  try {
    const rounds = await db.loadRounds();
    if (rounds.length > 0) {
      mapper.seed(rounds);
      historyView.setAll(mapper.getRounds());
    }
  } catch (err) {
    console.warn("[nan] history restore failed", err);
  }

  const wsUrl =
    (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";

  const client = new WsClient({
    url: wsUrl,
    store,
    onEnvelope: (env) => mapper.feed(env),
  });
  dock.setConnection("connecting");
  client.connect();

  store.subscribe(() => {
    dock.setConnection(store.getState().connection);
  });

  sendInput = (text) => client.sendInput(text);
}

void boot();
