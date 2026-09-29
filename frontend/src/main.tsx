// NAN frontend — entry
// 数据层（store/ws/mapper/db，框架无关）+ React 呈现层（assistant-ui）。

import "./style.css";
import { createRoot } from "react-dom/client";
import { createElement } from "react";

import { Store } from "./store";
import { Mapper } from "./history/mapper";
import { createRoundStore } from "./persistence/db";
import { App } from "./ui/App";

const store = new Store();
const mapper = new Mapper();
const db = createRoundStore();

let sendInput: (text: string) => void = () => {};

const root = createRoot(document.getElementById("root")!);
root.render(
  createElement(App, {
    store,
    mapper,
    db,
    sendInput: (text) => sendInput(text),
  }),
);

async function boot() {
  // 恢复本地历史（重放会按 mid 覆盖对齐）
  try {
    const rounds = await db.loadRounds();
    if (rounds.length > 0) mapper.seed(rounds);
  } catch (err) {
    console.warn("[nan] history restore failed", err);
  }

  const useMock = new URLSearchParams(location.search).has("mock");

  if (useMock) {
    const { createMockClient } = await import("./dev/mock");
    const mock = createMockClient({
      onEnvelope: (env) => {
        store.applyMessage(env);
        mapper.feed(env);
      },
      onConnection: (state) => store.setConnection(state),
    });
    sendInput = (text) => mock.sendInput(text);
    mock.connect();
    return;
  }

  const wsUrl =
    (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  const { WsClient } = await import("./ws");
  const client = new WsClient({
    url: wsUrl,
    store,
    onEnvelope: (env) => mapper.feed(env),
    onHello: (bootId) => mapper.onHello(bootId),
  });
  store.setConnection("connecting");
  client.connect();
  sendInput = (text) => client.sendInput(text);
}

void boot();
