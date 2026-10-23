// 入口：一个标签页 = 一份事件流的视图，各标签页的队列互不影响。
//
// ?mock 走本地剧本流；默认连同源 /ws（dev 下由 vite 代理到网关）。

import { createRoot } from "react-dom/client";

import "./style.css";
import { Hub } from "./kernel/history/hub";
import { GlobalStore } from "./kernel/store";
import { Shell } from "./shell/Shell";
import { applyTheme, loadTheme } from "./shell/theme";
import { MockTransport } from "./transport/mock";
import { WsTransport } from "./transport/ws";

applyTheme(loadTheme());

const global = new GlobalStore();
const hub = new Hub(global);

const mock = new URLSearchParams(location.search).has("mock");
const transport = mock ? new MockTransport(hub) : new WsTransport(hub);

createRoot(document.getElementById("root")!).render(
  <Shell hub={hub} transport={transport} />,
);
