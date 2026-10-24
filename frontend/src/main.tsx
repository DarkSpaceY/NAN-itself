// Entry: one tab = one view of the event stream; each tab's queue is
// independent.
//
// ?mock runs the local scripted stream; by default the same-origin
// /ws is used (in dev, vite proxies it to the gateway).

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
