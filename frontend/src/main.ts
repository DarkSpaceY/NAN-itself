// NAN frontend — bootstrap
// 架构：舞台（布局引擎管理面）+ 中央 dock（历史面板 + 输入框）
// 阶段 1：数据层接线（store + ws），渲染在阶段 2/3。

import { Store } from "./store";
import { WsClient } from "./ws";

const store = new Store();

const wsUrl =
  (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";

const client = new WsClient({ url: wsUrl, store });
client.connect();

// 临时探针：阶段 2/3 接入渲染后移除
store.subscribe(() => {
  const s = store.getState();
  console.debug(
    "[nan]",
    s.connection,
    `seq=${s.lastSeq}`,
    `rounds=${s.rounds.length}`,
    `items=${Object.keys(s.items).length}`,
    s.status ?? "-",
  );
});

console.log("NAN frontend boot");
