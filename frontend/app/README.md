# NAN UI

NAN 的前端(β 三明治骨架 + 薄暮原语系统)。

- 设计规范:`../DESIGN.md`
- 事件协议:`../PROTOCOL.md`(v2)
- 结构/风格历史样例:`../samples/`(存档)

## 运行

```bash
npm install --cache /tmp/npm-cache   # 若 ~/.npm 权限受限
npm run dev                          # http://localhost:5173
npm run build                        # tsc --noEmit && vite build
```

前端只连接同源 WebSocket:`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`。
生产环境由网关在 `127.0.0.1:8765` 同时服务静态前端与 `/ws`;开发环境由 Vite 把 `/ws` 代理到该端口。

## 结构

```
src/
├── protocol.ts               # 事件契约(PROTOCOL.md v2 的镜像)
├── state/store.ts            # 事件 → 流条目的纯折叠(含 glyph/label/日期分隔派生)
├── state/useAgentStream2.ts  # WS 客户端:心跳 / 重连 / mid 去重 / seq 去重
├── design/                   # 原语组件:RecordItem / Message2
├── shell/                    # 骨架:TopBar(状态机)/ Composer
└── styles/index.css          # tokens + 原语样式
```

## 原语 → 事件对照

| 原语 | 事件 |
|---|---|
| Message(用户) | `user_input` |
| Message(NAN) | `output_started / output_delta / output_done` |
| Record | `record_started / record_detail / record_done / record_failed / record_void` |
| Pulse | `hello.content.status` / `status` |

日期分隔不是协议事件:前端按每个事件的数值 `ts` 自行派生。
