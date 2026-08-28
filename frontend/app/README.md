# NAN UI

NAN 的前端(β 三明治骨架 + 薄暮原语系统)。

- 设计规范:`../DESIGN.md`
- 事件协议:`../PROTOCOL.md`
- 结构/风格历史样例:`../samples/`(存档)

## 运行

```bash
npm install --cache /tmp/npm-cache   # 若 ~/.npm 权限受限
npm run dev        # http://localhost:5173 ,默认 mock 回放
npm run dev -- --mode ws   # 或直接加 ?ws=1 查询参数 → 连接 ws://…/ws 网关
```

- **mock 模式**(默认):本地回放一个完整回合,不依赖后端;
- **ws 模式**:`http://localhost:5173/?ws=1`,需要 NAN 进程的网关在 `127.0.0.1:8765`(Phase 2)。

## 结构

```
src/
├── protocol.ts        # 事件契约(PROTOCOL.md 的镜像)
├── state/store.ts     # 事件 → 流条目的纯折叠
├── state/useAgentStream.ts  # WS 客户端 / mock 驱动装配
├── mock/driver.ts     # 开发用事件回放
├── design/            # 原语组件:RecordRow / Message
├── shell/             # 骨架:TopBar(状态机)/ Composer
└── styles/index.css   # tokens + 原语样式
```

## 原语 → 事件对照

| 原语 | 事件 |
|---|---|
| Message(用户) | `user_input` |
| Message(NAN) | `output_started/delta/done` |
| Record | `record_started/detail/done/failed` |
| Pulse | `hello.status` / `status` |
| Divider | `divider` |