# NAN WS 事件协议 v1

状态:冻结。前端 `useAgentStream` 与后端网关共同遵守。
原则:**协议事件 ↔ UI 原语一一对应**,不引入 UI 用不到的字段。

## 可见性边界(黑盒原则)

主流只渲染 agent 层动作:user_input / modules.query / tool / skill / response。
**模块内部过程(planner think、memory review…)一律不可见**——模块对 UI 是黑盒,
只有其抽象属性(可进入非主流展示,v2)。引擎在回合开始的 `modules.query_snapshot`
处发一条 `record_started(kind:"module", name:"modules.query")`,planner 阻塞多久
都只是这一行的转轮。

## 通道

- 网关:`ws://127.0.0.1:8765/ws`(开发时经 Vite proxy `/ws`)
- 消息为 JSON,每行一条;字段名全小写。

## client → server

| t | 字段 | 语义 |
|---|---|---|
| `input` | `text`, `mid?` | 用户输入(等价 stdin 一行);`mid` 为客户端消息 id,服务端按 mid 去重(断线重连补发同一 mid 不重复投递),回显 `user_input` 原样携带 mid |
| `ping` | — | 心跳 |

## 连接层约定(断线重连)

1. 检活只由心跳负责:client 每 15s 发 `ping`,4s 内无 `pong` 判死重连。
2. **回执看门狗不踢线**:`input` 发出后 5s 内未收到对应 `user_input` 回显,客户端只撤销本地的 pending 状态——"回显慢"(回合繁忙/GIL 停顿)不是"消息丢失",踢线重发会导致同一条消息重复投递。
3. 真正断线(onclose)时若回执未到 → 重连后以**同一 mid** 补发;服务端 ingest 按 mid 去重,保证恰好一次。stdin 入口 mid 为空,不参与去重。

## server → client

| t | 对应原语 | 字段 | 语义 |
|---|---|---|---|
| `hello` | — | `seq, boot, model, base_url, status` | 握手;`boot` 为进程唯一 id,客户端检测到变化即清空本地流并重置 seq 基线 |
| `status` | Pulse | `state:"idle"\|"working"\|"error"`, `tools?`, `subagents?`, `next_hop?` | 循环状态机变化 |
| `user_input` | Message | `id, text` | 用户输入回显(进了 Inbox 才发) |
| `record_started` | Record | `id, kind:"tool"\|"module"\|"agent"\|"verb"\|"skill"\|"error"`, `name`, `summary?` | 机器过程开始(braille 转轮) |
| `record_detail` | Record | `id, line(html-free 纯文本)` | 详节逐行追加 |
| `record_done` | Record | `id, summary?, note?` | ✓ 自动折叠 |
| `record_failed` | Record | `id, summary?` | ✗ 保持展开 |
| `output_started` | Message | `id` | NAN 文本开始 |
| `output_delta` | Message | `id, text` | 流式增量(直接拼接) |
| `output_done` | Message | `id, ts, duration` | 停止打字;尾部 Note `✓ ts · duration` |
| `output_cancelled` | Message | `id` | 文本流中途出现 tool_call,撤回该段(非最终输出) |
| `divider` | Divider | `label` | 日期分隔(网关在跨天时自动注入) |

## 约定

1. `id` 由 server 生成,单调递增;前端不生成 id。
2. `record_detail.line` 为纯文本;前端可对 `·`、`✓`、`✗` 做轻量着色,不做 HTML 注入。
3. `sleep` 是普通 verb:产生 `record_started(name:"sleep")` + `record_done(summary:"下一跳 22:42:13")`,无专门事件。
4. 子代理简报 = `record_started(kind:"agent")` 的 detail;报告 = 独立 `record_started(kind:"agent", name:"report · <task>")`。
5. 断线重连:client 重连后收到 `hello`,随后 server 重放最近 200 条事件的**折叠投影**(当前流快照),前端以快照重建流。
