# NAN WS 事件协议 v1

状态:冻结。前端 `useAgentStream` 与后端网关共同遵守。
原则:**协议事件 ↔ UI 原语一一对应**,不引入 UI 用不到的字段。

## 可见性边界(黑盒原则)

主流只渲染 agent 层动作:user_input / modules.ask / tool / skill / response。
**模块内部过程(planner think、memory review…)一律不可见**——模块对 UI 是黑盒,
只有其抽象属性(可进入非主流展示,v2)。引擎在回合开始的 ambient 询问
(`modules.query_snapshot`)处,对每个 RUNNING 模块发一条
`record_started(kind:"module", name:"<module_id>")`,planner 阻塞多久
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

每条事件 = **信封字段**(顶层,由传输层/`sink` 填)+ `t`(类型判别)+ `content`(业务载荷)。

信封字段:

| 字段 | 由谁填 | 语义 |
|---|---|---|
| `seq` | EventBus | 单调递增序号;前端按此去重历史重放 |
| `ts` | EventBus | wall-clock 数值 epoch;前端在渲染/折叠边缘转成日期或时间(数值形式由 bus 统一加盖) |
| `t` | 调用方 | **唯一**事件类型判别字段(扁平字符串) |
| `id` | sink | 事件 id,UI 行关联键;调用方可显式提供以复用既有行,否则由 sink 生成 |
| `boot_id` | sink | 进程唯一 id,`attach` 时记录;已设置则每条事件都带,未设置则不带该键 |

`content` 规则:

- 恒为一个对象;无载荷的事件也必须为 `{}`,绝不省略。
- 承载该 `t` 的全部业务字段(不再是散落在顶层的 kwargs)。
- 身份三键(见下)仅在 agent 上下文中出现。

各 `t` 的 `content` 形状:

| t | 对应原语 | content | 语义 |
|---|---|---|---|
| `hello` | — | `boot, model, base_url, status:{state}` | 握手;`boot` 为进程唯一 id,客户端检测到变化即清空本地流并重置 seq 基线。此事件无 `id`(不经 sink) |
| `status` | Pulse | `state:"idle"\|"working"\|"error"` | 循环状态机变化 |
| `user_input` | Message | `text`, `mid?` | 用户输入回显(进了 Inbox 才发);`mid` 原样回带 |
| `record_started` | Record | `kind:"tool"\|"skill"\|"spawn"\|"sleep"\|"finish"\|"module"\|"agent"\|"target"\|"error"`, `name`, `summary?` | 机器过程开始(braille 转轮) |
| `record_detail` | Record | `line`(html-free 纯文本) | 详节逐行追加 |
| `record_done` | Record | `summary?`, `note?` | ✓ 自动折叠 |
| `record_failed` | Record | `summary?` | ✗ 保持展开 |
| `record_void` | Record | `{}` | 该记录无用户可见产出,撤回行 |
| `output_started` | Message | `{}` | NAN 文本开始 |
| `output_delta` | Message | `text` | 流式增量(直接拼接) |
| `output_done` | Message | `duration` | 停止打字;信封 `ts` 为数值 epoch,前端渲染为人类可读;尾部 Note `✓ ts · duration` |
| `output_cancelled` | Message | `{}` | 文本流中途出现 tool_call,撤回该段(非最终输出) |

`id` 的关联规则:`record_started` 返回新生成/指定的 id,后续同一记录的 `record_detail` /
`record_done` / `record_failed` / `record_void` 复用该 id;`output_started` 同理,后接
`output_delta` / `output_done` / `output_cancelled`。`record_void` 与 `output_cancelled` 只撤回
最近一条同 id 的行,旧回合的同 id 行不受影响。

divider 不是协议事件:前端按每个事件的数值 `ts` 自行派生日期分隔(本地日期变化时插入)。

`pong` 是心跳应答,形状为 `{"t":"pong","content":{}}`,不参与上述流折叠。

## 公共身份字段(content 携带)

身份三键由发出该事件的一方在 `content` 中携带;未携带即表示该事件无 agent 上下文
(module 生命周期、gateway 的 `user_input` 回显)。未携带时,这三个键**不存在**(不是 `null`)。

- `agent_hash`:发出该事件的 agent 的稳定标识。
- `parent_hash`:其父 agent 的 `agent_hash`;root agent 为 `null`。
- `depth`:该 agent 在树上的深度;root 为 `0`。

非 agent 事件(module 生命周期、gateway 的 `user_input` 回显)无 agent 上下文,不携带这三个键。

## 约定

1. `id` 由 server 生成,单调递增;前端不生成 id。调用方也可显式提供 id 以关联既有行(如 `record_detail` 复用 `record_started` 的 id)。
2. `record_detail.line` 为纯文本;前端可对 `·`、`✓`、`✗` 做轻量着色,不做 HTML 注入。
3. verb 的 kind 映射:invoke_tool/list_tools/show_tool → `tool`,invoke_skill/list_skills/show_skill → `skill`,invoke_channels/list_channels/show_channels → `target`(glyph ⌖),spawn → `spawn`,sleep → `sleep`,finish → `finish`(glyph ⏻);未知 verb 落 `verb` 兜底(前端 glyph '•')。
4. 子代理简报 = `record_started(kind:"spawn")` 的 detail;报告 = 独立 `record_started(kind:"agent", name:"report · <task>")`。
5. 断线重连:client 重连后收到 `hello`,随后 server 重放最近 200 条事件的**折叠投影**(当前流快照),前端以快照重建流。
