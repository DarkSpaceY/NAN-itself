# NAN Memory System · 设计文档 v1

状态：待评审。评审通过后一次性平推实现。

---

## 0. 价值基线

1. **语义判断交给模型，机械操作交给有根据的算法**；系统内不存在手写的内容谓词。
2. **读路径零智能**：`query()` 只做廉价投影（打分是纯算法），LLM 调用全部位于写路径的后台任务。
3. **无损捕获**：每条 TurnRecord 先落 journal 再谈其他。
4. **生产方责任**：条目体量纪律在写入端解决（合并、升华、有界工作集），不设读取侧预算机制。
5. **权威存储人可读可编辑**：用户手工修改视为合法外部输入，模块检测变化后重新加载。
6. 记忆的可信度低于实时观测（persona 已规定该信任序）。

## 1. 三层结构

```
TurnRecord 流
      │ on_turn(): 全量追加 journal（含子代理 turn；REVIEW 仅消费主代理 turn）
      ▼
journal.jsonl ──(游标)──► 后台消费循环
      │  REVIEW：每批一次受限解码调用
      ▼
WORKSTATE（热层）          ENTRY STORE（冷层）
开放事务条目，≤10 条        原子事实 + 情节摘要
高频改写                    双时间轴、importance、links
      │ 情节闭合 → PROMOTE         │ 低频 REFLECT / DISTILL
      └────────► 合入 ◄────────────┘
                    │
                    ▼
            core block（常驻压缩投影，DISTILL 产物）
```

**情节的定义**：一条工作状态条目的完整生命周期（开启 → 累积 → 闭合）。情节闭合是短时向长时沉淀的唯一触发点。

## 2. Module 形态

- `class MemoryModule(Module)`，`id = "memory"`，注册进 `BUILTIN_MODULES`。
- `on_turn(record)`：追加 journal；depth==0 的记录进入待审队列。本方法只做这两件事。
- `start()`：启动后台消费循环 + 文件监听（检测用户对存储文件的手工编辑）。
- `query(turn)`：渲染输出（见 §6）。通过 `turn.depth` 可区分主/子代理，V1 输出相同内容。
- 运行时状态同步发布到自身 DataSpace（供既有持久化机制与其他模块依赖读取）。

## 3. 文件布局

```
workspace/memory/
├── WORKSTATE.md    # 开放事务条目（仅 open 状态）；高频原子重写
├── MEMORY.md       # 长期条目库；低频原子重写；权威存储
├── journal.jsonl   # TurnRecord 追加日志（全量、只追加）
├── journal.cursor  # 已消费序号（消费成功后推进）
└── embeddings.json # 嵌入缓存旁车（id+content_hash → 向量；可再生）
```

两文件分离的理由：改写频率相差两个数量级；损坏时的爆炸半径隔离。

### 条目语法（MEMORY.md）

```markdown
<!-- memory-next-id: 43 -->

## [m-000042] 用户偏好简洁的回答风格
kind: preference
importance: 7
created: 2026-08-26T14:00:00
valid_from: 2026-08-26T14:00:00
valid_to: null
source: j-000123
links: []

用户明确表示回答要简洁。
```

### 工作条目语法（WORKSTATE.md）

```markdown
## [w-000007] 调研 A 方案并给出结论
kind: task
opened: 2026-08-26T10:00:00
last_active: 2026-08-26T14:20:00

- 08-26 已完成初步对比
```

闭合的条目从本文件移除，其生命周期数据由 journal 区间提供（PROMOTE 的输入）。

### journal 行格式

```json
{"seq": 123, "ts": "...", "agent_hash": "...", "depth": 0,
 "task": null, "user_input": "...", "reply": "...", "error": null}
```

## 4. 智能操作规格

所有 LLM 调用走受限解码（Ollama `format` JSON schema）。单次调用超时默认 90s，退避重试 2 次；仍失败则放弃该批（游标不动），循环休眠后重试。消费循环永不退出。

### REVIEW（高频）

输入渲染：

```
[当前开放工作状态]
w-0007 (task, opened 3d) 调研 A 方案……

[本批 turn]
--- j-120 (08-26 14:00)
用户: ……
NAN: ……
```

输出 schema：

```json
{
  "workstate_ops": [
    {"op": "open|progress|close",
     "target": "w-000007 | null",
     "kind": "task|promise|question",
     "content": "…",
     "note": "进度注记或闭合结果"}
  ],
  "facts": [
    {"content": "自包含原子事实（指代已消解）",
     "kind": "preference|agreement|project|other",
     "importance": 1}
  ]
}
```

应用顺序：先执行 workstate_ops 并重写 WORKSTATE.md；判定闭合的条目逐个触发 PROMOTE；facts 进入 MERGE。

### MERGE（随 REVIEW 触发）

整批候选一次调用。输入含每个候选的邻域（嵌入余弦 top-k，k=3；候选间重复由模型在同上下文中自行识别）。输出：

```json
{"decisions": [
  {"candidate": 0,
   "op": "ADD|UPDATE|DELETE|NOOP",
   "target": "m-000012 | null",
   "content": "最终文本"}
]}
```

UPDATE：旧条目置 `valid_to`，新内容以新 id 入库。

### PROMOTE（情节闭合时）

输入 = 该条目生命周期对应的 journal 区间渲染 + 闭合结果。输出 = 情节摘要（一段自包含文字 + importance），作为候选走 MERGE 入长期库。

### REFLECT（累计重要性触发）

新增条目的 importance 之和 ≥ 阈值（默认 40）时触发：取相关高分簇，产出洞见条目，`links` 指向来源。原条目不改动。

### DISTILL（贡献集变化超出容差）

core block = 高重要性（≥8）且有效条目的压缩投影，≤12 行。贡献集合（按 id 集）变化才重算。

## 5. 喂食策略（query）

```
<module>
[Memory]
- workstate w-0007 (task) 调研 A 方案——已完成初步对比，待整理结论
- (pref) 回答要简洁
- ……
</module>
```

渲染领地契约：本模块输出收拢于 `[Memory]` 单一头下；
workstate 行加 `workstate` 前缀，core 投影以缩进子块呈现，
禁止铸造 `[Workstate]`/`[Core]` 之类看似全局的 section。

分阶段：

| 阶段 | 条件 | 行为 |
|---|---|---|
| 全量期 | 有效条目数 ≤ FULL_INJECT_LIMIT（默认 60） | 全部注入，按 importance 降序 |
| 打分期 | 超过上限 | core block 必现 + 其余按三因子排序取 top-K |

三因子：`score = w_r·exp(−Δt_days/τ) + w_i·(importance/10) + w_s·sim(input, e)`。
默认 `w = (0.3, 0.4, 0.3)`，`τ = 30 天`。`sim` 在打分期启用嵌入余弦（旁车缓存）；BM25 作为无嵌入时的降级与关键词补充通道（中文按二元组切分）。

嵌入计算时机：进入打分期前由维护任务一次性回填，此后写路径增量计算。

## 6. 配置项汇总

| 参数 | 默认 | 说明 |
|---|---|---|
| poll_interval | 15s | 消费循环空转间隔 |
| batch_limit | 8 | 单批 turn 上限 |
| llm_timeout | 90s | 单次调用超时 |
| workstate_max_open | 10 | 开放条目上限，超出迫使 REVIEW 裁决 |
| full_inject_limit | 60 | 全量注入阈值 |
| reflect_threshold | 40 | 累计重要性触发线 |
| distill_tolerance | 1 | 贡献集变动容差 |
| τ / w_r / w_i / w_s | 30d / 0.3 / 0.4 / 0.3 | 打分参数 |

## 7. 失败与边界

| 场景 | 处理 |
|---|---|
| LLM 超时/异常 | 重试 2 次 → 放弃本批（游标不动）→ 退避休眠；队列无损 |
| 自管文件解析失败 | **绝不截断**：将坏文件改名 `.corrupt-<ts>` 存证，写入停摆，CRITICAL 日志，query 继续供给 DataSpace 中最后一次完好投影；等待人工介入 |
| 用户并发编辑 | 每次变更周期先重读文件再修改——外部编辑自然并入下一轮 |
| 进程重启 | journal + 游标天然支持续读；条目库从文件加载 |
| 子代理 turn | 全量进 journal（可溯源），REVIEW 不消费 |

## 8. 测试契约清单

单元层：
1. journal 追加 + 游标续读（模拟崩溃后重启不丢不重）
2. REVIEW 受限输出的三种 workstate op 应用（open/progress/close）
3. 闭合触发 PROMOTE，产物经 MERGE 成为情节条目
4. MERGE 四操作路径（ADD / UPDATE 置 valid_to / DELETE / NOOP）
5. 手工编辑存储文件 → 热重载生效
6. 打分期切换（构造超过 full_inject_limit 的条目集）
7. REFLECT 阈值触发
8. 损坏文件安全停摆行为
9. on_turn/query 的耗时上界（复用基础设施隔离测试模式）

真机冒烟场景：
- S1 偏好闭环：告知偏好 → 数轮后新提问 → 回答体现该偏好且 `<module>` 含对应条目
- S2 多回合事务：任务开启 → 中途可见于 Workstate → 闭合后 MEMORY 出现情节摘要 → 新回合引用
- S3 断点续传：消费中途 SIGKILL → 重启 → 无丢失无重复
- S4 满员引导下全套运行（MCP 六件套在场）

## 9. 实验计划（experiments/memory/，评审期间执行）

| 实验 | 解锁决策 |
|---|---|
| X1′ EXTRACT/REVIEW+MERGE 提示词在 qwen2.5:7b 受限解码下的质量、延迟、坏输出率、超时行为 | 消化机制可行性；失败策略参数 |
| X2 qwen3-embedding:4b 经 Ollama embeddings 的延迟与区分度 | 打分期嵌入方案的时机确认 |
| X4 冒烟日志中真实 turn 的体积/节奏分布 | 门控与批量参数标定 |

## 10. 实施顺序（单次平推内部的构建次序）

model/格式定义 → 存储读写层 → REVIEW/MERGE/PROMOTE/REFLECT/DISTILL 五操作 → 消费循环与文件监听 → query 喂食 → 单元测试 → 冒烟 S1–S4。

---

## 附：调研备忘（要点摘录）

- Mem0：抽取-更新两阶段 + ADD/UPDATE/DELETE/NOOP 操作语义（本文 MERGE 直接采用）
- Zep/Graphiti：双时间轴有效期字段（本文 valid_from/valid_to 来源）
- Generative Agents：三因子检索公式与累计重要性反思触发（本文打分与 REFLECT 来源）
- Letta：常驻 core block 与后台重组（本文 DISTILL 与消费循环形态参照）
- A-MEM：links 字段为未来关联结构预留位
- Claude memory tool：人可读目录、空白开局、用户可直接编辑（本文存储立场来源）
- Manus：todo 式外置工作结构的有效性佐证 WORKSTATE 层（但其 KV-cache 导向不采纳）
