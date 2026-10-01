# infraend 设计文档

> Todo: 实现 infra 模型推理和批量持续学习后端

## 1. 定位

infraend 是 NAN 的第三个"端"（与前端、agent 后端并列）：

- **独立进程**，作为**可选部件**安装（用户硬件各异，能力分层降级）
- 对 NAN 暴露 **OpenAI 兼容 API**（底座：llama.cpp `llama-server` 二开）
- 与 agent 后端的关系：Turn 写盘 → infraend 主动处理数据，单向解耦

## 2. 推理端

- 以 llama.cpp 为底座做二开，**插件式接入推理加速 method**
- 优化不分位置：无论作用于权重（量化）、上下文（KV）、调度（batching）、解码（投机），都以可插拔 method 形式注册——插件接口按"横切"设计，不绑死某一层
- 并发模型：持续开机 + 多 agent 并发（parent/children），batching/调度是常驻能力
- LoRA 热切换：训练产出新 adapter 后，经 GGUF 转换 → llama.cpp `/lora-adapters` 端点运行时热挂载（无需重载模型）

### 2.1 Adapter 版本管理与灰度（已定）

**多实例形态**（llama.cpp 的 LoRA 热加载是服务器级而非请求级）：每个 adapter 版本起一个 llama-server 实例，前面轻量路由层按 agent_hash 分流。显存换隔离与秒级回滚，同时适配不同用户硬件的部署弹性。

**版本谱系**：每周从固定 base 重训 → 线性版本链（`nan-7b+adapter-<年周>`），不做谱系合并（adapter 合并工程不成熟）；保留最近 N 个版本可回滚。

**四级灰度流水线**：

```
新 adapter 产出（每周）
 ↓ ① 影子评估（无真实流量）：评估集重放 + benchmark 回归 + 宪法逐条款向量对比（新 vs 现役）
 ↓ ② 金丝雀（小流量）：部分 agent 会话路由到新版本，收集 Turn 做跨版本对比组
 ↓ ③ 全量替换，旧版本保留
 ↓ ④ 回滚通道常开：线上宪法评分或失败率劣化超阈值 → 自动/手动切回
```

**通过/回滚判据**：评估集重放不劣化 + 金丝雀期宪法加权分不下降 + 红线触发率不上升；阈值第一版拍脑袋、后续校准。Turn 的 `model` 字段须写入 adapter 版本号，作为跨版本对比组的记账基础。

## 3. 数据管道：Turn 是唯一数据源

Turn（见 `backend/nan_itself/modules/model.py`）是自包含的结构化决策记录，把 agent 在每个 step 看到的内容全部结构化：

- identity：agent_hash / parent_hash / depth —— 天然轨迹树
- inputs：task / world / ambient / reports —— 观测的结构化来源
- snapshots：persona / history —— 可精确复现该 turn 的 prompt 上下文
- flow：reply / calls / results —— (observation → action → outcome) 三元组
- outcome：usage / finish_reason / error / 时间戳 —— 质量筛选依据

设计含义：

1. 观测可归因、可重渲染（换 prompt 模板不需重跑 agent）、可构造变体（遮蔽/扰动/消融）
2. 事件流不是数据源；schema 演进由 Turn 定义方（backend）主导，infraend 跟随
3. 数据管道（Turn 解析 → 预处理 → 筛选 → 打包）在所有硬件层级上完整可用，是飞轮的核心工序

## 4. 训练循环（核心想法 + 调研结论）

### 4.1 飞轮概览

慢思考教师 → 快思考学生：

- 后台 rollout 采用**测试时扩展**（多次采样/反思/best-of-n），投入更多算力产出高质量轨迹
- 教师信号通过 **GRPO 相对优势 + 蒸馏**传回快思考学生（同权重自蒸馏，师生不必异权重）
- 每周批量训练：百~千条/天数据，攒批离线跑

### 4.2 GRPO 双奖励

- 奖励 = 任务完成率（稀疏） + 宪法对齐评分（稠密）
- 两路奖励**各自归一化后再组内比较**（防止尺度支配）
- credit assignment 备选路线：轨迹级广播（最稳）/ turn 级混合奖励（MT-GRPO）/ 步级锚点分组（GiGPO，与 Turn 的 parent_hash 树天然契合）
- 已知坑：熵塌缩（DAPO clip-higher 类缓解）、全对/全错死组零梯度、能力天花板固化（自训练压缩分布不扩边界——宪法奖励作为外部标准是打破封闭性的出口）

### 4.3 宪法评分：离散档位（已定）

- 序数化档位（如 0-3），介于二值裁决与连续打分之间：抗 hack（台阶边界无可爬梯度）且比二值稠密
- **每个档位绑定可核验的规则锚点**（如 0=违反宪法条款X / 1=无违反但未完成 / 2=完成但冗余 / 3=完成且路径最优），判断走 judge 集成投票定档
- 档位同时映射蒸馏数据筛选阈值（如 ≥2 档进 SFT 蒸馏集），天然形成质量闸门

### 4.4 轨迹数据的六层利用（按投入产出比排序）

1、2、3 层不依赖训练能力，是数据管道的常驻功能（服务含 L0 在内的所有层级）；4、5、6 依赖训练层级。

1. **评估集（回归测试）**：抽代表性 task + 当时观测，新 adapter 定期重放对比。eval_loss 会因数据 stale 失真，真实轨迹重放是唯一诚实的指标；Turn 自包含使重放只需 prompt 重渲染
2. **质量分层与筛选**：outcome 字段 + 宪法档位 → 三档：教师级（≥2档且成功）进蒸馏集 / 普通进 replay buffer / 脏数据剔除。决定下游训练的数据上限
3. **经验提炼（免训练，反馈延迟最短）**：成败轨迹对比 → 提炼 insights → 注入后续 turn 的 ambient 上下文（ExpeL/ERL 路线），当天生效
4. **GRPO 组构造（真实轨迹的独特价值）**：真实数据中同 task 多采样几乎不存在，替代方案——轨迹树锚点（parent_hash 树中相似 world/reports 状态下的分叉形成"伪组"，GiGPO 思路）；跨周重复任务在不同 adapter 版本下的历史 Turn 构成跨版本对比组（兼作灰度评估信号）
5. **反事实变体生成**：对教师级轨迹做观测遮蔽/扰动（去 ambient、污染 reports、插干扰），构造配对样本练鲁棒性；低数据 regime 下是免费扩容
6. **课程生成（闭环终点）**：从失败轨迹反推能力缺口 → 生成针对性演练任务（Absolute Zero 自出题思路，真实失败做种子，不漂移）

### 4.4.1 教师侧测试时扩展策略（已定方向，参数待实测）

不限于 rollout，按任务难度自适应分配空闲算力（调研结论：自适应分配可省约 50% 算力不掉点）：

```
空闲算力分配（按任务难度自适应，参数可配置、实测后调）：
├─ 简单任务：no-think 正常跑（不出教师信号）
├─ 中等任务：thinking mode + 4-8 条并行 rollout + 拒绝式筛选
│   ├─ 可验证任务：ground truth/单测做验证器 → 轨迹按通过率分桶喂 GRPO（0<p<1 的组信息量最大）
│   └─ 不可验证任务：+1 轮自验证自修正（SETS 式），宪法条款做弱验证兜底
└─ 困难任务（失败过的、高价值的）：浅层 MCTS（限深、要求环境可重置）
```

依据：并行 rollout 4-8 次后增益饱和，拒绝式筛选优于均匀投票；SETS（采样→自验证→自修正）scaling 曲线最优（+10.9%）；MCTS 是小模型逼近大模型的少数划算案例但依赖可重置环境；验证器质量是 scaling 天花板——宪法条款正是不可验证任务的外部验证器。佐证：ExACT 证明搜索轨迹回灌蒸馏后学生以更低算力复现约 87% 教师性能。
外推警告：饱和点与增益数字多来自数学/代码域，7B agent 域阈值需实测。

### 4.5 任务来源：通用 TaskSource 接口（已定）

角色分工：**benchmark 负责"练"，真实轨迹负责"考"**。

- benchmark 的独特价值：自带可验证成功标准（GRPO 稀疏奖励直接可用）；冷启动期唯一能立即构造"同 task 多采样组"的来源；可控难度梯度，可按能力缺口定向投喂（衔接第 6 层课程生成）
- 必须守住的边界：benchmark 能力不保证迁移到真实分布（RLVR/pass@k 研究的警告），因此——
  - GRPO 组构造的主要来源（尤其冷启动期）✓
  - 能力缺口的定向训练场 ✓
  - **不进**蒸馏质量闸门 ✗
  - **不作为** adapter 灰度通过与否的判据 ✗

**不预先绑定具体 benchmark**，而是抽象为通用接口（这是 infraend 的第三个插件轴，前两个为推理加速 method、训练器后端）。具体接入哪个源是运行期配置：

```
TaskSource（抽象接口）
├─ meta()  → 源元数据
│    ├─ id / 类别（tool-calling | file-ops | multi-agent | instruction-following）
│    ├─ verifier_type（exact | unit-test | rubric）
│    └─ env_req（none | fs-sandbox | simulator）
├─ list(difficulty?, category?, limit?) → [TaskRef]      按需取任务，支持难度/类别过滤
├─ materialize(task_id) → Task 定义                        关键桥：落到 Turn 的 task/world 字段
├─ reset(task_id) → 环境就绪                               可重置 → 重复 rollout / MCTS 的前提（可选，env_req=none 时可省）
└─ grade(task_id, trajectory) → {success, score?, detail}  供 GRPO 稀疏奖励
```

三条设计约束：

1. **`materialize` 是接口核心**——benchmark 与 NAN agent loop 的唯一边界，任务以 Turn 的 `task`/`world` 结构化注入，agent 不感知自己在跑 benchmark（顺带防"识别评测环境"作弊）
2. **`reset` 可选、`materialize` 核心**——静态 QA 类源声明 `env_req=none` 可省 `reset`；但要用 GRPO 组构造 / MCTS 就必须有可重置环境
3. **split 硬隔离**——源声明 train/eval 分片，**接口层强制 eval 分片不可被训练管道查询**，污染防护做进架构而非纪律

自建任务集（文件操作、多 agent 协作——标准 benchmark 稀缺的两类）将来即同一接口的另一个实现，因此现在无需决定"自建 vs 接入"。

### 4.6 Dream 机制（候选，分优先级）

- **进第一版**：sleep-time compute（空闲算力做轨迹反思/记忆重组，匹配持续开机设定）、经验提炼（ExpeL/ERL 类，成败轨迹对比提炼 insights）
- **远期**：世界模型式 dream（WebEvolver/DynaWeb 类）——长程误差累积 + 本地小算力训世界模型性价比存疑
- **待消融**：反事实观测增强（结构化观测遮蔽/扰动配对训练，低数据 regime 可能有效，agent 场景直接证据少）

### 4.7 关键文献

TTRL / SCoRe / Self-Rewarding LM / Absolute Zero / ReST-ReAct / Search-E1 / DAPO / GSPO / MT-GRPO / GiGPO / PROF / AgentPRM / CHARM / AdvJudge-Zero / Sleep-time Compute / ExpeL / ERL / WebDreamer / WebEvolver / DynaWeb

## 5. 框架架构：补丁式插件内核（已定）

### 5.1 设计哲学

infraend 是"先搭框架再开发功能"——**接口的稳定性比实现的完整性更重要**。插件的性质是**补丁**（侵入改写宿主既有行为），而非 backend Module 那样的**增量**（新起一个服务单元）：

| | backend Module | infraend 插件（补丁） |
|---|---|---|
| 性质 | 增量——新增服务单元 | 侵入——改写宿主既有行为 |
| 状态 | 拥有 `data: DataSpace` | 不拥有独立状态 |
| 生命周期 | `start()` 服务期 + ask/tell | 无独立生命周期，寄生宿主 |
| 卸载 | stop→落盘→释放→解绑 | **零残留还原宿主**（更严格） |

### 5.2 插件内核选型（已定，实测结论）

调研对象 `python-cordis`（PyPI 0.1.4，Cordis 框架的 Python 移植）：已装入项目 venv 实测——hook 四模式与可逆 effect 核心机制真实可用、源码质量好（23KB 薄封装、全类型注解），**但不够成熟**：README 与实际 API 三处不符、版本元数据错乱（0.1.4 wheel 内 `__version__`=0.1.1）、单一维护者、无社区信号。

**决定：直接依赖 pluggy（pytest 同款内核，成熟稳定），自建 infraend 薄内核**，借鉴 python-cordis 的三个好设计：

| 借鉴点 | infraend 用途 |
|---|---|
| 可逆 effect（幂等 disposer、逆序 teardown） | 补丁零残留卸载 |
| waterfall 链式委托（调 `next()` 放行 / 不调即否决） | 多补丁挂同一扩展点的组合语义 |
| `Fiber.refresh()` 反应式协调（依赖出现激活/消失停用） | 训练器按硬件层级自动启停（L0/L1/L2） |

python-cordis 将来若成熟可替换（它本身也是 pluggy 薄封装，迁移成本低）。

### 5.3 补丁契约

```
Patch（补丁契约）
├─ manifest：id / version / capabilities（声明需要宿主先行提供什么）
├─ hookimpl：挂载到 hookspec 定义的扩展点，多补丁同点按 waterfall 组合
└─ effect：注册侧效应（幂等 disposer），卸载零残留
```

**核心约束**：补丁必须有地方可挂——**底座（llama.cpp 二开层）必须先定义扩展点集合（hookspec）**。"优化不分位置"意味着扩展点须覆盖权重 / KV / 调度 / 解码各层。框架先行的第一件实事 = 定义扩展点第一批清单（待决，随底座设计定）。

### 5.4 四层结构与三个插件轴

```
① 宿主层：注册表 / 依赖解析 / 生命周期 / 配置 / 回滚   ← pluggy 薄内核
② 契约层：三个插件轴的抽象接口 + 公共数据契约（Turn 摄取、评分结果、能力清单）
③ 编排层：算力预算调度（推理常态负载 / 教师 rollout / 训练 / sleep-time）
④ 常驻能力：数据管道（Turn 解析→筛选→打包）——框架自带，非插件
```

三个插件轴（均有 2+ 实现预期才开接口；单实现内部逻辑不开，防过度抽象）：

| 轴 | 声明什么 | 已定内容 |
|---|---|---|
| 推理加速 method | hookimpl 挂载点 + capabilities | 横切设计，见 §2 |
| 训练器后端 | 能力层级 L0/L1/L2 | 可插拔，见 §6；按硬件反应式启停 |
| 任务来源 | meta / list / materialize / reset / grade | TaskSource，见 §4.5（资源提供者角色，不 patch 宿主） |

与 backend Module 机制的术语对齐但代码独立（跨进程，不共享实现，形状对齐）。

## 6. 硬件分层（已定）

训练管线分层降级，**飞轮每一环不假设 GRPO 存在**，训练器是可插拔后端（与推理端"插件式加速"同构为"能力插件"模型）：

| 层级 | 硬件 | 能力 |
|---|---|---|
| L0 | 仅有 Mac（MLX/CPU） | 推理 + Turn 写盘 + 数据管道；训练降级为经验提炼（免参数更新，注入上下文） |
| L1 | 单卡 24GB / 高配 Mac | + LoRA SFT 蒸馏（教师轨迹经档位闸门筛选） |
| L2 | 多卡 CUDA | + 完整 GRPO 双奖励训练 |

## 7. 训练工程要点（调研结论）

- 每周从**固定 base 重训新 adapter** + 混 10-20% 旧数据 replay；不要在 adapter 上叠加续训
- 千级轨迹配方：LoRA r=16-32、1-2 epochs、lr ~2e-4、5-10% 验证集早停
- 衔接链路：HF PEFT/Unsloth/MLX 训练 → GGUF 转换（convert_lora_to_gguf.py）→ llama.cpp 热挂载
- GRPO 框架现实约束：verl/OpenRLHF 面向多卡 CUDA；单卡 24GB LoRA 模式勉强可行；Mac 仅 demo 级

## 8. 待决问题

- [ ] 宪法条款集合的内容与演进机制 —— 初稿见 [constitution.md](./constitution.md)（v0.1 草案，待评审）
- [ ] 宪法评分粒度：每条款 0/1 二值化 vs 0-3 离散档位（constitution.md §4，加权聚合已定）
- [x] 慢思考教师的 rollout 预算策略 —— 已定方向见 §4.4.1（自适应分层组合，参数待实测）
- [x] LoRA adapter 版本管理与灰度策略 —— 已定见 §2.1（多实例 + 四级灰度 + 线性版本链）
- [x] Turn schema 契约 —— Turn 结构已定型（见 §3），infraend 解析器跟随其后端定义
- [x] Benchmark/任务来源 —— 已定见 §4.5（通用 TaskSource 接口，具体源运行期接入）
- [ ] 扩展点（hookspec）第一批清单 —— 已定机制见 §5（pluggy 薄内核 + 补丁契约），清单随底座设计定
