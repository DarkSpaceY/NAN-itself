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

### 4.4 Dream 机制（候选，分优先级）

- **进第一版**：sleep-time compute（空闲算力做轨迹反思/记忆重组，匹配持续开机设定）、经验提炼（ExpeL/ERL 类，成败轨迹对比提炼 insights）
- **远期**：世界模型式 dream（WebEvolver/DynaWeb 类）——长程误差累积 + 本地小算力训世界模型性价比存疑
- **待消融**：反事实观测增强（结构化观测遮蔽/扰动配对训练，低数据 regime 可能有效，agent 场景直接证据少）

### 4.5 关键文献

TTRL / SCoRe / Self-Rewarding LM / Absolute Zero / ReST-ReAct / Search-E1 / DAPO / GSPO / MT-GRPO / GiGPO / PROF / AgentPRM / CHARM / AdvJudge-Zero / Sleep-time Compute / ExpeL / ERL / WebDreamer / WebEvolver / DynaWeb

## 5. 硬件分层（已定）

训练管线分层降级，**飞轮每一环不假设 GRPO 存在**，训练器是可插拔后端（与推理端"插件式加速"同构为"能力插件"模型）：

| 层级 | 硬件 | 能力 |
|---|---|---|
| L0 | 仅有 Mac（MLX/CPU） | 推理 + Turn 写盘 + 数据管道；训练降级为经验提炼（免参数更新，注入上下文） |
| L1 | 单卡 24GB / 高配 Mac | + LoRA SFT 蒸馏（教师轨迹经档位闸门筛选） |
| L2 | 多卡 CUDA | + 完整 GRPO 双奖励训练 |

## 6. 训练工程要点（调研结论）

- 每周从**固定 base 重训新 adapter** + 混 10-20% 旧数据 replay；不要在 adapter 上叠加续训
- 千级轨迹配方：LoRA r=16-32、1-2 epochs、lr ~2e-4、5-10% 验证集早停
- 衔接链路：HF PEFT/Unsloth/MLX 训练 → GGUF 转换（convert_lora_to_gguf.py）→ llama.cpp 热挂载
- GRPO 框架现实约束：verl/OpenRLHF 面向多卡 CUDA；单卡 24GB LoRA 模式勉强可行；Mac 仅 demo 级

## 7. 待决问题

- [ ] 宪法条款集合本身的内容与演进机制
- [ ] 慢思考教师的 rollout 预算策略（何时触发、占多少空闲算力）
- [ ] LoRA adapter 版本管理与灰度策略（新 adapter 先部分 agent 试用、按后续 turn 质量对比再全量）
- [ ] Turn 写盘格式与 infraend 解析器的 schema 契约（跟随后端协议 v2 演进）
- [ ] 插件式加速 method 的第一批清单与接口草案
