"""task_agent — NAN-Agent 的任务执行模块

本包负责单个任务的完整执行生命周期，包括：
  - 任务代理（TaskAgent）：接收任务并驱动执行
  - 推理循环（ReActLoop）：实现 ReAct（Reasoning + Acting）迭代推理
  - 执行轨迹（Trajectory）：记录任务执行过程中的完整步骤链
  - 步骤记录（StepRecord）：记录单次推理/行动的详细信息
  - 代理状态（AgentState）：代理运行时的状态枚举
  - 任务结果（TaskResult）与循环结果（LoopResult）：执行产出的结构化结果

导出的核心组件：
  TaskAgent, ReActLoop, Trajectory, StepRecord, AgentState, TaskResult, LoopResult, TrajectoryStep
"""

# 导入任务代理核心类与状态/结果定义
from nan_agent.task_agent.agent import AgentState, TaskAgent, TaskResult
# task_agent 包 — 上下文增强已由 cognition.enrich_task_context() 处理
from nan_agent.task_agent.react_loop import LoopResult, ReActLoop  # 导入推理循环及其结果类型
from nan_agent.task_agent.trajectory import StepRecord, Trajectory, TrajectoryStep  # 导入执行轨迹、步骤记录及步骤别名

__all__ = [
    "TaskAgent",      # 任务代理 — 任务的顶层执行入口
    "ReActLoop",      # 推理循环 — ReAct 迭代推理引擎
    "Trajectory",     # 执行轨迹 — 完整的步骤链记录
    "StepRecord",     # 步骤记录 — 单次推理/行动的详细记录
    "AgentState",     # 代理状态 — 运行时状态枚举
    "TaskResult",     # 任务结果 — 任务执行的结构化产出
    "LoopResult",     # 循环结果 — 单轮推理循环的产出
    "TrajectoryStep", # 轨迹步骤 — StepRecord 的别名，用于语义化表达
]