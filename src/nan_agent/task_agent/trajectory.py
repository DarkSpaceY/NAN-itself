"""任务执行轨迹记录模块。

本模块负责记录 TaskAgent 执行任务的完整过程，提供轨迹的构建、序列化与反序列化能力。

核心组件：
    - StepRecord: 单步执行记录数据类，包含步骤编号、时间戳、动作、观察、思考、结果等字段
    - TrajectoryStep: StepRecord 的别名，用于语义化表达
    - Trajectory: 任务执行轨迹记录器，管理整个任务执行过程的步骤序列，
      支持添加步骤、标记完成、生成快照/摘要、序列化与反序列化等操作

动作类型（VALID_ACTIONS）：
    - observe: 观察环境，获取信息
    - think: 思考推理，形成决策
    - act: 执行动作，调用工具
    - finish: 完成任务，记录结果
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nan_agent.exceptions import ActionError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StepRecord:
    """单步执行记录数据类。

    记录 TaskAgent 在任务执行过程中每一步的完整信息，包括步骤编号、
    时间戳、动作类型、观察结果、思考内容、执行参数和结果等。

    属性：
        step_number: 步骤序号，从 1 开始递增
        timestamp: 步骤发生的时间戳（ISO 8601 格式，UTC 时区）
        action: 动作类型，取值为 observe / think / act / finish 之一
        observation: 观察结果字典，默认为空字典
        thought: 该步骤的思考/推理内容，默认为空字符串
        action_type: 具体的动作类型名称（如工具名），默认为空字符串
        action_params: 动作参数字典，默认为空字典
        result: 该步骤的执行结果，默认为空字符串
        tool_name: 使用的工具名称，仅在 action 为 act 时有效，默认为空字符串
        duration_ms: 该步骤的执行耗时（毫秒），默认为 0.0
    """

    step_number: int
    timestamp: str
    action: str
    observation: dict = field(default_factory=dict)
    thought: str = ""
    action_type: str = ""
    action_params: dict = field(default_factory=dict)
    result: Any = ""
    tool_name: str = ""
    duration_ms: float = 0.0


# TrajectoryStep 是 StepRecord 的语义化别名，用于在轨迹上下文中表达"轨迹步骤"的概念
TrajectoryStep = StepRecord


class Trajectory:
    """任务执行轨迹记录器。

    记录 TaskAgent 执行任务的完整过程，包括每一步的观察、思考、动作和结果。
    支持轨迹的构建、完成标记、快照生成、摘要输出、序列化与反序列化。

    核心属性：
        task_intent: 任务意图描述（只读）
        source: 任务来源，"user"（用户直接发起）或 "got_action"（思维图引擎委托）（只读）
        steps: 步骤记录列表的副本（只读）
        result: 最终结果文本（只读）

    使用方式：
        1. 创建轨迹：traj = Trajectory(task_intent="...", source="user")
        2. 添加步骤：traj.add_step(observation="...", thought="...", action="...", action_result="...")
        3. 完成轨迹：traj.finish(result="任务完成")
        4. 获取快照/摘要：traj.to_snapshot() / traj.to_summary()
        5. 序列化/反序列化：traj.to_json() / Trajectory.from_dict(data)
    """

    # 合法的动作类型集合：observe（观察）、think（思考）、act（执行）、finish（完成）
    VALID_ACTIONS = frozenset({"observe", "think", "act", "finish"})

    def __init__(self, task_intent: str, source: str = "user"):
        """初始化任务执行轨迹。

        参数：
            task_intent: 任务意图描述，不可为空
            source: 任务来源，必须为 "user"（用户直接发起）或 "got_action"（思维图引擎委托），
                    默认为 "user"

        异常：
            ActionError: 当 task_intent 为空或 source 不合法时抛出
        """
        if not task_intent:
            raise ActionError("task_intent must not be empty", error_code="E501")
        if source not in ("user", "got_action"):
            raise ActionError(
                f"source must be 'user' or 'got_action', got '{source}'",
                error_code="E501",
            )
        self._task_intent = task_intent
        self._source = source
        self._steps: list[StepRecord] = []  # 步骤记录列表
        self._started_at = datetime.now(timezone.utc)  # 轨迹开始时间（UTC）
        self._finished_at: datetime | None = None  # 轨迹完成时间（UTC），未完成时为 None
        self._finished = False  # 轨迹是否已完成
        self._result = ""  # 最终结果文本
        self._agent_id = ""  # 关联的 Agent ID

    @property
    def task_intent(self) -> str:
        """获取任务意图描述。"""
        return self._task_intent

    @property
    def source(self) -> str:
        """获取任务来源（"user" 或 "got_action"）。"""
        return self._source

    @property
    def steps(self) -> list[StepRecord]:
        """获取步骤记录列表的副本（修改副本不影响原始数据）。"""
        return list(self._steps)

    @property
    def result(self) -> str:
        """获取最终结果文本。"""
        return self._result

    def finish(self, result: str = "") -> StepRecord:
        """标记轨迹完成，记录最终结果。

        将轨迹标记为已完成状态，记录完成时间和最终结果，
        并自动追加一个 action 为 "finish" 的步骤记录。

        参数：
            result: 最终结果文本，默认为空字符串

        返回：
            StepRecord: 新创建的 finish 步骤记录

        异常：
            ActionError: 当轨迹已经完成时再次调用会抛出异常
        """
        if self._finished:
            raise ActionError(
                "Cannot finish: trajectory is already finished",
                error_code="E501",
            )
        self._result = result
        self._finished = True
        self._finished_at = datetime.now(timezone.utc)
        # 创建 finish 步骤并追加到步骤列表
        step = StepRecord(
            step_number=len(self._steps) + 1,
            timestamp=self._finished_at.isoformat(),
            action="finish",
            result=result,
        )
        self._steps.append(step)
        return step

    def to_snapshot(self) -> dict:
        """生成完整的轨迹快照字典。

        将轨迹的所有信息序列化为字典，包括元数据、步骤列表和统计信息。
        统计信息包括步骤类型分布（observe/think/act/finish 各多少步）、
        使用的工具列表和总执行动作数。

        返回：
            dict: 包含以下键的快照字典：
                - task_intent: 任务意图
                - source: 任务来源
                - agent_id: Agent ID
                - started_at: 开始时间（ISO 8601）
                - finished_at: 完成时间（ISO 8601），未完成时为 None
                - finished: 是否已完成
                - result: 最终结果
                - steps: 步骤记录列表
                - total_steps: 总步骤数
                - duration_seconds: 总耗时（秒）
                - step_type_distribution: 各动作类型的步骤数分布
                - tools_used: 使用的工具名称列表（去重排序）
                - total_actions_count: 总执行动作数（action 为 act 的步骤数）
        """
        duration_seconds = 0.0
        # 计算总耗时：已完成时用完成时间减去开始时间
        if self._finished and self._finished_at is not None:
            duration_seconds = (
                self._finished_at - self._started_at
            ).total_seconds()
        # 未完成时用最后一个步骤的时间戳估算耗时
        elif self._steps:
            last_ts = self._steps[-1].timestamp
            try:
                last_dt = datetime.fromisoformat(last_ts)
                duration_seconds = (
                    last_dt - self._started_at
                ).total_seconds()
            except (ValueError, TypeError) as e:
                logger.debug("trajectory_duration_parse_failed", timestamp=last_ts, error=str(e))
                duration_seconds = 0.0

        # 将每个步骤转换为字典
        step_dicts = []
        for s in self._steps:
            step_dicts.append(
                {
                    "step_number": s.step_number,
                    "timestamp": s.timestamp,
                    "action": s.action,
                    "observation": s.observation,
                    "thought": s.thought,
                    "action_type": s.action_type,
                    "action_params": s.action_params,
                    "result": s.result,
                    "tool_name": s.tool_name,
                    "duration_ms": s.duration_ms,
                }
            )

        # 统计各动作类型的步骤数
        action_counts = Counter(s.action for s in self._steps)
        step_type_distribution = {
            action: action_counts.get(action, 0)
            for action in self.VALID_ACTIONS
        }

        # 收集所有使用过的工具名称（去重排序）
        tools_used = sorted(
            {s.tool_name for s in self._steps if s.action == "act" and s.tool_name}
        )

        # 统计总执行动作数（action 为 act 的步骤数）
        total_actions_count = sum(
            1 for s in self._steps if s.action == "act"
        )

        return {
            "task_intent": self._task_intent,
            "source": self._source,
            "agent_id": self._agent_id,
            "started_at": self._started_at.isoformat(),
            "finished_at": self._finished_at.isoformat() if self._finished_at else None,
            "finished": self._finished,
            "result": self._result,
            "steps": step_dicts,
            "total_steps": len(self._steps),
            "duration_seconds": duration_seconds,
            "step_type_distribution": step_type_distribution,
            "tools_used": tools_used,
            "total_actions_count": total_actions_count,
        }

    def add_step(
        self,
        observation: str = "",
        thought: str = "",
        action: str = "",
        action_result: str = "",
    ) -> StepRecord:
        """添加一个执行步骤到轨迹中。

        根据传入的参数自动推断动作类型（observe/think/act）：
        - 若 action 为空，推断为 "observe"（纯观察步骤）
        - 若 action 为 "observe" 或 "think"，保持原值
        - 若 action 为其他值（如工具名），推断为 "act"（执行动作步骤）

        参数：
            observation: 观察内容文本，会被包装为 {"raw": observation} 存入 observation 字段
            thought: 思考/推理内容
            action: 动作名称，用于推断动作类型和记录工具名称
            action_result: 动作执行结果

        返回：
            StepRecord: 新创建的步骤记录

        异常：
            ActionError: 当轨迹已完成时再次添加步骤会抛出异常
        """
        if self._finished:
            raise ActionError(
                "Cannot add step: trajectory is already finished",
                error_code="E501",
            )
        step = StepRecord(
            step_number=len(self._steps) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            # 动作类型推断逻辑：空值→observe，observe/think→保持原值，其他→act
            action="act" if action and action != "observe" and action != "think" else (action or "observe"),
            # 将观察文本包装为字典格式
            observation={"raw": observation} if observation else {},
            thought=thought,
            action_type=action,
            result=action_result,
            # 工具名称：仅当 action 为实际工具调用时才记录，observe/think/空值不记录
            tool_name=action if action not in ("observe", "think", "") else "",
        )
        self._steps.append(step)
        return step

    def to_summary(self) -> str:
        """生成人类可读的轨迹摘要文本。

        基于快照数据生成格式化的多行文本，包含任务意图、来源、状态、
        时间信息、步骤分布、结果和工具使用情况。

        返回：
            str: 格式化的轨迹摘要文本
        """
        snapshot = self.to_snapshot()
        lines = [
            f"Task: {snapshot['task_intent']}",
            f"Source: {snapshot['source']}",
            f"Status: {'Finished' if snapshot['finished'] else 'In Progress'}",
            f"Started: {snapshot['started_at']}",
        ]
        if snapshot["finished_at"]:
            lines.append(f"Finished: {snapshot['finished_at']}")
        lines.append(f"Total Steps: {snapshot['total_steps']}")
        lines.append(
            f"Duration: {snapshot['duration_seconds']:.2f}s"
        )
        lines.append("Step Distribution:")
        for action, count in snapshot["step_type_distribution"].items():
            lines.append(f"  {action}: {count}")
        if snapshot["result"]:
            lines.append(f"Result: {snapshot['result']}")
        if snapshot["tools_used"]:
            lines.append(f"Tools Used: {', '.join(snapshot['tools_used'])}")
        if snapshot["total_actions_count"]:
            lines.append(
                f"Total Actions: {snapshot['total_actions_count']}"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        """将轨迹序列化为 JSON 字符串。

        基于 to_snapshot() 生成的字典进行 JSON 序列化，
        使用 2 空格缩进，确保非 ASCII 字符不被转义。

        返回：
            str: 格式化的 JSON 字符串
        """
        return json.dumps(self.to_snapshot(), indent=2, default=str, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        """从字典反序列化恢复 Trajectory 对象。

        将 to_snapshot() 生成的字典还原为完整的 Trajectory 实例，
        包括元数据、时间信息和所有步骤记录。

        参数：
            data: 包含轨迹数据的字典，至少需要 "task_intent" 键，
                  可选键包括 "source"、"agent_id"、"started_at"、"finished_at"、
                  "finished"、"result"、"steps" 等

        返回：
            Trajectory: 恢复后的轨迹对象
        """
        traj = cls(
            task_intent=data["task_intent"],
            source=data.get("source", "user"),
        )
        # 恢复 Agent ID
        traj._agent_id = data.get("agent_id", "")

        # 恢复开始时间
        started_at_str = data.get("started_at")
        if started_at_str:
            traj._started_at = datetime.fromisoformat(started_at_str)

        # 恢复完成时间
        finished_at_str = data.get("finished_at")
        if finished_at_str:
            traj._finished_at = datetime.fromisoformat(finished_at_str)

        # 恢复完成状态和结果
        traj._finished = data.get("finished", False)
        traj._result = data.get("result", "")

        # 逐条恢复步骤记录
        for step_data in data.get("steps", []):
            step = StepRecord(
                step_number=step_data.get("step_number", 0),
                timestamp=step_data.get("timestamp", ""),
                action=step_data.get("action", ""),
                observation=step_data.get("observation", {}),
                thought=step_data.get("thought", ""),
                action_type=step_data.get("action_type", ""),
                action_params=step_data.get("action_params", {}),
                result=step_data.get("result", ""),
                tool_name=step_data.get("tool_name", ""),
                duration_ms=step_data.get("duration_ms", 0.0),
            )
            traj._steps.append(step)

        return traj

    def __repr__(self) -> str:
        """返回轨迹对象的简洁字符串表示，包含任务意图、来源、步骤数和状态。"""
        status = "finished" if self._finished else "active"
        return (
            f"Trajectory(task_intent={self._task_intent!r}, "
            f"source={self._source!r}, "
            f"steps={len(self._steps)}, "
            f"status={status})"
        )