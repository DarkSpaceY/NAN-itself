"""任务代理模块 — NAN-Agent 的核心任务执行引擎。

本模块实现了 TaskAgent，即 NAN-Agent 系统中的任务执行代理。TaskAgent 接收用户任务，
通过 ReAct（Reasoning + Acting）循环进行推理与行动的迭代执行，最终产出任务结果。

核心组件：
    - AgentState: 代理状态枚举，追踪代理生命周期（初始化 → 运行 → 完成/错误）
    - TaskResult: 任务执行结果的数据类，封装执行状态、输出和度量信息
    - TaskAgent: 任务代理本体，协调以下子系统完成复杂任务：
        · cognition（认知推理）— 提供语言理解与推理能力
        · hard_memory（长期记忆）— 存储和检索历史经验
        · self_value（自我价值系统）— 评估行为与价值观的一致性
        · soft_memory（软记忆/触发器）— 检测好奇心和失败模式
        · action_room（动作执行空间）— 管理可用技能与动作执行
        · got_engine（思维图引擎）— 支持任务委托与思维图推理

执行流程概览：
    初始化 → 技能匹配(Agentic Search) → ReAct 循环 → 自我价值整合 →
    软记忆触发器检查 → 轨迹导出 → 经验提取
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from nan_agent.logging.logger import Timer, bind_correlation_id, clear_correlation_id, get_logger, log_event
from nan_agent.task_agent.react_loop import LoopResult, ReActLoop
from nan_agent.task_agent.trajectory import Trajectory

logger = get_logger(__name__)


class AgentState(str, Enum):
    """代理状态枚举，追踪 TaskAgent 的生命周期状态。

    继承 str 和 Enum，使得状态值既可作为枚举使用，也可直接作为字符串比较。
    状态转换路径：INITIALIZING → RUNNING → FINISHED / ERROR
    """

    INITIALIZING = "initializing"  # 初始化中：代理已创建，尚未开始执行任务
    RUNNING = "running"            # 运行中：代理正在执行 ReAct 循环
    FINISHED = "finished"          # 已完成：任务执行成功结束
    ERROR = "error"                # 错误：任务执行过程中发生异常


@dataclass
class TaskResult:
    """任务执行结果的数据类，封装一次任务运行的完整信息。

    属性：
        task_intent: 任务的原始意图描述
        source: 任务来源标识，默认为 "user"（用户直接发起）
        success: 任务是否成功完成
        result: 任务的最终输出文本
        steps_run: ReAct 循环执行的步骤数
        duration_seconds: 任务执行耗时（秒）
        error: 错误信息，仅在任务失败时填充
    """

    task_intent: str              # 任务的原始意图描述
    source: str = "user"          # 任务来源标识，默认为 "user"（用户直接发起）
    success: bool = False         # 任务是否成功完成
    result: str = ""              # 任务的最终输出文本
    steps_run: int = 0            # ReAct 循环执行的步骤数
    duration_seconds: float = 0.0 # 任务执行耗时（秒）
    error: str = ""               # 错误信息，仅在任务失败时填充


class TaskAgent:
    """任务执行代理 — NAN-Agent 系统的核心任务执行单元。

    TaskAgent 接收用户任务，通过 ReAct（Reasoning + Acting）循环进行迭代推理与行动，
    协调多个子系统（认知、记忆、价值观、动作空间等）完成复杂任务。

    核心属性：
        _agent_id: 代理唯一标识符（UUID）
        _cognition: 认知推理子系统，提供语言理解与推理能力
        _hard_memory: 长期记忆子系统，存储和检索历史经验
        _self_value: 自我价值子系统，评估行为与价值观的一致性
        _soft_memory: 软记忆子系统，包含触发器（好奇心、失败模式检测）
        _action_room: 动作执行空间，管理可用技能与动作执行
        _got_engine: 思维图引擎，支持任务委托与思维图推理
        _state: 当前代理状态（AgentState 枚举）
        _trajectory: 任务执行轨迹，记录每一步的思考、行动和观察
        _aborted: 是否已被中止
        _abort_reason: 中止原因

    使用方式：
        async with TaskAgent(cognition=..., action_room=...) as agent:
            result = await agent.run("执行某项任务")
    """

    def __init__(
        self,
        cognition,
        hard_memory=None,
        self_value=None,
        soft_memory=None,
        action_room=None,
        got_engine=None,
    ):
        """初始化任务代理。

        Args:
            cognition: 认知推理子系统（必需），提供语言理解与推理能力
            hard_memory: 长期记忆子系统（可选），用于经验存储与检索
            self_value: 自我价值子系统（可选），用于价值观一致性评估
            soft_memory: 软记忆子系统（可选），包含好奇心和失败模式触发器
            action_room: 动作执行空间（可选），管理可用技能与动作执行
            got_engine: 思维图引擎（可选），支持任务委托与思维图推理
        """
        self._agent_id = str(uuid.uuid4())
        self._cognition = cognition
        self._hard_memory = hard_memory
        self._self_value = self_value
        self._soft_memory = soft_memory
        self._action_room = action_room
        self._got_engine = got_engine

        # 将 action_room 中的技能管理器传递给认知子系统，用于上下文丰富
        if action_room and hasattr(action_room, "skill_manager"):
            cognition.skill_trees = action_room.skill_manager

        self._state = AgentState.INITIALIZING
        self._trajectory: Optional[Trajectory] = None
        self._aborted = False
        self._abort_reason = ""

        log_event(
            logger,
            "task_agent_created",
            agent_id=self._agent_id,
            state=self._state.value,
        )

    @property
    def state(self) -> AgentState:
        """获取代理当前状态。

        Returns:
            AgentState: 代理当前的生命周期状态
        """
        return self._state

    async def run(
        self,
        task: str,
        max_rounds: int = 20,
        source: str = "user",
    ) -> TaskResult:
        """执行任务的主入口方法。

        绑定关联 ID 用于日志追踪，委托 _run_impl 执行实际逻辑，
        并在完成后清理关联 ID。

        Args:
            task: 任务描述文本
            max_rounds: ReAct 循环最大轮次数，默认 20
            source: 任务来源标识，默认 "user"

        Returns:
            TaskResult: 包含执行状态、输出和度量信息的任务结果
        """
        start_time = time.time()

        # 绑定关联 ID，使本次任务的所有日志可被追踪
        bind_correlation_id(f"ta-{self._agent_id}")
        try:
            return await self._run_impl(task, max_rounds, source, start_time)
        finally:
            # 确保无论成功或异常，都清理关联 ID
            clear_correlation_id()

    async def _run_impl(
        self,
        task: str,
        max_rounds: int,
        source: str,
        start_time: float,
    ) -> TaskResult:
        """任务执行的核心实现方法。

        执行流程：
            1. 状态初始化 → 将代理状态切换为 RUNNING
            2. 技能匹配 → 通过 Agentic Search 预匹配相关技能
            3. ReAct 循环 → 迭代执行推理与行动，直到任务完成或达到最大轮次
            4. 自我价值整合 → 评估任务结果与价值观的一致性，必要时学习与修正
            5. 软记忆触发器检查 → 检测好奇心触发和失败模式
            6. 轨迹导出 → 将执行轨迹持久化到文件
            7. 经验提取 → 从执行轨迹中提取高质量经验并存入长期记忆

        Args:
            task: 任务描述文本
            max_rounds: ReAct 循环最大轮次数
            source: 任务来源标识
            start_time: 任务开始时间戳

        Returns:
            TaskResult: 包含执行状态、输出和度量信息的任务结果
        """
        # ---- 步骤 1：状态初始化 ----
        self._state = AgentState.RUNNING
        self._aborted = False
        self._abort_reason = ""

        # 创建执行轨迹对象，记录任务意图和来源
        self._trajectory = Trajectory(task_intent=task, source=source)

        log_event(
            logger,
            "task_agent_run_start",
            agent_id=self._agent_id,
            task=task[:100],
            source=source,
        )

        # ---- 步骤 2：Agentic Search 技能匹配 ----
        # 在 ReAct 循环开始前，通过技能管理器预匹配与任务相关的技能，
        # 为认知子系统提供上下文丰富信息
        if self._action_room and hasattr(self._action_room, 'skill_manager') and self._action_room.skill_manager:
            try:
                matched_skills = await self._action_room.skill_manager.agentic_search(task, self._cognition)
                if matched_skills:
                    logger.info("agentic_skills_matched", count=1)
            except Exception as e:
                logger.warning("agentic_skill_search_failed", error=str(e))

        # ---- 步骤 3~7：核心执行流程 ----
        with Timer(logger, "task_agent_run", warn_threshold_ms=30000):
            try:
                # ---- 步骤 3：ReAct 循环 ----
                # 构建并运行 ReAct 循环，传入各子系统引用
                react_loop = ReActLoop(
                    cognition=self._cognition,
                    action_room=self._action_room,
                    trajectory=self._trajectory,
                    got_engine=self._got_engine,
                    soft_memory=self._soft_memory,
                )
                # 同步中止状态到 ReAct 循环
                react_loop._aborted = self._aborted
                react_loop._abort_reason = self._abort_reason

                with Timer(logger, "react_loop_step", warn_threshold_ms=30000):
                    finish_reason = await react_loop.step(
                        input_text=task,
                        task_intent=task,
                        max_rounds=max_rounds,
                    )

                # 解析 ReAct 循环的完成原因
                if isinstance(finish_reason, LoopResult):
                    loop_ok = finish_reason.success
                    final_output = finish_reason.final_output
                    loop_error = finish_reason.error
                else:
                    # 兼容旧版返回格式（直接返回字符串）
                    loop_ok = True
                    final_output = finish_reason or f"Task completed: {task}"
                    loop_error = ""

                # 将 ReAct 循环中的步骤同步到轨迹对象
                self._sync_trajectory_from_react_loop(react_loop)

                # ---- 步骤 4：自我价值整合 ----
                # 评估任务结果与代理价值观的一致性，
                # 若不一致度超过阈值则触发学习和价值观修正
                if self._self_value is not None:
                    with Timer(logger, "self_value_integration", warn_threshold_ms=5000):
                        try:
                            # 构建事件文本，包含任务和结果的摘要
                            event_text = f"Task: {task}. Result: {final_output}"
                            # 处理事件，更新价值观系统的内部状态
                            await self._self_value.process_event(event_text)

                            # 获取所有相关价值观名称，检查不一致度
                            relevant_values = [v.name for v in self._self_value.values.list_all()]
                            dissonance = await self._self_value.check_dissonance(
                                event_text, relevant_values,
                            )

                            # 不一致度超过 0.3 阈值时，触发学习与修正
                            if dissonance.get("overall_dissonance", 0) > 0.3:
                                await self._self_value.learn_from_dissonance(
                                    event_text, dissonance,
                                )

                                # 如果价值观系统判断需要修正价值观本身
                                if self._self_value.should_refine_values():
                                    await self._self_value.refine_values()
                        except Exception as e:
                            logger.warning("self_value_integration_error", error=str(e))

                # ---- 步骤 5：软记忆触发器检查 ----
                # 检查好奇心触发和失败模式检测
                if self._soft_memory and hasattr(self._soft_memory, '_triggers') and self._soft_memory._triggers:
                    try:
                        # 好奇心触发检查：检测输出中是否包含值得深入探索的内容
                        curiosity_check = self._soft_memory._triggers.trigger_curiosity(final_output)
                        if curiosity_check:
                            log_event(logger, "curiosity_triggered", detail=str(curiosity_check)[:100])

                        # 失败模式检测：检测输出中是否包含已知的失败模式
                        failure_check = self._soft_memory._triggers.check_failure(task)
                        if failure_check:
                            log_event(logger, "failure_pattern_detected", detail=task[:100])
                    except Exception as e:
                        logger.warning("soft_memory_triggers_failed", error=str(e))

                # ---- 步骤 6 & 7：轨迹导出与经验提取 ----
                with Timer(logger, "post_task_processing", warn_threshold_ms=5000):
                    # 步骤 6：轨迹导出 — 将执行轨迹持久化为 JSON 文件
                    if self._trajectory is not None:
                        try:
                            summary = self._trajectory.to_summary()
                            json_output = self._trajectory.to_json()
                            logger.info("trajectory_summary", summary=summary[:200])
                            import os
                            os.makedirs("data/trajectories", exist_ok=True)
                            traj_file = f"data/trajectories/{self._agent_id}_{int(start_time)}.json"
                            with open(traj_file, "w") as f:
                                f.write(json_output)
                        except Exception as e:
                            logger.debug("trajectory_export_failed", error=str(e))

                    # 步骤 7：记忆存储 — 将任务轨迹摘要存入记忆
                    if self._hard_memory is not None:
                        try:
                            trajectory_text = self._trajectory.to_summary() if self._trajectory else ""
                            if trajectory_text:
                                emotional_state = None
                                if self._cognition is not None and self._cognition.self_value is not None:
                                    try:
                                        emotional_state = self._cognition.self_value.get_emotional_state()
                                    except Exception as e:
                                        logger.debug("agent_emotional_state_failed", error=str(e))
                                await self._hard_memory.add_memcell(
                                    trajectory_text, source="task_trajectory",
                                    emotional_state=emotional_state,
                                    timestamp=datetime.now(timezone.utc).isoformat(),
                                )
                                logger.info("task_trajectory_stored")
                        except Exception as e:
                            logger.debug("task_trajectory_store_failed", error=str(e))

                # ---- 任务完成，更新状态并返回结果 ----
                self._state = AgentState.FINISHED

                duration = time.time() - start_time
                steps_run = len(react_loop.steps)

                log_event(
                    logger,
                    "task_agent_run_complete",
                    agent_id=self._agent_id,
                    duration=duration,
                    steps=steps_run,
                )

                return TaskResult(
                    task_intent=task,
                    source=source,
                    success=loop_ok,
                    result=final_output,
                    steps_run=steps_run,
                    duration_seconds=duration,
                    error=loop_error,
                )

            except Exception as e:
                # 任务执行异常，将状态切换为 ERROR
                self._state = AgentState.ERROR
                duration = time.time() - start_time
                logger.exception(
                    "task_agent_run_error",
                    agent_id=self._agent_id,
                    error=str(e),
                )

                return TaskResult(
                    task_intent=task,
                    source=source,
                    success=False,
                    result="",
                    steps_run=len(self._trajectory.steps) if self._trajectory else 0,
                    duration_seconds=duration,
                    error=str(e),
                )

    def _sync_trajectory_from_react_loop(self, react_loop: ReActLoop) -> None:
        """将 ReAct 循环中的步骤同步到代理的轨迹对象。

        遍历 ReAct 循环的每一步，将观察、思考、行动和行动结果
        转换为可序列化的格式后添加到轨迹对象中。

        Args:
            react_loop: 已执行完毕的 ReAct 循环实例
        """
        if self._trajectory is None:
            return

        for step in react_loop.steps:
            # 将观察结果序列化为 JSON 字符串，失败则回退为 str()
            observation = ""
            if step.observation:
                try:
                    import json
                    observation = json.dumps(step.observation, ensure_ascii=False)
                except Exception:
                    observation = str(step.observation)

            thought = step.thought or ""
            action = step.action or ""
            # 将行动结果序列化为 JSON 字符串，失败则回退为 str()
            action_result = ""
            if step.result:
                try:
                    import json
                    action_result = json.dumps(step.result, ensure_ascii=False)
                except Exception:
                    action_result = str(step.result)

            self._trajectory.add_step(
                observation=observation,
                thought=thought,
                action=action,
                action_result=action_result,
            )

    async def run_multi_modal(
        self,
        multimodal_input,
        max_rounds: int = 20,
    ) -> TaskResult:
        """处理多模态输入并执行任务。

        从多模态输入对象中提取文本内容，然后委托给 run 方法执行。
        如果输入对象不支持 get_text() 方法，则回退为 str() 转换。

        Args:
            multimodal_input: 多模态输入对象，需支持 get_text() 方法
            max_rounds: ReAct 循环最大轮次数，默认 20

        Returns:
            TaskResult: 包含执行状态、输出和度量信息的任务结果
        """
        # 尝试从多模态输入中提取文本，失败则回退为字符串转换
        text = ""
        try:
            text = multimodal_input.get_text()
        except Exception:
            text = str(multimodal_input)

        logger.info(
            "run_multi_modal",
            agent_id=self._agent_id,
            text_preview=text[:100],
        )

        return await self.run(task=text, max_rounds=max_rounds)

    def delegate_to_got(self, task: str):
        """将任务委托给思维图（GoT）引擎。

        当任务适合以思维图方式处理时，可调用此方法将任务委托给 GoT 引擎，
        由引擎创建对应的任务节点。

        Args:
            task: 要委托的任务描述

        Returns:
            思维图引擎创建的任务节点，若引擎不可用则返回 None
        """
        if self._got_engine is None:
            logger.warning(
                "delegate_to_got_no_engine",
                agent_id=self._agent_id,
            )
            return None

        log_event(
            logger,
            "delegate_to_got",
            agent_id=self._agent_id,
            task=task[:100],
        )

        return self._got_engine.create_task_node(task)

    def abort(self, reason: str = "") -> None:
        """中止代理的任务执行。

        设置中止标志和原因，正在运行的 ReAct 循环将在下一次迭代时检查此标志并停止。

        Args:
            reason: 中止原因描述，默认为空字符串
        """
        self._aborted = True
        self._abort_reason = reason
        logger.info(
            "task_agent_aborted",
            agent_id=self._agent_id,
            reason=reason,
        )

    def destroy(self) -> None:
        """销毁代理，释放资源。

        将状态设为 FINISHED，清空轨迹对象和中止标志，
        使代理不再可用。通常在 async with 退出时自动调用。
        """
        self._state = AgentState.FINISHED
        self._trajectory = None
        self._aborted = False
        self._abort_reason = ""
        logger.info(
            "task_agent_destroyed",
            agent_id=self._agent_id,
        )

    async def __aenter__(self) -> "TaskAgent":
        """异步上下文管理器入口，返回代理实例本身。"""
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """异步上下文管理器出口，自动销毁代理释放资源。"""
        self.destroy()