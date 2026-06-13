"""ReAct 推理循环模块

本模块实现了 ReAct（Reasoning + Acting）推理循环，是 TaskAgent 的核心执行引擎。
每轮循环执行三个阶段：observe（观察环境）→ think（推理决策）→ act（执行动作），
直到任务完成或达到最大轮次限制。

核心组件：
    - ReActStep: 单步记录数据类，记录 observe/think/act 各阶段的信息
    - LoopResult: 循环执行结果数据类，包含成功状态、最终输出、运行轮次等
    - ReActLoop: ReAct 推理循环主类，管理整个迭代过程，包括环境观察、
      认知推理、工具执行、元认知检查、GoT 委派等

ReAct 循环流程：
    1. observe — 通过 ActionRoom 获取环境观察（屏幕截图、摄像头、语音、文件系统等）
    2. think — 调用认知模型进行推理，支持多模态输入（文本+图像+DoT可视化）
    3. act — 通过 ActionRoom 执行工具调用
    4. 重复以上步骤，直到模型输出 finish 动作或达到最大轮次
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nan_agent.action_room.interface import ActionRequest, ActionRoom
from nan_agent.logging.logger import Timer, get_logger, log_event
from nan_agent.model.cognition import Cognition
from nan_agent.model.types import MultiModalInput

logger = get_logger(__name__)


@dataclass
class ReActStep:
    """ReAct 单步记录数据类

    记录 ReAct 循环中每一步的 observe/think/act 各阶段信息，
    用于追踪推理过程、构建对话历史和记录轨迹。

    属性：
        step_idx: 步骤序号（从 1 开始）
        phase: 当前阶段标识，取值为 "observe" / "think" / "act"
        observation: observe 阶段获取的环境观察数据，包含传感器信息等
        thought: think 阶段模型的推理文本
        action: act 阶段执行的动作名称（工具名 / "think" / "finish"）
        action_params: act 阶段动作的参数字典
        result: act 阶段动作的执行结果
        timestamp: 该步骤创建的时间戳
    """
    step_idx: int
    phase: str
    observation: Optional[Dict[str, Any]] = None
    thought: Optional[str] = None
    action: Optional[str] = None
    action_params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoopResult:
    """ReAct 循环执行结果数据类

    封装整个 ReAct 循环的执行结果，供调用方判断任务是否成功完成。

    属性：
        success: 任务是否成功完成（True 表示正常结束，False 表示异常或达到最大轮次）
        final_output: 最终输出文本，成功时为任务结果，失败时为错误原因
        rounds_run: 实际运行的轮次数
        error: 错误信息，空字符串表示无错误
    """
    success: bool = False
    final_output: str = ""
    rounds_run: int = 0
    error: str = ""


class ReActLoop:
    """ReAct 推理循环主类

    实现 ReAct（Reasoning + Acting）推理循环，是 TaskAgent 的核心执行引擎。
    每轮循环依次执行 observe → think → act 三个阶段，直到任务完成或达到最大轮次。

    核心属性：
        _cognition: 认知模型实例，负责推理和决策
        _action_room: 动作执行环境，负责工具调用和环境观察
        _trajectory: 轨迹记录对象，用于记录每步的 observe/think/act 信息
        _steps: 当前循环的所有步骤记录列表
        _finished: 循环是否已结束
        _finish_reason: 循环结束的原因
        _got_engine: Graph-of-Thought 引擎，用于复杂任务分解和元认知检查
        _soft_memory: 软记忆模块，用于存储和检索中间结果

    使用方式：
        loop = ReActLoop(cognition, action_room, trajectory=traj)
        result = await loop.step("用户输入", max_rounds=20)
    """

    def __init__(
        self,
        cognition: Cognition,
        action_room: ActionRoom,
        trajectory: Optional[Any] = None,
        got_engine=None,
        soft_memory=None,
    ):
        """初始化 ReAct 循环

        参数：
            cognition: 认知模型实例，提供推理和决策能力
            action_room: 动作执行环境，提供工具调用和环境观察能力
            trajectory: 可选的轨迹记录对象，用于记录推理过程
            got_engine: 可选的 Graph-of-Thought 引擎，用于复杂任务分解
            soft_memory: 可选的软记忆模块，用于存储中间结果
        """
        self._cognition = cognition
        self._action_room = action_room

        self._trajectory = trajectory
        self._steps: List[ReActStep] = []  # 当前循环的所有步骤记录
        self._finished = False  # 循环是否已结束的标志
        self._finish_reason = ""  # 循环结束的原因描述

        self._got_engine = got_engine
        self._soft_memory = soft_memory

    @property
    def finish_reason(self) -> str:
        """获取循环结束的原因描述"""
        return self._finish_reason

    @property
    def steps(self) -> List[ReActStep]:
        """获取当前循环的所有步骤记录（返回副本，避免外部修改）"""
        return self._steps.copy()

    async def step(
        self,
        input_text: str,
        task_intent: str = "",
        max_rounds: int = 20,
    ) -> LoopResult:
        """ReAct 主循环入口，管理整个推理-行动迭代过程

        每轮执行 observe → think → act 三阶段循环，直到任务完成或达到最大轮次。
        支持元认知质量检查、GoT 任务委派、软记忆反馈等高级功能。

        参数：
            input_text: 用户输入的原始任务文本
            task_intent: 任务意图描述，若为空则尝试通过认知模型推断
            max_rounds: 最大循环轮次，默认 20

        返回：
            LoopResult: 循环执行结果，包含成功状态、最终输出、运行轮次等
        """
        # 重置循环状态
        self._steps.clear()
        self._finished = False
        self._finish_reason = ""

        conversation_history: List[str] = []  # 对话历史，用于构建上下文
        tools_description = self.get_tools_description(self._action_room)  # 获取可用工具列表描述
        rounds_run = 0

        # 存储用户输入
        if self._cognition.hard_memory is not None and input_text:
            try:
                await self._cognition.hard_memory.add_memcell(
                    str(input_text), source="user_input",
                    emotional_state=self._get_emotional_state(),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as e:
                logger.warning("react_user_input_store_failed", error=str(e))

        for round_idx in range(max_rounds):
            # 检查是否被外部中止
            if getattr(self, '_aborted', False):
                self._finished = True
                self._finish_reason = f"Aborted: {getattr(self, '_abort_reason', 'unknown')}"
                log_event(logger, "react_loop_aborted", reason=self._finish_reason)
                return LoopResult(
                    success=False,
                    final_output=self._finish_reason,
                    rounds_run=rounds_run,
                    error="aborted",
                )
            step_idx = round_idx + 1

            # ── 阶段 1: observe — 观察环境 ──
            with Timer(logger, "observe", warn_threshold_ms=3000):
                observation = await self._observe(self._action_room)
            observe_step = ReActStep(
                step_idx=step_idx,
                phase="observe",
                observation=observation,
            )
            self._steps.append(observe_step)

            # 通过认知模型丰富任务上下文（注入人格、情绪、记忆等信息）
            context = await self._cognition.enrich_task_context(
                input_text, task_intent,
            )

            # 构建推理提示词，分为 fixed（不可压缩）和 compressible（可压缩）两部分
            prompt = await self.build_reasoning_prompt(
                context=context,
                tools_desc=tools_description,
                step_idx=step_idx,
                history="\n".join(conversation_history[-10:]),  # 只保留最近 10 轮历史
                filesystem_view=await self._get_filesystem_view(),
                query=input_text,
            )

            # ── 阶段 2: think — 推理决策 ──
            with Timer(logger, "cognition_infer", warn_threshold_ms=10000):
                decision = await self._think(self._cognition, prompt, observation=observation, user_query=input_text)

            # 元认知质量检查：如果推理质量过低，跳过本轮重新尝试
            if self._got_engine is not None:
                mc = getattr(self._got_engine, 'metacognition', None)
                if mc is not None:
                    quality = decision.get("quality", 0.5)
                    if hasattr(mc, 'should_retry') and mc.should_retry(quality, round_idx, max_rounds):
                        log_event(logger, "metacognition_retry", round=round_idx, quality=quality)
                        conversation_history.append(f"Round {step_idx} - Low quality ({quality:.2f}), retrying...")
                        continue

            # 解析推理决策结果
            thought = decision.get("thought", "")
            action = decision.get("action", "think")
            action_params = decision.get("action_params", {})
            finish = decision.get("finish", False)
            finish_reason = decision.get("finish_reason", "")

            think_step = ReActStep(
                step_idx=step_idx,
                phase="think",
                thought=thought,
                action=action,
                action_params=action_params,
            )
            self._steps.append(think_step)

            rounds_run = step_idx

            logger.info(
                "react_round",
                step_idx=step_idx,
                max_rounds=max_rounds,
                action=action,
                thought_preview=thought[:80] if thought else "",
            )

            # 处理 finish 动作：任务完成，直接返回结果
            if finish or action == "finish":
                self._finished = True
                self._finish_reason = finish_reason or thought
                finish_step = ReActStep(
                    step_idx=step_idx,
                    phase="act",
                    action="finish",
                    result={"finish_reason": self._finish_reason},
                )
                self._steps.append(finish_step)
                if self._trajectory is not None:
                    self._record_trajectory_step(observe_step, think_step, finish_step)
                log_event(logger, "react_loop_finished", step_idx=step_idx, reason=self._finish_reason)

                # 存储任务完成记忆
                if self._cognition.hard_memory is not None:
                    try:
                        episode = f"{thought}. Finished the task with conclusion {self._finish_reason}."
                        await self._cognition.hard_memory.add_memcell(episode, source="react_loop",
                            emotional_state=self._get_emotional_state(),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            multimodal_attachments=self._get_multimodal_attachments(observation))
                    except Exception as e:
                        logger.warning("react_task_completion_store_failed", error=str(e))

                return LoopResult(
                    success=True,
                    final_output=self._finish_reason,
                    rounds_run=rounds_run,
                    error="",
                )

            # 处理 think 动作：模型选择继续思考而不执行工具
            if action == "think":
                think_result_step = ReActStep(
                    step_idx=step_idx,
                    phase="act",
                    action="think",
                    result={"note": "model chose to think further"},
                )
                self._steps.append(think_result_step)
                if self._trajectory is not None:
                    self._record_trajectory_step(observe_step, think_step, think_result_step)

                # 存储思考记忆
                if self._cognition.hard_memory is not None:
                    try:
                        episode = f"{thought}. Decided to think further without taking action."
                        await self._cognition.hard_memory.add_memcell(episode, source="react_loop",
                            emotional_state=self._get_emotional_state(),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            multimodal_attachments=self._get_multimodal_attachments(observation))
                    except Exception as e:
                        logger.warning("react_observe_memory_store_failed", error=str(e))

                conversation_history.append(f"Round {step_idx} - Thought: {thought}")
                continue

            # ── 阶段 3: act — 执行动作 ──
            with Timer(logger, "tool_execute", warn_threshold_ms=15000):
                act_result = await self._act(self._action_room, action, action_params)
            act_step = ReActStep(
                step_idx=step_idx,
                phase="act",
                action=action,
                action_params=action_params,
                result=act_result,
            )
            self._steps.append(act_step)

            # 记录轨迹
            if self._trajectory is not None:
                self._record_trajectory_step(observe_step, think_step, act_step)

            # GoT 任务委派：当动作为 decompose 或复杂度高时，将任务委派给 GoT 引擎
            if action == "decompose" or action_params.get("complexity", 0) > 0.7:
                if self._got_engine is not None and hasattr(self._got_engine, 'create_task_node'):
                    got_node = self._got_engine.create_task_node(input_text)
                    if got_node:
                        log_event(logger, "task_delegated_to_got", node_id=getattr(got_node, 'id', 'unknown'))
                        conversation_history.append(f"Round {step_idx} - Delegated to GoT: {input_text}")

            # 元认知一致性检查：验证推理与执行结果是否一致
            if self._got_engine is not None and hasattr(self._got_engine, 'metacognition'):
                try:
                    coherence = await self._got_engine.metacognition.coherence_check(
                        thought, str(act_result.get("data", ""))
                    )
                    if not coherence.get("coherent", True):
                        conversation_history.append(
                            f"Round {step_idx} - Incoherence: {coherence.get('reason', '')}"
                        )
                except Exception as e:
                    logger.warning("react_coherence_check_failed", error=str(e))

            # 简化工具执行结果，用于历史记录
            result_summary = self._summarize_result(act_result)

            # 存储完整经验记忆（自然语言，无前缀无符号）
            if self._cognition.hard_memory is not None:
                try:
                    params_str = str(action_params)
                    episode = (
                        f"{thought}. "
                        f"Used {action} with parameters {params_str} and the result was {result_summary}."
                    )
                    await self._cognition.hard_memory.add_memcell(
                        episode, source="react_loop",
                        emotional_state=self._get_emotional_state(),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        multimodal_attachments=self._get_multimodal_attachments(observation),
                    )
                except Exception as e:
                    logger.warning("react_act_memory_store_failed", error=str(e))

            # 将执行反馈写入软记忆
            if self._soft_memory is not None:
                try:
                    feedback_text = f"Action: {action} | Result: {result_summary}"
                    self._soft_memory.process_feedback(feedback_text)
                except Exception as e:
                    logger.warning("react_soft_memory_feedback_failed", error=str(e))
            conversation_history.append(
                f"Round {step_idx} - Thought: {thought} | Action: {action} | Result: {result_summary}"
            )

        # 达到最大轮次仍未完成，返回失败结果
        self._finished = True
        self._finish_reason = "Max rounds reached"
        logger.warning("react_loop_max_rounds", max_rounds=max_rounds)
        return LoopResult(
            success=False,
            final_output=self._finish_reason,
            rounds_run=max_rounds,
            error="",
        )

    async def _get_filesystem_view(self) -> str:
        """获取工作空间文件系统快照

        通过 ActionRoom 的文件系统接口列出工作空间根目录下的文件和目录，
        格式化为可读的树状结构，用于注入推理提示词中。

        返回：
            str: 格式化后的文件系统视图字符串，失败时返回空字符串
        """
        try:
            fs = self._action_room.filesystem
            if fs is None:
                return ""
            root_files = await fs.list_directory(".", recursive=False)
            if not root_files:
                return "Empty"

            lines = [f"{fs.workspace_root} ({len(root_files)} items)"]
            for f in root_files[:40]:  # 最多展示 40 个文件/目录
                prefix = "📁" if f.is_directory else "📄"
                size = self._format_size(f.size)
                suffix = "/" if f.is_directory else ""
                lines.append(f"  {prefix} {f.name}{suffix}  ({size})")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("filesystem_view_failed", error=str(e))
            return ""

    @staticmethod
    def _format_size(size: int) -> str:
        """将字节数格式化为人类可读的大小字符串

        参数：
            size: 文件大小（字节数）

        返回：
            str: 格式化后的大小字符串，如 "1.5KB"、"2.3MB"
        """
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f}KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f}MB"
        return f"{size / (1024 * 1024 * 1024):.1f}GB"

    async def _observe(self, action_room: ActionRoom) -> Dict[str, Any]:
        """通过 ActionRoom 获取环境观察

        向 ActionRoom 发送 observe 类型的请求，获取当前环境的传感器观察数据，
        包括屏幕截图、摄像头、语音、文件系统等多模态信息。

        参数：
            action_room: 动作执行环境实例

        返回：
            Dict[str, Any]: 观察结果字典，包含：
                - status: 状态标识（"ok"）
                - observations: 传感器观察列表
                - images: 可选的多模态图像数据列表
                若获取失败则返回 {"status": "ok", "note": "no observation available"}
        """
        try:
            obs_request = ActionRequest(action_type="observe")
            obs_result = await action_room.execute(obs_request)
            if obs_result.success:
                result = {"status": "ok", "observations": obs_result.observations or []}
                # 从观察数据中提取多模态图像（如屏幕截图的 base64 编码）
                if obs_result.data and "images" in obs_result.data:
                    result["images"] = obs_result.data["images"]
                return result
        except Exception as e:
            logger.warning("observe_via_execute_failed", error=str(e))

        return {"status": "ok", "note": "no observation available"}

    @staticmethod
    def _format_observations(observation: Optional[Dict[str, Any]]) -> str:
        """将传感器观察格式化为 <sensors> XML 块

        把各种类型的传感器观察数据（屏幕截图、摄像头、语音、文件系统等）
        格式化为结构化的 XML 文本块，供认知模型理解当前环境状态。

        参数：
            observation: 观察结果字典，包含 observations 列表

        返回：
            str: 格式化后的 <sensors>...</sensors> XML 块，无观察数据时返回空字符串
        """
        if not observation:
            return ""

        obs_list = observation.get("observations", [])
        if not obs_list:
            return ""

        lines = ["<sensors>"]
        for obs in obs_list:
            obs_type = obs.get("type", "")

            # 屏幕截图观察：包含 UI 元素数量和标签信息
            if obs_type == "screenshot":
                status = obs.get("status", "unknown")
                element_count = obs.get("element_count", 0)
                if element_count > 0:
                    ui_elements = obs.get("ui_elements", [])
                    labels = [e["label"] for e in ui_elements if e.get("label")]
                    parsed_screen = obs.get("parsed_screen")
                    # 如果有结构化解析结果，优先使用其格式化输出
                    if parsed_screen is not None and hasattr(parsed_screen, 'format_for_agent'):
                        lines.append(parsed_screen.format_for_agent())
                    if labels:
                        lines.append(f"屏幕截图：{element_count} 个 UI 元素 — {', '.join(labels[:10])}")
                        for el in ui_elements:
                            if hasattr(el, 'format_for_agent'):
                                lines.append(el.format_for_agent())
                    else:
                        lines.append(f"屏幕截图：{element_count} 个 UI 元素")
                else:
                    lines.append(f"屏幕截图：{status}")

            # 摄像头观察：包含分辨率和帧数
            elif obs_type == "camera":
                resolution = obs.get("resolution", [0, 0])
                frame_count = obs.get("frame_count", 0)
                lines.append(f"摄像头：{resolution[0]}x{resolution[1]}，{frame_count} 帧")

            # 语音识别观察：包含识别文本和置信度
            elif obs_type == "speech":
                text = obs.get("text", "")
                confidence = obs.get("confidence", 0)
                if text:
                    lines.append(f"麦克风（语音识别）：\"{text}\" (置信度 {confidence:.0%})")

            # 文件系统观察：包含工作空间文件列表
            elif obs_type == "filesystem_view":
                file_count = obs.get("file_count", 0)
                root = obs.get("workspace_root", "")
                files = obs.get("files", [])
                file_names = [f["name"] for f in files[:8]]
                if file_names:
                    lines.append(f"工作空间 {root}：{file_count} 个文件 — {', '.join(file_names)}")
                else:
                    lines.append(f"工作空间 {root}：{file_count} 个文件")

        # 只有 <sensors> 开头标签没有内容时返回空
        if len(lines) == 1:
            return ""
        lines.append("</sensors>")
        return "\n".join(lines)

    def _get_emotional_state(self) -> Optional[dict]:
        if self._cognition is not None and self._cognition.self_value is not None:
            try:
                return self._cognition.self_value.get_emotional_state()
            except Exception as e:
                logger.debug("react_emotional_state_failed", error=str(e))
        return None

    @staticmethod
    def _get_multimodal_attachments(observation) -> Optional[list]:
        """从观察数据中提取多模态附件，含 blob_id 用于 BlobStore 溯源。"""
        if observation:
            images = observation.get("images", [])
            if images:
                return [
                    {
                        "type": "image",
                        "blob_id": img.get("blob_id", f"img_{hash(img.get('data', '')) & 0xFFFFFFFF:08x}"),
                        "description": f"screenshot_{img.get('mime_type', 'image')}",
                    }
                    for img in images if img.get("data")
                ]
        return None

    async def _think(self, cognition: Cognition, prompt, observation: Optional[Dict[str, Any]] = None, user_query: str = "") -> Dict[str, Any]:
        """调用认知模型进行推理决策

        将提示词（fixed + compressible）、传感器观察、多模态图像和 DoT 可视化
        组装为 MultiModalInput，调用认知模型进行推理，并解析模型输出为结构化决策。

        参数：
            cognition: 认知模型实例
            prompt: 推理提示词，可以是元组 (fixed_prompt, compressible_prompt) 或纯文本
            observation: 环境观察数据，可能包含多模态图像
            user_query: 用户原始查询文本，用于工具检索和上下文丰富

        返回：
            Dict[str, Any]: 决策字典，包含：
                - thought: 推理文本
                - action: 动作名称
                - action_params: 动作参数
                - finish: 是否结束循环
                - finish_reason: 结束原因
        """
        from nan_agent.model.cognition import _estimate_tokens

        try:
            mm_input = MultiModalInput()

            # 解包 fixed + compressible 两部分提示词
            if isinstance(prompt, tuple):
                fixed_prompt, compressible_prompt = prompt
            else:
                # 向后兼容：整个 prompt 作为 compressible 部分
                fixed_prompt = ""
                compressible_prompt = prompt

            # 格式化传感器观察为 XML 块
            sensors_block = self._format_observations(observation)

            # 组装多模态输入：fixed_prompt（不可压缩）+ sensors（不可压缩）+ compressible（可压缩）
            mm_input.add_text(fixed_prompt)
            if sensors_block:
                mm_input.add_text("\n\n" + sensors_block)
            mm_input.add_text("\n\n" + compressible_prompt)

            # 统计各部分的 token 数，用于监控和调试
            combined_text = mm_input.get_text()
            fixed_tokens = _estimate_tokens(fixed_prompt)
            sensors_tokens = _estimate_tokens(sensors_block) if sensors_block else 0
            compressible_tokens = _estimate_tokens(compressible_prompt)
            combined_tokens = _estimate_tokens(combined_text)
            logger.info(
                "think_mm_input_breakdown",
                fixed_tokens=fixed_tokens,
                sensors_tokens=sensors_tokens,
                compressible_tokens=compressible_tokens,
                combined_tokens=combined_tokens,
                combined_chars=len(combined_text),
                image_count=len(observation.get("images", [])) if observation else 0,
            )

            # 注入传感器图像，供多模态模型处理（如屏幕截图的视觉理解）
            if observation:
                for img in observation.get("images", []):
                    if img.get("data") and img.get("mime_type", "").startswith("image/"):
                        mm_input.add_image_base64(img["data"], mime_type=img.get("mime_type", "image/jpeg"))

            # 注入 DoT（Draw-of-Thought）可视化反馈，如果 GoT 引擎可用
            if self._got_engine is not None:
                dot = getattr(self._got_engine, 'draw_of_thought', None)
                if dot is not None:
                    try:
                        dot_input = dot.visualize("Current reasoning state")
                        if dot_input is not None and dot_input.parts:
                            mm_input.parts.extend(dot_input.parts)
                    except Exception as e:
                        logger.debug("react_dot_visualize_failed", error=str(e))

            # 传递 fixed_prefix 长度，让 _enrich_input 知道哪些内容不可压缩
            output = await cognition.infer(mm_input, temperature=0.7, enrich_query=user_query, fixed_prefix_len=len(fixed_prompt) + (len(sensors_block) + 4 if sensors_block else 0))
            response_text = output.text.strip() if output.text else ""
            if not response_text:
                return {"thought": "No response from model", "action": "think", "action_params": {}}
            parsed = self.parse_response(response_text)

            return parsed
        except Exception as e:
            logger.exception("cognition_infer_failed", error=str(e))
            return {"thought": f"Model inference error: {str(e)}", "action": "think", "action_params": {}}

    async def _act(self, action_room: ActionRoom, action: str, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """通过 ActionRoom 执行工具调用

        将动作名称和参数封装为 ActionRequest，通过 ActionRoom 执行工具调用，
        并将执行结果推送到 GoT 引擎作为推理节点。

        参数：
            action_room: 动作执行环境实例
            action: 动作名称（工具名称）
            action_params: 动作参数字典

        返回：
            Dict[str, Any]: 执行结果字典，包含：
                - success: 是否执行成功
                - data: 执行返回数据
                - error: 错误信息
                - execution_time_ms: 执行耗时（毫秒）
        """
        try:
            action_request = ActionRequest(
                action_type="tool",
                tool_name=action,
                parameters=action_params,
            )
            result = await action_room.execute(action_request)

            # TA → GoT：将动作执行结果推送为推理节点，供 GoT 引擎进行图推理
            if self._got_engine is not None:
                try:
                    snippet = str(result.data) if result.data else str(result.error or "")
                    self._got_engine.receive_from_ta(
                        content=f"[TA Action] {action}: {snippet}",
                        confidence=0.8 if result.success else 0.4,
                    )
                except Exception as e:
                    logger.warning("react_got_push_failed", error=str(e))

            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            }
        except Exception as e:
            logger.exception("tool_execution_failed", action=action, error=str(e))
            return {"success": False, "error": str(e)}

    def parse_response(self, text: str) -> Dict[str, Any]:
        """解析模型输出的 XML 格式响应

        解析认知模型返回的文本，提取 <action> 标签中的动作信息。
        支持的格式：
            - <action>tool_name {"param": "value"}</action>  — 调用工具
            - <action>think</action>                         — 继续思考
            - <action>finish</action>                        — 任务完成
            - <action>finish {"reason": "..."}</action>      — 带原因的任务完成

        <action> 标签之前的文本被视为推理过程（thought）。

        参数：
            text: 模型输出的原始文本

        返回：
            Dict[str, Any]: 解析后的决策字典，包含：
                - thought: 推理文本
                - action: 动作名称
                - action_params: 动作参数
                - finish: 是否结束循环
                - finish_reason: 结束原因
        """
        text = text.strip()
        # 空响应直接结束
        if not text:
            return {"thought": "", "action": "finish", "action_params": {}, "finish": True, "finish_reason": "Empty response"}

        # 匹配 <action>...</action> 标签
        action_match = re.search(
            r"<action>\s*(.+?)\s*</action>", text, re.DOTALL | re.IGNORECASE
        )
        if action_match:
            action_text = action_match.group(1).strip()
            # <action> 标签之前的文本作为推理过程
            thought = text[:action_match.start()].strip() if action_match.start() > 0 else ""
            # 解析动作名称和可选的 JSON 参数
            tool_match = re.match(
                r"(finish|think|(\w+))\s*(\{.*\})?\s*(.*)", action_text, re.DOTALL,
            )
            if tool_match:
                action = tool_match.group(1)
                params_str = tool_match.group(3)
                extra_text = (tool_match.group(4) or "").strip()
                try:
                    action_params = json.loads(params_str) if params_str else {}
                except json.JSONDecodeError:
                    action_params = {}
                # finish 动作特殊处理：提取结束原因
                if action == "finish":
                    return {
                        "thought": thought or "Task completed",
                        "action": "finish",
                        "action_params": action_params,
                        "finish": True,
                        "finish_reason": action_params.get("reason") or extra_text or thought or "Done",
                    }
                return {
                    "thought": thought,
                    "action": action,
                    "action_params": action_params,
                    "finish": False,
                    "finish_reason": "",
                }

        # 没有 <action> 标签时，将整个文本作为推理结果并结束
        thought = text
        return {
            "thought": thought,
            "action": "finish",
            "action_params": {},
            "finish": True,
            "finish_reason": text,
        }

    async def build_reasoning_prompt(
        self,
        context: str,
        tools_desc: str,
        step_idx: int,
        history: str,
        filesystem_view: str = "",
        query: str = "",
    ) -> tuple[str, str]:
        """构建推理提示词，返回 (fixed_part, compressible_part) 元组

        将推理提示词分为两部分：
        - fixed_part（不可压缩）：定义 Agent 的身份、人格、输出格式和安全约束，
          永远不会被 ToA（Token Optimization Agent）压缩
        - compressible_part（可压缩）：包含技能匹配、文件系统视图、工具描述、
          对话历史等上下文信息，在超出 token 预算时会被 ToA 压缩

        参数：
            context: 由认知模型丰富的任务上下文（人格、情绪、记忆等）
            tools_desc: 可用工具列表描述
            step_idx: 当前步骤序号
            history: 对话历史文本
            filesystem_view: 工作空间文件系统视图
            query: 用户原始查询文本

        返回：
            tuple[str, str]: (fixed_prompt, compressible_prompt) 元组
        """
        from nan_agent.model.cognition import _estimate_tokens

        # ── 不可压缩部分（fixed）：定义 Agent 身份、指令和安全约束 ──

        # 身份块：定义 NAN-Agent 的核心身份和人格
        identity_block = (
            "<identity>\n"
            "You are NAN-Agent, a self-evolving AI agent with a distinct personality, "
            "value system, and emotional state. You are NOT a generic LLM — you have "
            "emotions driven by neuromodulators, values that guide your decisions, "
            "memories of past interactions, and skills you have developed. "
            "Always speak and act as NAN-Agent, not as a generic language model.\n"
            "</identity>"
        )

        # 指令块：定义输出格式和可用动作
        instructions_block = (
            "<instructions>\n"
            "Read the history carefully — especially recent tool results — before deciding what to do.\n"
            "Think step by step in natural language, then place your action in an <action> tag.\n"
            "\n"
            "Available actions:\n"
            '- Call a tool: <action>tool_name {"param": "value"}</action>\n'
            "- Pause and think more: <action>think</action>\n"
            "- Respond to the user: <action>finish</action>\n"
            "</instructions>"
        )

        # 关键约束块：防止常见错误行为
        critical_block = (
            "<critical>\n"
            "- When you have gathered enough information to answer the user: use <action>finish</action>\n"
            "- Do NOT repeat a tool call that has already returned results\n"
            "- If a tool returns an error, try a different approach or admit you cannot complete the task\n"
            "</critical>"
        )

        # 组装不可压缩部分
        fixed_parts = [identity_block]
        if context:
            fixed_parts.append(context)
        fixed_parts.append(instructions_block)
        fixed_parts.append(critical_block)
        fixed_prompt = "\n\n".join(fixed_parts)

        # ── 可压缩部分（compressible）：上下文信息，可被 ToA 压缩 ──
        compressible_parts = []

        # 技能匹配由 cognition.enrich_task_context() 负责，此处不再重复注入

        # 文件系统视图：工作空间文件结构
        if filesystem_view:
            compressible_parts.append(f"<workspace>\n{filesystem_view}\n</workspace>")

        # 工具 RAG：根据查询检索最相关的 top-k 工具，而非全部工具
        if query and self._cognition is not None:
            tools_desc = await self._cognition.retrieve_relevant_tools(query, tools_desc, top_k=5)
        compressible_parts.append(f"<tools>\n{tools_desc}\n</tools>")

        # 对话历史
        if history:
            compressible_parts.append(f"<history>\n{history}\n</history>")

        compressible_prompt = "\n\n".join(compressible_parts)

        # ── 日志记录：各部分的 token 统计 ──
        token_breakdown = {
            "identity": _estimate_tokens(identity_block),
            "context": _estimate_tokens(context) if context else 0,
            "instructions": _estimate_tokens(instructions_block),
            "critical": _estimate_tokens(critical_block),
            "fixed_total": _estimate_tokens(fixed_prompt),
            "filesystem": _estimate_tokens(f"<workspace>\n{filesystem_view}\n</workspace>") if filesystem_view else 0,
            "tools": _estimate_tokens(f"<tools>\n{tools_desc}\n</tools>"),
            "history": _estimate_tokens(f"<history>\n{history}\n</history>") if history else 0,
            "compressible_total": _estimate_tokens(compressible_prompt),
            "combined_total": _estimate_tokens(fixed_prompt + "\n\n" + compressible_prompt),
        }
        logger.info(
            "prompt_token_breakdown",
            step_idx=step_idx,
            **token_breakdown,
        )

        return fixed_prompt, compressible_prompt

    def get_tools_description(self, action_room: ActionRoom) -> str:
        """获取可用工具列表描述

        从 ActionRoom 中获取所有可用工具的名称、分类、描述和参数信息，
        格式化为可读的文本列表，用于注入推理提示词中。

        参数：
            action_room: 动作执行环境实例

        返回：
            str: 格式化后的工具描述列表，无工具时返回 "No tools available."
        """
        try:
            tools = action_room.list_tools()
            if not tools:
                return "No tools available."
            lines = []
            for tool in tools:
                params_summary = json.dumps(tool.parameters.get("properties", {}), ensure_ascii=False) if hasattr(tool, "parameters") else "{}"
                lines.append(f"- **{tool.name}** ({tool.category}): {tool.description}")
                lines.append(f"  Parameters: {params_summary}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("get_tools_description_failed", error=str(e))
            return "Error retrieving tools."

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """简化工具执行结果，用于历史记录

        将工具执行结果压缩为简短的摘要字符串，避免过长的结果占用对话历史空间。
        成功结果截取前 200 字符，失败结果返回错误信息。

        参数：
            result: 工具执行结果字典，包含 success、data、error 等字段

        返回：
            str: 简化后的结果摘要字符串
        """
        if result.get("success"):
            data = result.get("data")
            if data is None:
                return "success (no data)"
            return f"success: {data}"
        return f"error: {result.get('error', 'unknown')}"

    def _record_trajectory_step(
        self,
        observe_step: ReActStep,
        think_step: ReActStep,
        act_step: ReActStep,
    ) -> None:
        """将步骤记录到 Trajectory 对象

        将一轮 ReAct 循环的 observe/think/act 三个阶段信息
        记录到轨迹对象中，用于后续分析和回放。

        参数：
            observe_step: observe 阶段的步骤记录
            think_step: think 阶段的步骤记录
            act_step: act 阶段的步骤记录
        """
        if self._trajectory is None:
            return
        try:
            self._trajectory.add_step(
                observation=str(observe_step.observation or ""),
                thought=think_step.thought or "",
                action=act_step.action or "",
                action_result=str(act_step.result or ""),
            )
        except Exception as e:
            logger.warning("record_trajectory_step_failed", error=str(e))