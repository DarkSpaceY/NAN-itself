"""
SelfValue — 智能体自我意识主门面类。

self_value 模块的核心入口，整合了智能体的：
- 人格系统（MBTI + HEXACO）
- 神经调质动力学
- 价值观体系
- 自我叙事
- 认知失调检测
- 自我反思与学习

生命周期：
1. initialize(mbti)     → 设定 MBTI 类型，初始化所有子系统
2. process_event(event)  → 处理外部事件，更新神经调质状态
3. check_dissonance()    → 检测行为与价值观的冲突
4. generate_reflection() → 生成反思内容
5. learn_from_dissonance() → 通过反思进行对比学习
6. refine_values()       → 周期性价值观精炼

对外接口：
- get_personality_context() → 获取人格上下文（注入 LLM prompt）
- get_valence_arousal()     → 获取效价-唤醒度情绪坐标
- get_emotional_state()     → 获取完整情绪状态
- solidify_personality()    → 导出可持久化的人格快照
"""

import json
import re
from typing import Optional

from nan_agent.logging.logger import get_logger
from nan_agent.model.cognition import Cognition
from nan_agent.model.types import MultiModalInput
from nan_agent.self_value.coupling import CouplingNetwork
from nan_agent.self_value.dynamics import NeuromodulatorDynamics
from nan_agent.self_value.evaluator import ReleaseEvaluator
from nan_agent.self_value.hexaco import HEXACO
from nan_agent.self_value.mbti_map import MBTIMapper, MBTIProfile
from nan_agent.self_value.narrative import ReflectionLevel, SelfNarrative
from nan_agent.self_value.neuromodulators import ALL_NEUROMODULATORS
from nan_agent.self_value.values import ValueItem, ValueLibrary

logger = get_logger(__name__)


class SelfValue:
    """智能体自我意识主门面。

    整合人格、神经调质、价值观和叙事系统，提供统一的自我意识接口。
    需传入 Cognition 实例用于 LLM 推理，可选的 hard_memory/soft_memory
    用于支持的反思学习功能。
    """
    def __init__(
        self,
        cognition: Cognition,
        hard_memory=None,
        soft_memory=None,
    ):
        self.cognition = cognition
        self.hard_memory = hard_memory
        self.soft_memory = soft_memory

        self._mapper = MBTIMapper()
        self._coupling = CouplingNetwork()
        self._dynamics = NeuromodulatorDynamics(ALL_NEUROMODULATORS, self._coupling)
        self._evaluator = ReleaseEvaluator(cognition)

        self.profile: Optional[MBTIProfile] = None
        self.hexaco = HEXACO()
        self.values = ValueLibrary()
        self.narrative = SelfNarrative()

    @property
    def dynamics(self):
        return self._dynamics

    def initialize(self, mbti: str = "INTJ") -> None:
        """初始化智能体人格，设定 MBTI 类型并构建所有子系统。

        执行步骤：
        1. 通过 MBTIMapper 将 MBTI 类型映射为 MBTIProfile
        2. 初始化 HEXACO 人格特质
        3. 配置神经调质基线（baseline、sensitivity、decay_rate）
        4. 设置自我描述和核心价值观
        5. 将核心价值观注册到 ValueLibrary

        Args:
            mbti: MBTI 类型码，如 "INTJ", "ENFP"。默认 "INTJ"。
        """
        self.profile = self._mapper.map(mbti)

        self.hexaco = HEXACO(**self.profile.hexaco)

        for nm_name, nm_config in self.profile.neuromodulator_baselines.items():
            state = self._dynamics._states.get(nm_name)
            if state is not None:
                state.baseline = nm_config.get("baseline", state.baseline)
                state.sensitivity = nm_config.get("sensitivity", state.sensitivity)
                state.decay_rate = nm_config.get("decay_rate", state.decay_rate)
                state.concentration = state.baseline

        self.narrative.self_description = self.profile.self_description
        self.narrative.core_values = list(self.profile.core_values)
        self.narrative.long_term_goals = list(self.profile.long_term_goals)

        for value_name in self.profile.core_values:
            self.values.add(ValueItem(name=value_name, direction="positive", weight=0.7))

        logger.info(
            "self_value_initialized",
            mbti=self.profile.mbti,
            hexaco=self.hexaco.to_dict(),
            core_values=self.profile.core_values,
        )

    def fine_tune_hexaco(self, **kwargs) -> None:
        """微调 HEXACO 人格特质值。

        Args:
            **kwargs: 键为特质名（如 honesty_humility），值为 0.0-1.0 的新值
        """
        for key, value in kwargs.items():
            if hasattr(self.hexaco, key):
                setattr(self.hexaco, key, max(0.0, min(1.0, value)))
                if self.profile is not None:
                    self.profile.fine_tune_hexaco(**{key: value})

        logger.info("hexaco_fine_tuned", **kwargs)

    def fine_tune_neuromodulator(self, name: str, **kwargs) -> None:
        """微调指定神经调质的参数。

        Args:
            name: 神经调质名称（如 "dopamine"）
            **kwargs: 键为参数名（baseline/sensitivity/decay_rate），值为新值
        """
        state = self._dynamics._states.get(name)
        if state is None:
            logger.warning("fine_tune_neuromodulator_not_found", name=name)
            return

        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, max(0.0, min(1.0, value)))
                state.concentration = max(0.0, min(1.0, state.concentration))

        if self.profile is not None:
            self.profile.fine_tune_neuromodulator(name, **kwargs)

        logger.info("neuromodulator_fine_tuned", name=name, **kwargs)

    def get_personality_context(self) -> dict:
        """获取完整的人格上下文，用于注入 LLM prompt。

        Returns:
            包含神经调质浓度、效价-唤醒度、HEXACO 特质、
            自我描述和核心价值观的字典
        """
        context = self._dynamics.get_personality_context()
        context["hexaco"] = self.hexaco.to_dict()
        context["self_description"] = self.narrative.self_description
        context["core_values"] = self.narrative.core_values
        return context

    def get_valence_arousal(self) -> dict[str, float]:
        """获取当前效价-唤醒度情绪坐标。

        Returns:
            {"valence": 0.0-1.0, "arousal": 0.0-1.0}
        """
        return self._dynamics.compute_valence_arousal()

    def get_emotional_state(self) -> dict:
        """获取完整情绪状态，包含效价、唤醒度和各神经调质浓度。

        Returns:
            {"valence": float, "arousal": float, "concentrations": {名称: 浓度}}
        """
        va = self._dynamics.compute_valence_arousal()
        concentrations = {
            name: state.concentration
            for name, state in self._dynamics.get_states().items()
        }
        return {"valence": va.get("valence", 0.5), "arousal": va.get("arousal", 0.5), "concentrations": concentrations}

    async def process_event(self, event: str) -> dict:
        """处理外部事件，更新神经调质状态并返回新的情绪坐标。

        完整流程：
        1. 确保已初始化（未初始化则使用默认 INTJ）
        2. 通过 ReleaseEvaluator 将事件转换为释放向量
        3. 应用释放向量到神经调质动力学引擎
        4. 执行一个时间步的衰减
        5. 计算并返回新的效价-唤醒度

        Args:
            event: 事件描述文本

        Returns:
            {"valence": float, "arousal": float}
        """
        if self.profile is None:
            self.initialize()

        release = await self._evaluator.evaluate(event, self.profile)

        if release:
            self._dynamics.apply_release_vector(release)

        self._dynamics.step(dt=1.0)

        va = self._dynamics.compute_valence_arousal()

        logger.info(
            "event_processed",
            valence=round(va["valence"], 4),
            arousal=round(va["arousal"], 4),
        )

        return va

    async def check_dissonance(self, behavior: str, relevant_values: list[str]) -> dict:
        """检测行为与价值观的认知失调。

        使用 LLM 评估行为是否与智能体的价值观存在冲突：
        - 正面价值观（direction='positive'）：如果行为违背该价值观，标记冲突
        - 负面价值观（direction='negative'）：如果行为符合该价值观，标记冲突

        当整体失调度 > 0.3 时，自动记录事件触发反思。
        overall_dissonance 由代码端计算（所有 severity 的加权平均），
        而非依赖 LLM 返回值，确保多冲突累积效应被正确反映。

        Args:
            behavior: 行为描述文本
            relevant_values: 需要检查的相关价值观名称列表

        Returns:
            {"overall_dissonance": float, "assessments": [{value_name, conflict, severity, reason}]}
        """
        if not relevant_values:
            return {"overall_dissonance": 0.0, "assessments": []}

        all_values = self.values.extract_metadata_for_prompt()
        relevant_meta = [v for v in all_values if v["name"] in relevant_values]

        if not relevant_meta:
            return {"overall_dissonance": 0.0, "assessments": []}

        prompt_text = (
            "You are a value consistency evaluator. Analyze whether the behavior "
            "contradicts the agent's values.\n\n"
            "Rules:\n"
            "- For positive values (direction='positive'), if the behavior violates "
            "the value, mark conflict=true\n"
            "- For negative values (direction='negative'), if the behavior aligns "
            "with the value (e.g., lying matches 'deception'), mark conflict=true\n"
            "- severity: continuous scale 0.0-1.0 indicating conflict intensity:\n"
            "  0.0 = no conflict, 0.2 = minor tension, 0.4 = noticeable conflict, "
            "0.6 = clear violation, 0.8 = strong contradiction, 1.0 = severe violation\n"
            "  Do NOT use only 0 or 1 — use intermediate values for partial conflicts.\n"
            "- reason: brief explanation\n\n"
            f"Behavior: {behavior}\n\n"
            f"Values: {json.dumps(relevant_meta, ensure_ascii=False)}\n\n"
            "Return ONLY a JSON object with keys 'assessments' (array of "
            "{value_name, conflict, severity, reason}) and 'overall_dissonance' "
            "(weighted average of all severity values, not just the max). "
            "If no conflicts exist, overall_dissonance should be 0.0."
        )

        user_input = MultiModalInput()
        user_input.add_text(prompt_text)

        try:
            output = await self.cognition.infer_small(
                user_input, temperature=0.3,
            )
            response_text = output.text.strip() if output and output.text else "{}"
        except Exception as e:
            logger.warning("check_dissonance_infer_failed", error=str(e))
            return {"overall_dissonance": 0.0, "assessments": []}

        try:
            response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
            response_text = re.sub(r'\n?\s*```$', '', response_text)
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("check_dissonance_parse_failed", raw=response_text[:200])
            return {"overall_dissonance": 0.0, "assessments": []}

        assessments = data.get("assessments", [])

        # 代码端计算 overall_dissonance：所有 severity 的加权平均
        # 这比依赖 LLM 返回的 max 更准确，能反映多冲突的累积效应
        if assessments:
            severities = []
            for a in assessments:
                sev = a.get("severity", 0.0)
                if isinstance(sev, (int, float)):
                    severities.append(max(0.0, min(1.0, float(sev))))
                else:
                    severities.append(0.0)
            overall = sum(severities) / len(severities) if severities else 0.0
        else:
            overall = 0.0

        if overall > 0.3:
            conflict_names = [
                a["value_name"] for a in assessments if a.get("conflict")
            ]
            content = (
                f"Value conflict detected for behavior '{behavior}': "
                f"{', '.join(conflict_names)}"
            )
            self.narrative.record_reflection(
                ReflectionLevel.EVENT_TRIGGERED, content,
            )

        return {"overall_dissonance": overall, "assessments": assessments}

    async def generate_reflection(self, behavior: str, dissonance: dict) -> dict:
        """根据认知失调结果生成自我反思。

        使用 LLM 生成包含改进建议的反思内容，自动记录深度反思到叙事系统。

        Args:
            behavior: 原始行为描述
            dissonance: check_dissonance() 的返回结果

        Returns:
            {"reflection_text": str, "conflict_value": str,
             "chosen_behavior": str, "reject_behavior": str,
             "severity": float, "learning_topic": str}
        """
        conflicts = [
            a for a in dissonance.get("assessments", [])
            if a.get("conflict")
        ]
        if not conflicts:
            return {
                "reflection_text": "",
                "chosen_behavior": behavior,
                "reject_behavior": "",
                "learning_topic": "",
            }

        conflict_summary = json.dumps(conflicts, ensure_ascii=False, indent=2)

        prompt_text = (
            "You are a self-reflective agent. You have detected a conflict between "
            "your behavior and your values. Generate a reflection.\n\n"
            f"Behavior: {behavior}\n\n"
            f"Conflicts: {conflict_summary}\n\n"
            "Return a JSON object with keys:\n"
            "- reflection_text: your reflection on the conflict\n"
            "- conflict_value: the primary value involved\n"
            "- chosen_behavior: what you should have done instead\n"
            "- reject_behavior: the original problematic behavior\n"
            "- severity: how severe the conflict is (0-1)\n"
            "- learning_topic: a short label for this lesson\n\n"
            "The chosen_behavior should be specific and actionable, something "
            "an AI agent could learn to do through contrastive training."
        )

        user_input = MultiModalInput()
        user_input.add_text(prompt_text)

        try:
            output = await self.cognition.infer_small(
                user_input, temperature=0.5, max_tokens=512,
            )
            response_text = output.text.strip() if output and output.text else "{}"
        except Exception as e:
            logger.warning("generate_reflection_infer_failed", error=str(e))
            return {
                "reflection_text": "",
                "chosen_behavior": "",
                "reject_behavior": behavior,
                "learning_topic": "",
            }

        try:
            response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
            response_text = re.sub(r'\n?\s*```$', '', response_text)
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("generate_reflection_parse_failed", raw=response_text[:200])
            return {
                "reflection_text": "",
                "chosen_behavior": "",
                "reject_behavior": behavior,
                "learning_topic": "",
            }

        reflection_text = data.get("reflection_text", "")
        if reflection_text:
            self.narrative.record_reflection(
                ReflectionLevel.DEEP,
                reflection_text,
            )

        return {
            "reflection_text": reflection_text,
            "conflict_value": data.get("conflict_value", ""),
            "chosen_behavior": data.get("chosen_behavior", ""),
            "reject_behavior": data.get("reject_behavior", behavior),
            "severity": data.get("severity", 0.0),
            "learning_topic": data.get("learning_topic", ""),
        }

    async def learn_from_dissonance(self, behavior: str, dissonance: dict) -> bool:
        """从认知失调中学习，通过对比学习优化行为。

        流程：
        1. 生成反思（chosen_behavior vs reject_behavior）
        2. 如果 soft_memory 可用，通过 incremental_train 进行对比学习

        Args:
            behavior: 原始行为
            dissonance: 认知失调评估结果

        Returns:
            是否成功进行了学习
        """
        reflection = await self.generate_reflection(behavior, dissonance)

        if not reflection.get("chosen_behavior") or not reflection.get("reject_behavior"):
            return False

        if self.soft_memory is None:
            logger.info("learn_from_dissonance_skip_no_soft_memory")
            return False

        training_pair = [{
            "prompt": reflection.get("learning_topic", "Behavior correction"),
            "chosen": reflection["chosen_behavior"],
            "reject": reflection["reject_behavior"],
        }]

        try:
            result = await self.soft_memory.incremental_train(
                training_pair,
                label=f"reflection_{reflection.get('conflict_value', '')}",
            )
            logger.info(
                "learn_from_dissonance_trained",
                topic=reflection.get("learning_topic"),
            )
            return result is not None
        except Exception as e:
            logger.warning("learn_from_dissonance_train_failed", error=str(e))
            return False

    def should_refine_values(self, min_count: int = 10, min_hours: float = 24.0) -> bool:
        """判断是否应该触发价值观精炼。

        触发条件（满足任一）：
        - 未精炼的反思数量 >= min_count
        - 距上次精炼 >= min_hours 小时

        Args:
            min_count: 最小未精炼反思数量阈值
            min_hours: 最小精炼间隔（小时）

        Returns:
            是否应该触发精炼
        """
        unrefined = self.narrative.unrefined_reflection_count()
        if unrefined >= min_count:
            return True
        hours = self.narrative.hours_since_last_refinement()
        if hours == float("inf"):
            return False
        return hours >= min_hours

    async def refine_values(self) -> dict:
        """执行价值观精炼。

        使用 LLM 分析最近的反思记录，自动调整价值观体系：
        - 更新自我描述
        - 调整现有价值观权重
        - 添加新价值观
        - 废弃不再适用的价值观

        Returns:
            精炼结果字典，包含调整/新增/废弃的价值观列表
        """
        recent_reflections = [
            h for h in self.narrative.history
            if h.get("type") == "reflection"
        ][-20:]

        values_meta = self.values.extract_metadata_for_prompt()
        current_desc = self.narrative.self_description

        prompt_text = (
            "You are a self-aware agent reflecting on your values. "
            "Based on recent experiences and reflections, decide what kind "
            "of agent you want to become going forward.\n\n"
            "This is NOT about summarizing what happened — it's about actively "
            "choosing who you want to be.\n\n"
            f"Current self-narrative: {current_desc}\n\n"
            f"Current values: {json.dumps(values_meta, ensure_ascii=False, indent=2)}\n\n"
            f"Recent reflections: {json.dumps(recent_reflections, ensure_ascii=False, indent=2)}\n\n"
            "CRITICAL: self_description_update must be a coherent first-person narrative "
            "(200-500 words) that REWRITES and INTEGRATES your existing self-narrative "
            "with new insights from recent reflections. Do NOT simply append events — "
            "weave them into your evolving sense of identity. Your narrative should read "
            "as a unified self-definition, not a chronological log.\n\n"
            "Return a JSON object with keys:\n"
            "- self_description_update: rewritten self-narrative integrating old and new, "
            "or 'unchanged' if no meaningful evolution occurred\n"
            "- value_adjustments: array of {name, action, new_weight, reason}\n"
            "  action: 'strengthen', 'weaken', or 'maintain'\n"
            "- new_values: array of {name, direction, weight, description}\n"
            "- deprecated_values: array of value names to deprecate\n"
            "- reflection_summary: a brief summary of this refinement cycle\n"
        )

        user_input = MultiModalInput()
        user_input.add_text(prompt_text)

        try:
            output = await self.cognition.infer_small(
                user_input, temperature=0.5,
            )
            response_text = output.text.strip() if output and output.text else "{}"
        except Exception as e:
            logger.warning("refine_values_infer_failed", error=str(e))
            return {"error": str(e)}

        try:
            response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text)
            response_text = re.sub(r'\n?\s*```$', '', response_text)
            data = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("refine_values_parse_failed", raw=response_text[:200])
            return {"error": "json_parse_failed"}

        result = {
            "self_description_updated": False,
            "values_adjusted": [],
            "values_added": [],
            "values_deprecated": [],
            "reflection_summary": data.get("reflection_summary", ""),
        }

        new_desc = data.get("self_description_update", "")
        if new_desc and new_desc != "unchanged" and new_desc != current_desc:
            self.narrative.update_self_description(new_desc)
            result["self_description_updated"] = True

        changes = self.values.apply_refinement(data)
        result["values_adjusted"] = changes["adjusted"]
        result["values_added"] = changes["added"]
        result["values_deprecated"] = changes["deprecated"]

        self.narrative.mark_refinement_complete()

        self.narrative.record_reflection(
            ReflectionLevel.DEEP,
            f"Value refinement: {data.get('reflection_summary', 'periodic refinement')}",
        )

        logger.info("refine_values_complete", **result)
        return result

    def solidify_personality(self) -> dict:
        """导出可持久化的人格快照。

        Returns:
            包含 HEXACO、价值观列表和自我描述的字典，
            可用于序列化保存和恢复。
        """
        values_list = [
            {"name": v.name, "direction": v.direction, "weight": v.weight}
            for v in self.values.list_all()
        ]

        return {
            "hexaco": self.hexaco.to_dict(),
            "values": values_list,
            "self_description": self.narrative.self_description,
        }