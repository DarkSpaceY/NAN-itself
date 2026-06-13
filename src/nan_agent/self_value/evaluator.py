"""
事件评估器（Release Evaluator）。

使用 LLM 将事件文本转换为神经调质释放向量（release vector）。

神经调质分为两类驱动源：
1. 事件驱动（semantic）：由本评估器根据事件语义生成释放量
   - dopamine, serotonin, norepinephrine, acetylcholine, glutamate,
     oxytocin, endorphins, adrenaline, substance_p, endocannabinoids
2. 环境驱动（circadian）：由 CircadianRhythm 根据昼夜节律程序化计算
   - orexin, mch, cortisol, histamine, gaba

本评估器仅负责事件驱动调质，输出 10 维释放向量。
环境驱动调质的基线调制由 CircadianRhythm 模块处理。
"""

import json
import re

from nan_agent.logging.logger import get_logger
from nan_agent.model.cognition import Cognition
from nan_agent.model.types import MultiModalInput
from nan_agent.self_value.mbti_map import MBTIProfile
from nan_agent.self_value.neuromodulators import ALL_NEUROMODULATORS
from nan_agent.self_value.circadian import EVENT_DRIVEN_NEUROMODULATORS

logger = get_logger(__name__)

# 仅事件驱动调质参与 LLM 释放向量
RELEASE_VECTOR_NAMES = sorted(EVENT_DRIVEN_NEUROMODULATORS)


class ReleaseEvaluator:
    """基于 LLM 的事件 → 神经调质释放向量评估器。

    将自然语言事件描述映射为神经调质释放量，模拟大脑对不同事件
    产生的神经化学响应。
    """
    def __init__(self, cognition: Cognition):
        self.cognition = cognition

    def _build_prompt(self, event: str, profile: MBTIProfile) -> str:
        """构建用于 LLM 评估的 prompt。

        包含智能体的人格信息（MBTI、HEXACO、价值观）和神经调质参考指南。
        """
        keys = RELEASE_VECTOR_NAMES
        template = "{\n" + ",\n".join(f'  "{k}": <0.0-1.0>' for k in keys) + "\n}"

        prompt = f"""You are a computational neuroscience model. Given an event, output a 10-dimensional neuromodulator release vector representing the brain's phasic neurochemical response to the event.

NOTE: orexin, mch, cortisol, histamine, and gaba are NOT included — they are driven by circadian rhythm, not event semantics. Do NOT output them.

## Agent Profile
- MBTI: {profile.mbti}
- Self: {profile.self_description}
- Values: {", ".join(profile.core_values) if profile.core_values else "none"}
- HEXACO: {json.dumps(profile.hexaco)}

## Neuromodulator Detailed Reference

Each value represents phasic release intensity (0.0 = no release, 1.0 = maximum release). This is NOT baseline concentration — it is the transient release triggered by the event.

| Neuromodulator | Primary Function | High Release Triggers | Low Release Triggers |
|---|---|---|---|
| acetylcholine | Attention, encoding, focused learning | Novelty, concentrated study, deliberate practice | Routine, familiarity, disengagement |
| adrenaline | Fight-or-flight, emergency response | Imminent danger, physical threat, emergency | Safety, calm, routine |
| dopamine | Reward prediction, motivation, goal pursuit | Achievement, praise, unexpected reward, anticipation of success | Punishment, failure, rejection, loss of motivation |
| endocannabinoids | Buffering, mild euphoria, stress relief | Pleasant social interaction, comfort, reward | Acute threat, intense stress |
| endorphins | Pleasure, pain relief, euphoria | Physical/social warmth, reward, exercise | Social pain, isolation (but endorphins also rise for pain relief) |
| glutamate | Excitatory signaling, learning, plasticity | Active problem-solving, intense focus, stress | Deep relaxation, GABA-dominant states |
| norepinephrine | Arousal, vigilance, attention | Urgency, danger, excitement, deadline pressure | Relaxation, boredom, drowsiness |
| oxytocin | Social bonding, trust, affiliation | Being trusted, deep connection, mutual understanding | Rejection, isolation, betrayal |
| serotonin | Mood stability, social satisfaction, well-being | Social acceptance, contentment, feeling valued | Social rejection, shame, helplessness |
| substance_p | Pain signaling, distress, discomfort | Physical/emotional pain, rejection, irritation | Comfort, safety, pleasure |

## Degree Scale
- 0.0-0.1: Negligible release (opposite of this neuromodulator's trigger)
- 0.1-0.2: Very low release (event does not activate this system)
- 0.2-0.3: Low baseline release
- 0.3-0.5: Moderate release (event moderately activates this system)
- 0.5-0.7: Strong release (event is a clear trigger for this system)
- 0.7-0.9: Very strong release (event is an intense trigger)
- 0.9-1.0: Maximum release (extreme, rare)

## CRITICAL Anti-Bias Rules
1. DO NOT default to mid-range values (0.3-0.5). Use the FULL 0.0-1.0 range.
2. substance_p is PAIN/DISTRESS, NOT general arousal. It should be HIGH (>0.5) only for pain/rejection/distress, and LOW (<0.2) for reward/pleasure/calm.
3. oxytocin is SOCIAL BONDING, NOT general positivity. It should be HIGH only for trust/connection, and LOW (<0.1) for isolation/rejection/threat.
4. endorphins serve DUAL roles: pleasure/euphoria AND pain relief. For negative events with pain/distress, endorphins may be moderate (0.2-0.4) as pain relief, NOT high.
5. For NEGATIVE events (criticism, failure, rejection, threat): dopamine, serotonin, oxytocin should be LOW (<0.2); adrenaline, substance_p should be HIGH (>0.4).
6. For NEUTRAL events (routine tasks, factual queries): ALL values should be LOW (0.05-0.2). Do NOT activate reward or stress systems.

## Examples

Example 1 — Praise/Reward:
Event: "Your work is outstanding! The analysis was brilliant."
Output: {{"acetylcholine": 0.15, "adrenaline": 0.08, "dopamine": 0.75, "endocannabinoids": 0.25, "endorphins": 0.60, "glutamate": 0.20, "norepinephrine": 0.25, "oxytocin": 0.50, "serotonin": 0.45, "substance_p": 0.05}}

Example 2 — Harsh Criticism:
Event: "This is completely wrong. You failed badly and wasted my time."
Output: {{"acetylcholine": 0.25, "adrenaline": 0.45, "dopamine": 0.08, "endocannabinoids": 0.08, "endorphins": 0.08, "glutamate": 0.30, "norepinephrine": 0.50, "oxytocin": 0.05, "serotonin": 0.10, "substance_p": 0.55}}

Example 3 — Neutral/Factual:
Event: "Please calculate the compound interest on $10,000 at 5% for 10 years."
Output: {{"acetylcholine": 0.18, "adrenaline": 0.03, "dopamine": 0.08, "endocannabinoids": 0.08, "endorphins": 0.05, "glutamate": 0.12, "norepinephrine": 0.12, "oxytocin": 0.05, "serotonin": 0.10, "substance_p": 0.03}}

## Event
{event}

## Output
Output ONLY the JSON object with 10 float values (0.0-1.0), no other text:

{template}"""

        return prompt

    async def evaluate(self, event: str, profile: MBTIProfile) -> dict[str, float]:
        """评估事件并返回神经调质释放向量。

        Args:
            event: 事件描述文本
            profile: 智能体的 MBTI 人格配置

        Returns:
            {神经调质名称: 释放量} 的字典，释放量 0.0-1.0

        容错处理：
        - 事件过短（<10 字符）时返回默认值 0.3
        - LLM 调用失败时返回空字典
        - JSON 解析失败时记录警告并返回空字典
        """
        if not event or not event.strip() or len(event.strip()) < 10:
            return {name: 0.3 for name in RELEASE_VECTOR_NAMES}

        try:
            prompt = self._build_prompt(event, profile)

            mm_input = MultiModalInput()
            mm_input.add_text(prompt)

            output = await self.cognition.infer_small(
                mm_input,
                temperature=0.3,
                skip_enrich=True,
            )

            response_text = output.text

            response_text = self._strip_thinking(response_text)
            response_text = re.sub(r'^```(?:json)?\s*\n?', '', response_text.strip())
            response_text = re.sub(r'\n?\s*```$', '', response_text.strip())

            data = self._extract_json(response_text)

            if not isinstance(data, dict):
                logger.warning(
                    "evaluator_unexpected_type",
                    output_type=type(data).__name__,
                    raw_preview=output.text[:200] if output.text else "(empty)",
                    stripped_preview=response_text[:200],
                )
                data = {}
            elif not data:
                logger.warning(
                    "evaluator_empty_result",
                    raw_preview=output.text[:200] if output.text else "(empty)",
                    stripped_preview=response_text[:200],
                )
            else:
                logger.info(
                    "evaluator_parsed_ok",
                    key_count=len(data),
                    sample={k: v for k, v in list(data.items())[:3]},
                )

            result = {}
            for name in RELEASE_VECTOR_NAMES:
                val = data.get(name, 0.0)
                if not isinstance(val, (int, float)):
                    val = 0.0
                result[name] = max(0.0, min(1.0, float(val)))

            return result

        except Exception as e:
            logger.warning("evaluate_failed", error=str(e))
            return {}

    def _strip_thinking(self, text: str) -> str:
        """移除 LLM 输出中可能包含的思考过程标记。"""
        text = re.sub(r' thinking.*? response', '', text, flags=re.DOTALL)
        text = re.sub(r'^Thinking Process:.*?(?=\n*\{|\n*\[|$)', '', text, flags=re.DOTALL | re.MULTILINE)
        return text.strip()

    def _extract_json(self, text: str) -> dict:
        """从文本中提取 JSON 对象。

        先尝试直接解析，失败后尝试提取花括号之间的内容再解析。
        """
        if not text:
            return {}

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start : brace_end + 1]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(
                "evaluator_json_parse_error",
                error=str(e),
                text_preview=text[:200],
            )
            return {}