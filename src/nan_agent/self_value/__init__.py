"""
self_value 模块 — 智能体的自我意识与人格系统。

本模块是 NAN-Agent 的"自我价值"核心，负责建模 AI 智能体的：
- 人格特质（HEXACO 六因素模型 + MBTI 类型映射）
- 神经递质动力学（模拟类脑神经调质浓度变化）
- 价值观体系（可演化的价值观库）
- 自我叙事（反思与成长记录）
- 情绪状态（效价-唤醒度二维模型 + 自然语言描述）
- 认知失调检测与自我反思

主要入口类：SelfValue（interface.py）
"""

from nan_agent.self_value.neuromodulators import (
    ALL_NEUROMODULATORS,
    CATEGORY_AMINO,
    CATEGORY_MONOAMINE,
    CATEGORY_NEUROPEPTIDE,
    CATEGORY_OTHER,
    Neuromodulator,
    NeuromodulatorState,
)

from nan_agent.self_value.mbti_map import (
    ALL_MBTI_TYPES,
    MBTIProfile,
    MBTIMapper,
)

from nan_agent.self_value.evaluator import (
    RELEASE_VECTOR_NAMES,
    ReleaseEvaluator,
)

from nan_agent.self_value.hexaco import (
    HEXACO,
    TRAIT_NAMES,
)

from nan_agent.self_value.values import (
    ValueItem,
    ValueLibrary,
)

__all__ = [
    "CATEGORY_MONOAMINE",
    "CATEGORY_AMINO",
    "CATEGORY_NEUROPEPTIDE",
    "CATEGORY_OTHER",
    "Neuromodulator",
    "NeuromodulatorState",
    "ALL_NEUROMODULATORS",
    "MBTIProfile",
    "MBTIMapper",
    "ALL_MBTI_TYPES",
    "RELEASE_VECTOR_NAMES",
    "ReleaseEvaluator",
    "HEXACO",
    "TRAIT_NAMES",
    "ValueItem",
    "ValueLibrary",
]