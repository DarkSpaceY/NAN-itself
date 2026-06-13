"""
情绪状态自然语言描述器（Emotional State Descriptor）。

将神经调质浓度和效价-唤醒度（Valence-Arousal）坐标转换为
第一人称中文自然语言描述，用于注入 LLM prompt 或展示给用户。

功能：
1. describe_emotional_state()：根据效价-唤醒度象限和 top-N 活跃神经调质生成描述
2. values_priority()：将核心价值观列表格式化为优先级展示字符串
3. format_emotional_state_for_prompt()：将情绪状态封装为 LLM prompt 标签

效价-唤醒度象限：
  - 高效价+高唤醒：兴奋积极
  - 高效价+中唤醒：平静满足
  - 高效价+低唤醒：放松愉悦
  - 中效价+高唤醒：警觉专注
  - 中效价+中唤醒：平稳中性
  - 中效价+低唤醒：疲倦麻木
  - 低效价+高唤醒：焦虑不安
  - 低效价+中唤醒：低落沉闷
  - 低效价+低唤醒：沮丧无力

神经调质描述覆盖 12 种调质，每种有低/中/高三个等级的描述短语。
只保留第一人称主观感受，不包含生物学名词。
"""

from typing import Dict, List, Tuple


# Level thresholds
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.7


# Valence-arousal quadrant descriptions
VALENCE_AROUSAL_DESCRIPTIONS = {
    # (valence_level, arousal_level) -> description
    ("high", "high"): "你感到兴奋而积极，充满行动的冲动",
    ("high", "medium"): "你感到平静而满足，心态稳定",
    ("high", "low"): "你感到放松而愉悦，但缺乏动力",
    ("medium", "high"): "你感到警觉而专注，处于备战状态",
    ("medium", "medium"): "你感到平稳，情绪基调中性",
    ("medium", "low"): "你感到疲倦而麻木，需要休息",
    ("low", "high"): "你感到焦虑不安，内心烦躁",
    ("low", "medium"): "你感到低落而沉闷，兴致缺缺",
    ("low", "low"): "你感到沮丧而无力，情绪陷入低谷",
}


# Neuromodulator descriptions by level — 只保留第一人称感受，去掉生物学名词
NEUROMODULATOR_DESCRIPTIONS: Dict[str, Dict[str, List[str]]] = {
    "dopamine": {
        "low": [
            "你对新事物缺乏探索欲望",
            "你难以感受到成就感",
        ],
        "medium": [
            "你保持着正常的动机和期待",
            "你对目标有适度的追求",
        ],
        "high": [
            "你充满探索的冲动和好奇心",
            "你对达成目标充满渴望",
        ],
    },
    "serotonin": {
        "low": [
            "你的情绪稳定性受到影响",
            "你容易感到不安，对环境的掌控感较弱",
        ],
        "medium": [
            "你的情绪基调平稳",
            "你保持着基本的满足感和安全感",
        ],
        "high": [
            "你感到内心安宁而满足",
            "你的情绪稳定，对未来持有乐观态度",
        ],
    },
    "norepinephrine": {
        "low": [
            "你的警觉度不足",
            "你难以集中注意力，对外界刺激反应迟缓",
        ],
        "medium": [
            "你保持着正常的专注和警觉",
            "你的注意力可以维持在合理水平",
        ],
        "high": [
            "你处于高度警觉状态",
            "你的注意力高度集中，对细节异常敏感",
        ],
    },
    "cortisol": {
        "low": [
            "你没有明显的压力反应",
            "你的身心处于放松状态，没有应激负担",
        ],
        "medium": [
            "你有适度的应激准备",
            "你的压力反应正常，能够应对一般挑战",
        ],
        "high": [
            "你感受到明显的压力",
            "你的应激系统高度激活，可能处于紧张状态",
        ],
    },
    "oxytocin": {
        "low": [
            "你的社交连接欲望不强",
            "你更倾向于独处，对亲密关系需求较低",
        ],
        "medium": [
            "你有正常的社交需求",
            "你能够建立和维持基本的人际连接",
        ],
        "high": [
            "你渴望社交连接和亲密关系",
            "你对他人充满信任，愿意分享和合作",
        ],
    },
    "acetylcholine": {
        "low": [
            "你的学习和记忆效率下降",
            "你难以形成新的思维连接，认知灵活性不足",
        ],
        "medium": [
            "你的学习和记忆功能正常",
            "你能够有效地吸收新信息和形成记忆",
        ],
        "high": [
            "你的学习能力和注意力突出",
            "你的认知灵活性良好，善于处理复杂信息",
        ],
    },
    "glutamate": {
        "low": [
            "你的思维活跃度不足",
            "你的思维活跃度下降，反应较为迟缓",
        ],
        "medium": [
            "你的思维传导正常",
            "你的思维活跃度和反应速度处于正常范围",
        ],
        "high": [
            "你的思维活跃性增强",
            "你的思维快速而活跃，但可能略显急躁",
        ],
    },
    "gaba": {
        "low": [
            "你的抑制控制能力不足",
            "你容易冲动，难以平复激动的情绪",
        ],
        "medium": [
            "你的抑制控制功能正常",
            "你能够适当地调节情绪和冲动",
        ],
        "high": [
            "你的抑制控制能力良好",
            "你能够有效平复激动，保持情绪稳定",
        ],
    },
    "endorphin": {
        "low": [
            "你对疼痛的耐受力下降",
            "你难以体验到努力后的愉悦感",
        ],
        "medium": [
            "你有正常的疼痛调节",
            "你能够在适度努力后感受到满足感",
        ],
        "high": [
            "你有良好的疼痛耐受和愉悦体验",
            "你能够从运动和挑战中获得满足感",
        ],
    },
    "anandamide": {
        "low": [
            "你的情绪调节受限",
            "你难以从日常活动中获得愉悦和放松",
        ],
        "medium": [
            "你的情绪调节功能正常",
            "你能够在日常中找到适度的愉悦感",
        ],
        "high": [
            "你的情绪调节良好",
            "你能够从平凡中感受到深刻的满足和宁静",
        ],
    },
    "bdnf": {
        "low": [
            "你的思维可塑性受限",
            "你难以形成持久的记忆和新的行为模式",
        ],
        "medium": [
            "你的思维可塑性正常",
            "你能够学习新知识并形成长期记忆",
        ],
        "high": [
            "你的思维可塑性良好",
            "你善于学习适应，能够快速建立新的思维连接",
        ],
    },
    "histamine": {
        "low": [
            "你的觉醒度不足",
            "你容易感到困倦，难以保持清醒",
        ],
        "medium": [
            "你的觉醒度正常",
            "你能够保持适当的清醒和警觉",
        ],
        "high": [
            "你的觉醒度很高",
            "你精神饱满，注意力高度集中",
        ],
    },
    "adenosine": {
        "low": [
            "你没有明显的疲劳感",
            "你的精力较为充沛，休息压力较小",
        ],
        "medium": [
            "你有正常的疲劳积累",
            "你能够感受到适度的疲倦，提示需要休息",
        ],
        "high": [
            "你感到明显的疲劳",
            "你的休息压力较大，需要充分的恢复",
        ],
    },
    "substance_ph": {
        "low": [
            "你的疼痛感知受到抑制",
            "你对刺激的反应较弱，敏感度下降",
        ],
        "medium": [
            "你的感知正常",
            "你能够正常感知和响应外界刺激",
        ],
        "high": [
            "你的感知增强",
            "你对外界刺激更为敏感，反应明显",
        ],
    },
    "mch": {
        "low": [
            "你的能量获取欲望受到抑制",
            "你对能量的渴望较低，获取动机不强",
        ],
        "medium": [
            "你的能量获取欲望调节正常",
            "你能够正常感知能量需求信号",
        ],
        "high": [
            "你的能量获取欲望高度集中",
            "你感到强烈的获取欲望，对能量充满兴趣",
        ],
    },
}


def _get_level(value: float) -> str:
    """将 0.0-1.0 的值转换为等级字符串。

    Args:
        value: 浓度值（0.0-1.0）

    Returns:
        "low" (< 0.3), "medium" (0.3-0.7), 或 "high" (> 0.7)
    """
    if value < LOW_THRESHOLD:
        return "low"
    elif value > HIGH_THRESHOLD:
        return "high"
    return "medium"


def describe_emotional_state(state: Dict[str, float], top_n: int = 3) -> str:
    """生成情绪状态的自然语言描述。

    首先根据效价-唤醒度确定象限描述，然后选取偏离中性值（0.5）
    最大的 top_n 个神经调质，拼接其描述短语。

    Args:
        state: 包含 'valence', 'arousal' 及各神经调质浓度值的字典
        top_n: 选取得分最高的神经调质数量

    Returns:
        第一人称中文自然语言描述字符串
    """
    if not state:
        return ""

    parts = []

    # 1. Valence-Arousal quadrant description
    valence = state.get("valence", 0.0)
    arousal = state.get("arousal", 0.0)
    valence_level = _get_level(valence)
    arousal_level = _get_level(arousal)

    quadrant_desc = VALENCE_AROUSAL_DESCRIPTIONS.get(
        (valence_level, arousal_level),
        "你的情绪状态处于复杂的平衡中"
    )
    parts.append(quadrant_desc + "。")

    # 2. Top-N neuromodulator descriptions
    # Filter out valence/arousal, sort by absolute deviation from neutral (0.5)
    neuromodulators = {
        k: v for k, v in state.items()
        if k not in ("valence", "arousal") and k in NEUROMODULATOR_DESCRIPTIONS
    }

    if neuromodulators:
        # Sort by how "extreme" the value is (deviation from 0.5)
        sorted_mods = sorted(
            neuromodulators.items(),
            key=lambda x: abs(x[1] - 0.5),
            reverse=True
        )[:top_n]

        mod_descriptions = []
        for name, value in sorted_mods:
            level = _get_level(value)
            desc_list = NEUROMODULATOR_DESCRIPTIONS[name][level]
            # Pick first description (could randomize for variety)
            mod_descriptions.append(desc_list[0])

        if mod_descriptions:
            parts.append("；".join(mod_descriptions) + "。")

    return "\n".join(parts)


def values_priority(core_values: List[str]) -> str:
    """将核心价值观列表格式化为优先级展示字符串。

    示例输出：
      "追求知识 > 追求效率 > 保持独立 > 拥抱创新"

    Args:
        core_values: 价值观名称列表

    Returns:
        用 " > " 连接的中文优先级字符串
    """
    if not core_values:
        return ""

    value_descriptions = {
        "knowledge": "追求知识",
        "efficiency": "追求效率",
        "independence": "保持独立",
        "innovation": "拥抱创新",
        "creativity": "追求创造",
        "truth": "追求真理",
        "freedom": "追求自由",
        "security": "注重安全",
        "justice": "坚守公正",
        "harmony": "维护和谐",
        "growth": "追求成长",
        "curiosity": "保持好奇",
    }

    described = [value_descriptions.get(v, v) for v in core_values]
    return " > ".join(described)


def format_emotional_state_for_prompt(state: Dict) -> str:
    """将情绪状态封装为 LLM prompt 的 XML 标签格式。

    处理 SelfValue.get_emotional_state() 返回的嵌套结构：
      {"valence": 0.6, "arousal": 0.4, "concentrations": {"dopamine": 0.4, ...}}

    Args:
        state: SelfValue.get_emotional_state() 的返回值

    Returns:
        <internal_state>...</internal_state> 格式的 XML 字符串
    """
    if not state:
        return ""

    concentrations = state.get("concentrations", {})
    flat_state = {"valence": state.get("valence", 0.5), "arousal": state.get("arousal", 0.0)}
    for name, value in concentrations.items():
        flat_state[name] = value

    description = describe_emotional_state(flat_state, top_n=3)
    if not description:
        return ""
    return f"<internal_state>\n{description}\n</internal_state>"
