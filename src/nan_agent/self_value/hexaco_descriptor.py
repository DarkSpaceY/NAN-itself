"""HEXACO 人格 5 级中文自然语言描述器。

将 HEXACO 六维度的数值（0.0-1.0）转换为中文自然语言描述。
每个维度分为 5 个等级：极低、较低、中等、较高、极高。

示例输出：
  "高度开放好奇、较强尽责自律、中等情绪稳定、较低外向、中等宜人、较高诚实谦逊"
"""

LOW_THRESHOLD = 0.2
MODERATE_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.6
VERY_HIGH_THRESHOLD = 0.8


def _get_level(value: float) -> str:
    """根据数值返回对应的等级标签。

    Args:
        value: 0.0-1.0 之间的特质值

    Returns:
        等级字符串：very_low / low / moderate / high / very_high
    """
    if value < LOW_THRESHOLD:
        return "very_low"
    elif value < MODERATE_THRESHOLD:
        return "low"
    elif value < HIGH_THRESHOLD:
        return "moderate"
    elif value < VERY_HIGH_THRESHOLD:
        return "high"
    return "very_high"


def describe_hexaco(hexaco: dict[str, float]) -> str:
    """将 HEXACO 数值转换为简洁的中文描述。

    Args:
        hexaco: 包含六个维度数值的字典，键为 TRAIT_NAMES 中的名称

    Returns:
        中文描述字符串，如 "高度开放好奇、较强尽责自律、中等情绪敏感..."

    Example:
        >>> describe_hexaco({"honesty_humility": 0.7, "emotionality": 0.5, ...})
        "较高诚实谦逊、中等情绪敏感、..."
    """
    if not hexaco:
        return ""

    level_labels = {
        "very_low": "极低", "low": "较低", "moderate": "中等",
        "high": "较高", "very_high": "极高",
    }
    trait_names = {
        "honesty_humility": "诚实谦逊",
        "emotionality": "情绪敏感",
        "extraversion": "外向",
        "agreeableness": "宜人合作",
        "conscientiousness": "尽责自律",
        "openness": "开放好奇",
    }

    parts = []
    for key, cn_name in trait_names.items():
        value = hexaco.get(key, 0.5)
        level = _get_level(value)
        parts.append(f"{level_labels.get(level, '中等')}{cn_name}")

    return "、".join(parts)


