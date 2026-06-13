"""
HEXACO 六因素人格模型。

HEXACO 是心理学中被广泛验证的人格结构模型，包含六个维度：
- Honesty-Humility（诚实-谦逊）
- Emotionality（情绪性）
- Extraversion（外向性）
- Agreeableness（宜人性）
- Conscientiousness（尽责性）
- Openness to Experience（开放性）

在本项目中，每个维度的取值为 0.0-1.0 的连续值，默认中性值 0.5。
HEXACO 人格数据将注入到 LLM 推理上下文中，影响智能体的决策风格。
"""

from dataclasses import dataclass

TRAIT_NAMES = [
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
]


@dataclass
class HEXACO:
    """HEXACO 六因素人格特质数据类。

    每个特质取值为 0.0（极低）到 1.0（极高），默认 0.5（中等）。
    在 __post_init__ 中自动裁剪到 [0.0, 1.0] 范围。
    """
    honesty_humility: float = 0.5
    emotionality: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    conscientiousness: float = 0.5
    openness: float = 0.5

    def __post_init__(self):
        self.honesty_humility = max(0.0, min(1.0, self.honesty_humility))
        self.emotionality = max(0.0, min(1.0, self.emotionality))
        self.extraversion = max(0.0, min(1.0, self.extraversion))
        self.agreeableness = max(0.0, min(1.0, self.agreeableness))
        self.conscientiousness = max(0.0, min(1.0, self.conscientiousness))
        self.openness = max(0.0, min(1.0, self.openness))

    def to_dict(self) -> dict[str, float]:
        """将所有特质序列化为字典，便于日志记录和存储。"""
        return {trait: getattr(self, trait) for trait in TRAIT_NAMES}

    @classmethod
    def from_dict(cls, data: dict) -> "HEXACO":
        """从字典反序列化创建 HEXACO 实例。"""
        return cls(
            honesty_humility=data.get("honesty_humility", 0.5),
            emotionality=data.get("emotionality", 0.5),
            extraversion=data.get("extraversion", 0.5),
            agreeableness=data.get("agreeableness", 0.5),
            conscientiousness=data.get("conscientiousness", 0.5),
            openness=data.get("openness", 0.5),
        )