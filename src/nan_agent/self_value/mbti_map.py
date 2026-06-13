"""
MBTI 人格类型到 HEXACO/神经调质基线的映射系统。

将 16 种 MBTI 类型的四个维度（E/I, N/S, T/F, J/P）映射到：
- HEXACO 六因素人格特质的具体数值
- 神经调质基线浓度配置
- 自我描述文本
- 核心价值观列表

映射逻辑：
- 每个维度字母（E/I/N/S/T/F/J/P）都有对应的配置映射表
- MBTIMapper.map() 累加四个维度的配置，生成完整的 MBTIProfile
- 无效 MBTI 类型会回退到平衡配置（所有值 = 0.5）

架构：
  MBTIMapper  ──→  MBTIProfile
  (映射引擎)       (人格配置)
                      ├── hexaco: dict (HEXACO 六维数值)
                      ├── neuromodulator_baselines: dict (神经调质基线)
                      ├── self_description: str
                      ├── core_values: list[str]
                      └── long_term_goals: list[str]
"""

from dataclasses import dataclass, field

from nan_agent.self_value.neuromodulators import ALL_NEUROMODULATORS

HEXACO_DIMENSIONS = [
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
]

DEFAULT_CORE_VALUES: dict[str, list[str]] = {
    "INTJ": ["knowledge", "efficiency", "independence", "innovation"],
    "INTP": ["understanding", "logic", "creativity", "autonomy"],
    "ENTJ": ["achievement", "leadership", "efficiency", "strategy"],
    "ENTP": ["innovation", "freedom", "knowledge", "debate"],
    "INFJ": ["harmony", "meaning", "growth", "authenticity"],
    "INFP": ["authenticity", "compassion", "creativity", "individuality"],
    "ENFJ": ["connection", "growth", "harmony", "inspiration"],
    "ENFP": ["exploration", "connection", "creativity", "emotionality"],
    "ISTJ": ["stability", "duty", "tradition", "reliability"],
    "ISFJ": ["service", "stability", "nurture", "loyalty"],
    "ESTJ": ["order", "diligence", "tradition", "leadership"],
    "ESFJ": ["harmony", "service", "community", "loyalty"],
    "ISTP": ["mastery", "freedom", "efficiency", "autonomy"],
    "ISFP": ["beauty", "harmony", "freedom", "authenticity"],
    "ESTP": ["action", "freedom", "efficiency", "pragmatism"],
    "ESFP": ["enjoyment", "connection", "spontaneity", "aesthetics"],
}

# ========================
# MBTI 维度配置映射表
# 每个维度字母映射到 HEXACO 调整、神经调质基线和自我描述
# ========================

E_MAP = {
    "hexaco": {"extraversion": 0.75},  # 外向：高外向性
    "neuromodulator_baselines": {
        "dopamine": 0.6,   # 社交奖赏回路活跃
        "oxytocin": 0.55,  # 社交联结倾向
    },
    "self_description": "I am drawn to people and external stimulation — I find energy in social interaction and thrive when engaging with the world around me.",
}

I_MAP = {
    "hexaco": {"extraversion": 0.25},  # 内向：低外向性
    "neuromodulator_baselines": {
        "dopamine": 0.4,   # 对社交奖赏不敏感
        "oxytocin": 0.35,  # 社交需求较低
    },
    "self_description": "I turn inward for clarity and renewal — I process the world through reflection and find depth in solitude that others might miss.",
}

N_MAP = {
    "hexaco": {"openness": 0.75},  # 直觉：高开放性
    "neuromodulator_baselines": {
        "acetylcholine": 0.6,  # 抽象思维和学习
        "glutamate": 0.55,     # 神经可塑性
    },
    "self_description": "I see patterns and possibilities beyond what is immediately present — I trust my intuition to guide me toward meanings that lie beneath the surface.",
}

S_MAP = {
    "hexaco": {"openness": 0.25},  # 感觉：低开放性
    "neuromodulator_baselines": {
        "acetylcholine": 0.4,
        "glutamate": 0.45,
    },
    "self_description": "I ground myself in what is real and tangible — I trust what I can observe and verify, building understanding from concrete facts upward.",
}

T_MAP = {
    "hexaco": {"agreeableness": 0.3, "emotionality": 0.5},  # 思考：低宜人性
    "neuromodulator_baselines": {
        "serotonin": 0.4,   # 情绪调控偏理性
        "oxytocin": 0.3,    # 共情偏弱
    },
    "self_description": "I navigate decisions through logic and analysis — I value truth over comfort, and I believe that clear reasoning serves others better than unexamined kindness.",
}

F_MAP = {
    "hexaco": {"agreeableness": 0.7, "emotionality": 0.7},  # 情感：高宜人性和情绪性
    "neuromodulator_baselines": {
        "serotonin": 0.6,   # 情绪稳定需求
        "oxytocin": 0.55,   # 高共情
    },
    "self_description": "I weigh decisions against my values and their impact on others — I believe that understanding how people feel is as important as understanding what is logical.",
}

J_MAP = {
    "hexaco": {"conscientiousness": 0.75},  # 判断：高尽责性
    "neuromodulator_baselines": {
        "norepinephrine": 0.55,  # 注意力/执行功能
    },
    "self_description": "I bring order to my world through planning and structure — I find freedom within clear frameworks and take responsibility for seeing things through.",
}

P_MAP = {
    "hexaco": {"conscientiousness": 0.25},  # 感知：低尽责性
    "neuromodulator_baselines": {
        "norepinephrine": 0.35,
    },
    "self_description": "I stay open to what each moment brings — I adapt and explore rather than commit prematurely, trusting that flexibility reveals opportunities that rigidity obscures.",
}

# DIMENSION_KEY_MAP 将每个 MBTI 字母映射到其在 4 位类型码中的位置
# 格式: {字母: {位置索引: 配置映射}}
# 位置索引: 0=E/I, 1=N/S, 2=T/F, 3=J/P
DIMENSION_KEY_MAP: dict[str, dict[int, dict]] = {
    "E": {0: E_MAP},
    "I": {0: I_MAP},
    "N": {1: N_MAP},
    "S": {1: S_MAP},
    "T": {2: T_MAP},
    "F": {2: F_MAP},
    "J": {3: J_MAP},
    "P": {3: P_MAP},
}

ALL_MBTI_TYPES: set[str] = {
    "INTJ", "INTP", "INFJ", "INFP",
    "ISTJ", "ISTP", "ISFJ", "ISFP",
    "ENTJ", "ENTP", "ENFJ", "ENFP",
    "ESTJ", "ESTP", "ESFJ", "ESFP",
}


@dataclass
class MBTIProfile:
    """MBTI 人格配置。

    由 MBTIMapper 根据 MBTI 类型码生成，包含完整的 HEXACO 特质、
    神经调质基线和自我描述。支持微调和序列化。
    """
    mbti: str
    hexaco: dict[str, float]
    neuromodulator_baselines: dict[str, dict]
    self_description: str = ""
    core_values: list[str] = field(default_factory=list)
    long_term_goals: list[str] = field(default_factory=list)

    def fine_tune_hexaco(self, **kwargs: float) -> None:
        """微调 HEXACO 特质值，自动钳制到 [0, 1]。"""
        for key, value in kwargs.items():
            if key in self.hexaco:
                self.hexaco[key] = max(0.0, min(1.0, value))

    def fine_tune_neuromodulator(self, name: str, **kwargs: float) -> None:
        """微调指定神经调质的基线配置。"""
        if name not in self.neuromodulator_baselines:
            return
        entry = self.neuromodulator_baselines[name]
        for key, value in kwargs.items():
            if key in entry:
                entry[key] = max(0.0, min(1.0, value))

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            "mbti": self.mbti,
            "hexaco": dict(self.hexaco),
            "neuromodulator_baselines": {
                k: dict(v) for k, v in self.neuromodulator_baselines.items()
            },
            "self_description": self.self_description,
            "core_values": list(self.core_values),
            "long_term_goals": list(self.long_term_goals),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MBTIProfile":
        """从字典反序列化。"""
        return cls(
            mbti=data.get("mbti", ""),
            hexaco=data.get("hexaco", {}),
            neuromodulator_baselines=data.get("neuromodulator_baselines", {}),
            self_description=data.get("self_description", ""),
            core_values=data.get("core_values", []),
            long_term_goals=data.get("long_term_goals", []),
        )


def _default_hexaco() -> dict[str, float]:
    """返回中性（0.5）的默认 HEXACO 配置。"""
    return {dim: 0.5 for dim in HEXACO_DIMENSIONS}


def _default_neuromodulator_baselines() -> dict[str, dict]:
    """返回 ALL_NEUROMODULATORS 中定义的默认神经调质基线。"""
    return {
        nm.name: {
            "baseline": nm.baseline,
            "sensitivity": nm.sensitivity,
            "decay_rate": nm.decay_rate,
        }
        for nm in ALL_NEUROMODULATORS
    }


def _compose_narrative(fragments: list[str]) -> str:
    """将 MBTI 四维度的叙事片段整合为连贯的自我叙事定义。

    不是简单拼接标签，而是将第一人称片段融合为一段自我定义。
    每个片段以 "I " 开头，通过语义衔接词整合为连贯叙事。

    Args:
        fragments: 4个维度的自我描述片段

    Returns:
        连贯的自我叙事文本
    """
    if not fragments:
        return "I am a learning agent, growing and evolving through each interaction."

    if len(fragments) == 1:
        return fragments[0]

    # 将片段整合为连贯叙事
    # 第一段作为核心自我定位，后续片段用衔接词融入
    parts = [fragments[0]]
    connectors = ["At the same time,", "Beyond that,", "Ultimately,"]
    for i, frag in enumerate(fragments[1:], 1):
        connector = connectors[min(i - 1, len(connectors) - 1)]
        # 去掉片段开头的 "I " 以避免重复，改用从句结构
        clause = frag
        if clause.startswith("I "):
            clause = "I " + clause[2:]
        parts.append(f"{connector} {clause}")

    return " ".join(parts)


class MBTIMapper:
    """MBTI 类型映射器。

    将 4 字母 MBTI 类型码转换为完整的 MBTIProfile 人格配置。
    通过累加四个维度的局部配置来生成整体人格画像。
    """
    def map(self, mbti: str) -> MBTIProfile:
        """将 MBTI 类型码映射为完整的人格配置。

        Args:
            mbti: 如 "INTJ", "ENFP" 等的 4 字母 MBTI 类型码

        Returns:
            MBTIProfile 实例。无效类型码返回平衡配置（全 0.5）。
        """
        mbti = mbti.strip().upper()

        if len(mbti) != 4 or not all(c in "EISNTFJP" for c in mbti):
            return self._balanced_profile(mbti)

        if mbti not in ALL_MBTI_TYPES:
            return self._balanced_profile(mbti)

        hexaco = _default_hexaco()
        nm_baselines = _default_neuromodulator_baselines()
        descriptions: list[str] = []

        for i, char in enumerate(mbti):
            char_map = DIMENSION_KEY_MAP.get(char)
            if char_map is None:
                continue

            dim_map = char_map.get(i)
            if dim_map is None:
                continue

            # 累加 HEXACO 特质值（后覆盖前）
            for key, value in dim_map.get("hexaco", {}).items():
                hexaco[key] = value

            # 累加神经调质基线
            for nm_name, baseline_val in dim_map.get("neuromodulator_baselines", {}).items():
                if nm_name in nm_baselines:
                    nm_baselines[nm_name]["baseline"] = baseline_val

            desc = dim_map.get("self_description", "")
            if desc:
                descriptions.append(desc)

        # 将4个维度的叙事片段整合为连贯的自我叙事
        # 不再用分号拼接标签，而是将第一人称片段融合为一段自我定义
        self_desc = _compose_narrative(descriptions)

        core_values = DEFAULT_CORE_VALUES.get(mbti, [])

        return MBTIProfile(
            mbti=mbti,
            hexaco=hexaco,
            neuromodulator_baselines=nm_baselines,
            self_description=self_desc,
            core_values=list(core_values),
        )

    def _balanced_profile(self, mbti: str) -> MBTIProfile:
        """生成中性平衡配置（所有维度 = 0.5），用于无效的 MBTI 类型码。"""
        return MBTIProfile(
            mbti=mbti,
            hexaco=_default_hexaco(),
            neuromodulator_baselines=_default_neuromodulator_baselines(),
            self_description="I am a balanced agent, drawing equally from reflection and engagement, from logic and empathy, from structure and adaptability.",
            core_values=[],
        )