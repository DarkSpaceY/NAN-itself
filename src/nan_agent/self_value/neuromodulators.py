"""
神经调质（Neuromodulator）数据模型与全局注册表。

模拟大脑中 15 种核心神经调质的浓度变化，用于驱动智能体的情绪状态和决策倾向。
每种神经调质具有以下属性：
- baseline：静息基线浓度
- sensitivity：对环境刺激的敏感度（释放放大系数）
- decay_rate：向基线衰减的速率
- category：生化分类（单胺类/氨基酸类/神经肽类/其他）

分类参考：
- CATEGORY_MONOAMINE: 单胺类（多巴胺、血清素、去甲肾上腺素、组胺等）
- CATEGORY_AMINO: 氨基酸类（谷氨酸、GABA）
- CATEGORY_NEUROPEPTIDE: 神经肽类（催产素、内啡肽、食欲素、P物质、MCH）
- CATEGORY_OTHER: 其他（乙酰胆碱、皮质醇、内源性大麻素、肾上腺素）
"""

from dataclasses import dataclass, field

CATEGORY_MONOAMINE = "monoamine"
CATEGORY_AMINO = "amino_acid"
CATEGORY_NEUROPEPTIDE = "neuropeptide"
CATEGORY_OTHER = "other"


@dataclass
class Neuromodulator:
    """神经调质的静态定义模板。

    定义了神经调质的元信息，用于初始化 NeuromodulatorState。
    ALL_NEUROMODULATORS 列表中的每个条目都是 Neuromodulator 实例。
    """
    name: str
    category: str
    baseline: float = 0.5
    sensitivity: float = 0.3
    decay_rate: float = 0.01
    description: str = ""
    autoreceptor_ec50: float = 0.65
    autoreceptor_hill: float = 2.0


@dataclass
class NeuromodulatorState:
    """神经调质的运行时状态。

    维护每种神经调质的实时浓度，支持三种动力学操作：
    1. apply_phasic：事件触发的阶段性释放（含自受体浓度依赖性抑制）
    2. apply_decay：浓度向基线衰减
    3. apply_coupling_effect：来自耦合网络的其他神经调质影响

    自受体抑制机制（autoreceptor feedback）：
    基于神经科学文献，突触前自受体（如D2、5-HT1A/B、alpha-2）在浓度升高时
    抑制进一步释放，形成负反馈。使用 Hill 函数建模：
        release_factor = 1 / (1 + (concentration / EC50) ^ nH)
    - 浓度远低于 EC50 时：release_factor ≈ 1（无抑制）
    - 浓度 = EC50 时：release_factor = 0.5（半数抑制）
    - 浓度远高于 EC50 时：release_factor → 0（几乎完全抑制）
    """
    name: str
    concentration: float = 0.5
    baseline: float = 0.5
    sensitivity: float = 0.3
    decay_rate: float = 0.01
    category: str = ""
    description: str = ""
    autoreceptor_ec50: float = 0.65
    autoreceptor_hill: float = 2.0

    @classmethod
    def from_neuromodulator(cls, nm: Neuromodulator) -> "NeuromodulatorState":
        """从 Neuromodulator 模板创建初始状态实例。"""
        return cls(
            name=nm.name,
            concentration=nm.baseline,
            baseline=nm.baseline,
            sensitivity=nm.sensitivity,
            decay_rate=nm.decay_rate,
            category=nm.category,
            description=nm.description,
            autoreceptor_ec50=nm.autoreceptor_ec50,
            autoreceptor_hill=nm.autoreceptor_hill,
        )

    def apply_phasic(self, release: float) -> None:
        """应用阶段性释放，含自受体浓度依赖性抑制。

        核心公式：
            effective_release = release * sensitivity * autoreceptor_factor
        其中 autoreceptor_factor = 1 / (1 + (concentration / EC50) ^ nH)

        自受体抑制确保：
        - 浓度在基线附近时，释放效果接近最大
        - 浓度接近饱和时，额外释放效果急剧递减
        - 防止正反馈导致浓度爆炸

        文献依据：
        - D2自受体对DA释放的Hill抑制 (Best et al. 2009, Zhang et al. 2024)
        - 5-HT1B自受体对5-HT释放的抑制 (Best et al. 2020)
        - alpha-2自受体对NE释放的抑制 (频率依赖, Wu et al. 2002)
        """
        autoreceptor_factor = 1.0 / (1.0 + (self.concentration / self.autoreceptor_ec50) ** self.autoreceptor_hill)
        self.concentration += release * self.sensitivity * autoreceptor_factor
        self.concentration = max(0.0, min(1.0, self.concentration))

    def apply_decay(self, dt: float = 1.0) -> None:
        """向基线衰减：浓度 -= decay_rate * (浓度 - 基线) * dt。"""
        self.concentration -= self.decay_rate * (self.concentration - self.baseline) * dt

    def apply_coupling_effect(self, delta: float) -> None:
        """应用耦合网络计算的浓度增量，含浓度依赖性抑制，钳制在 [0, 1]。

        耦合效应同样受目标浓度的影响：浓度越高，额外增量效果递减。
        这模拟了高浓度时转运体饱和和受体脱敏等生物学机制。
        使用与自受体相同的 Hill 函数形式，但 EC50 略高（耦合效应
        比直接释放更难被抑制，因为是间接通路）。
        """
        # 耦合效应的浓度依赖性抑制（EC50比自受体高10%）
        coupling_ec50 = self.autoreceptor_ec50 * 1.1
        coupling_factor = 1.0 / (1.0 + (self.concentration / coupling_ec50) ** self.autoreceptor_hill)
        self.concentration += delta * coupling_factor
        self.concentration = max(0.0, min(1.0, self.concentration))

    def to_dict(self) -> dict:
        """序列化所有状态字段为字典。"""
        return {
            "name": self.name,
            "concentration": self.concentration,
            "baseline": self.baseline,
            "sensitivity": self.sensitivity,
            "decay_rate": self.decay_rate,
            "category": self.category,
            "description": self.description,
            "autoreceptor_ec50": self.autoreceptor_ec50,
            "autoreceptor_hill": self.autoreceptor_hill,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NeuromodulatorState":
        """从字典反序列化创建状态实例。"""
        return cls(
            name=data["name"],
            concentration=data.get("concentration", 0.5),
            baseline=data.get("baseline", 0.5),
            sensitivity=data.get("sensitivity", 0.3),
            decay_rate=data.get("decay_rate", 0.01),
            category=data.get("category", ""),
            description=data.get("description", ""),
            autoreceptor_ec50=data.get("autoreceptor_ec50", 0.65),
            autoreceptor_hill=data.get("autoreceptor_hill", 2.0),
        )


ALL_NEUROMODULATORS: list[Neuromodulator] = [
    # ═══════════════════════════════════════════════════════════════
    # 参数调整依据：神经科学文献综述
    #
    # baseline（基线浓度，归一化到 0-1）：
    #   谷氨酸/GABA 最高 (μM级, ~1-10 μM) → 0.7
    #   单胺类中等 (nM级, ~1-50 nM) → 0.4-0.5
    #   ACh 中等 (nM级) → 0.5
    #   组胺偏低 (nM级, 觉醒期高但全天均值低) → 0.35
    #   神经肽/激素最低 (pM-nM级) → 0.15-0.25
    #
    # sensitivity（敏感度/释放放大系数）：
    #   与基线反相关：低基线浓度 → 高受体亲和力 → 高敏感度
    #   神经肽 (pM级) → 0.5-0.6（极高亲和力，微量释放即有效应）
    #   单胺类 (nM级) → 0.3-0.45（中等亲和力）
    #   氨基酸 (μM级) → 0.15-0.2（低亲和力，需要大量释放）
    #
    # decay_rate（衰减速率，每步向基线回归的比例）：
    #   三类时间尺度（基于局部实质清除，非CSF/血浆半衰期）：
    #   超快 (ms级, 氨基酸+ACh)：EAAT/GAT/AChE清除 → 0.08-0.10
    #   快-中 (100ms-s级, 单胺类+组胺)：DAT/SERT/NET清除 → 0.03-0.05
    #   中-慢 (s-min级, 神经肽+激素)：局部肽酶+扩散+HPA反馈 → 0.02-0.05
    #   注：神经肽的CSF半衰期(3-30min)严重高估了局部清除速度。
    #       催产素实质<1min(Stark 1989)，扩散清除~100ms级。
    #       皮质醇HPA快速反馈~15min但慢于其他。
    #
    # autoreceptor_ec50（自受体半数抑制浓度，归一化）：
    #   控制Hill函数自抑制的阈值。浓度超过此值时释放急剧递减。
    #   单胺类：EC50较低(0.55-0.65)，自受体反馈灵敏
    #   神经肽：EC50中等(0.60-0.70)，肽类自抑制机制较弱
    #   氨基酸：EC50较高(0.75-0.80)，谷氨酸/GABA缺乏经典自受体
    #   皮质醇：EC50较高(0.70)，HPA负反馈较慢
    #
    # autoreceptor_hill（Hill系数）：
    #   控制自抑制曲线的陡峭度。nH=2为典型值（协同性结合）。
    # ═══════════════════════════════════════════════════════════════

    # ── 单胺类 (Monoamines) ──────────────────────────────────────
    Neuromodulator(
        name="dopamine",
        category=CATEGORY_MONOAMINE,
        baseline=0.45,       # 中等基线 (nM级)；纹状体高但全脑均值中等
        sensitivity=0.45,    # 较高敏感度；阶段性释放可达基线200-400%
        decay_rate=0.06,     # 快衰减；DAT介导清除，半衰期~100-200ms
        description="奖励预测、动机",
        autoreceptor_ec50=0.60,  # D2自受体EC50~15-30nM，归一化后偏低
        autoreceptor_hill=2.0,   # D2自受体协同性抑制
    ),
    Neuromodulator(
        name="serotonin",
        category=CATEGORY_MONOAMINE,
        baseline=0.50,       # 中等基线 (nM级)；5-HT神经元持续放电
        sensitivity=0.30,    # 中等敏感度；阶段性释放幅度较小(~50%基线)
        decay_rate=0.055,    # 快衰减；SERT介导清除，半衰期~200-500ms
        description="情绪稳定、冲动控制",
        autoreceptor_ec50=0.60,  # 5-HT1A/1B自受体EC50~30-100nM
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="norepinephrine",
        category=CATEGORY_MONOAMINE,
        baseline=0.40,       # 中等基线 (nM级)；LC紧张性放电
        sensitivity=0.40,    # 较高敏感度；应激时阶段性释放可达基线300%
        decay_rate=0.065,    # 快衰减；NET介导清除，半衰期~100-300ms
        description="警觉、注意力",
        autoreceptor_ec50=0.55,  # alpha-2自受体EC50~100-300nM，较灵敏
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="histamine",
        category=CATEGORY_MONOAMINE,
        baseline=0.35,       # 偏低基线；觉醒期高但全天均值较低
        sensitivity=0.35,    # 中等敏感度
        decay_rate=0.07,     # 较快衰减；HNMT快速代谢，半衰期~50-100ms
        description="觉醒、昼夜节律",
        autoreceptor_ec50=0.65,  # H3自受体，中等灵敏度
        autoreceptor_hill=2.0,
    ),

    # ── 氨基酸类 (Amino Acids) — 超快时间尺度 ──────────────────
    Neuromodulator(
        name="glutamate",
        category=CATEGORY_AMINO,
        baseline=0.70,       # 最高基线 (μM级)；脑内最丰富的兴奋性递质
        sensitivity=0.15,    # 最低敏感度；低亲和力受体，需大量释放
        decay_rate=0.10,     # 超快衰减；EAAT清除，半衰期~1-5ms
        description="兴奋性、突触可塑性",
        autoreceptor_ec50=0.80,  # 谷氨酸缺乏经典自受体，mGluR2/3异源受体为主
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="gaba",
        category=CATEGORY_AMINO,
        baseline=0.70,       # 最高基线 (μM级)；脑内最丰富的抑制性递质
        sensitivity=0.15,    # 最低敏感度；与谷氨酸对称
        decay_rate=0.10,     # 超快衰减；GAT清除，半衰期~1-5ms
        description="抑制性、焦虑调节",
        autoreceptor_ec50=0.80,  # GABA-B自受体存在但效应弱
        autoreceptor_hill=2.0,
    ),

    # ── 其他类 (Other) ──────────────────────────────────────────
    Neuromodulator(
        name="acetylcholine",
        category=CATEGORY_OTHER,
        baseline=0.50,       # 中等基线 (nM级)；基底前脑持续释放
        sensitivity=0.30,    # 中等敏感度；nAChR快速但mAChR较慢
        decay_rate=0.08,     # 超快衰减；AChE水解，半衰期~1-2ms
        description="学习、记忆编码",
        autoreceptor_ec50=0.70,  # M2/M4自受体，中等灵敏度
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="cortisol",
        category=CATEGORY_OTHER,
        baseline=0.25,       # 低基线；静息态低，昼夜节律波动大
        sensitivity=0.50,    # 高敏感度；GR/MR受体高亲和力，HPA轴正反馈
        decay_rate=0.015,    # 慢衰减；HPA轴负反馈，快速反馈~15min
        description="HPA应激轴",
        autoreceptor_ec50=0.70,  # GR/MR介导负反馈，阈值较高
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="endocannabinoids",
        category=CATEGORY_OTHER,
        baseline=0.25,       # 低基线；按需合成，静息水平低
        sensitivity=0.45,    # 较高敏感度；CB1受体高亲和力
        decay_rate=0.035,    # 中-快衰减；FAAH/MAGL代谢，半衰期~数十秒
        description="逆行信号、应激缓冲",
        autoreceptor_ec50=0.65,  # CB1自受体介导的释放抑制
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="adrenaline",
        category=CATEGORY_OTHER,
        baseline=0.15,       # 最低基线；静息态极低，仅在急性应激时飙升
        sensitivity=0.60,    # 最高敏感度；战逃反应需要大幅跃升
        decay_rate=0.06,     # 快衰减；COMT/MAO快速清除，半衰期~2-3min
        description="急性应激、战逃",
        autoreceptor_ec50=0.55,  # 肾上腺素自受体反馈灵敏
        autoreceptor_hill=2.0,
    ),

    # ── 神经肽类 (Neuropeptides) — 中-慢时间尺度 ──────────────
    # 注：decay_rate 基于局部实质清除（扩散+肽酶），而非CSF/血浆半衰期
    Neuromodulator(
        name="oxytocin",
        category=CATEGORY_NEUROPEPTIDE,
        baseline=0.20,       # 低基线 (pM级)；脉冲式释放
        sensitivity=0.55,    # 高敏感度；OTR极高亲和力，pM级即可激活
        decay_rate=0.04,     # 中衰减；实质局部清除<1min(Stark 1989)，扩散~100ms
        description="社会联结、信任",
        autoreceptor_ec50=0.65,  # 催产素自受体反馈
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="endorphins",
        category=CATEGORY_NEUROPEPTIDE,
        baseline=0.20,       # 低基线 (pM级)；运动/疼痛时释放
        sensitivity=0.55,    # 高敏感度；MOR高亲和力，通过GABA去抑制间接激活DA
        decay_rate=0.035,    # 中衰减；肽酶代谢，局部清除~5-30s
        description="疼痛缓解、欣快",
        autoreceptor_ec50=0.65,  # MOR自受体介导的释放抑制
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="orexin",
        category=CATEGORY_NEUROPEPTIDE,
        baseline=0.25,       # 低基线 (pM级)；觉醒期释放
        sensitivity=0.50,    # 高敏感度；OX1R/OX2R高亲和力
        decay_rate=0.03,     # 中衰减；食欲素A二硫键稳定，局部清除~数十秒
        description="觉醒稳定、觅食动机",
        autoreceptor_ec50=0.65,  # 食欲素自受体反馈
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="substance_p",
        category=CATEGORY_NEUROPEPTIDE,
        baseline=0.15,       # 最低基线之一 (pM级)；伤害性刺激时释放
        sensitivity=0.55,    # 高敏感度；NK1受体高亲和力
        decay_rate=0.04,     # 中衰减；NEP等肽酶快速代谢，局部清除~1-5s
        description="疼痛传递、情绪",
        autoreceptor_ec50=0.65,  # NK1自受体反馈
        autoreceptor_hill=2.0,
    ),
    Neuromodulator(
        name="mch",
        category=CATEGORY_NEUROPEPTIDE,
        baseline=0.20,       # 低基线 (pM级)；能量平衡调节
        sensitivity=0.45,    # 较高敏感度；MCHR1高亲和力
        decay_rate=0.03,     # 中衰减；氨肽酶M/NEP降解，局部清除~数十秒
        description="能量平衡、睡眠",
        autoreceptor_ec50=0.70,  # MCH自受体反馈
        autoreceptor_hill=2.0,
    ),
]