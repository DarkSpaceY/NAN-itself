"""
神经调质耦合网络（Coupling Network）。

模拟大脑中不同神经调质之间的相互作用关系。每种神经调质浓度的变化
会通过耦合边（CouplingEdge）影响其他神经调质的浓度。

支持三种相互作用类型：
- inhibit：抑制关系，源浓度升高 → 目标浓度降低
- activate：激活关系，源浓度升高 → 目标浓度升高
- synergize：协同关系，源浓度升高 → 目标浓度小幅升高（0.5 倍系数）

耦合网络基于已知的神经科学文献构建，预设了 26 条已知的神经调质相互作用边。
"""

from dataclasses import dataclass, field

from nan_agent.self_value.neuromodulators import NeuromodulatorState


@dataclass
class CouplingEdge:
    """耦合网络中的一条有向边。

    表示源神经调质对目标神经调质的影响关系。
    """
    source: str
    target: str
    relation: str  # "inhibit" | "activate" | "synergize"
    strength: float = 0.3


@dataclass
class CouplingNetwork:
    """神经调质耦合网络。

    管理所有耦合边，并根据当前状态计算各神经调质的浓度增量。
    如果未提供边列表，则在 __post_init__ 中自动初始化已知的耦合关系。
    """
    edges: list[CouplingEdge] = field(default_factory=list)

    def __post_init__(self):
        if not self.edges:
            self._init_known_edges()

    def _init_known_edges(self):
        """初始化基于神经科学文献的已知耦合关系（35 条边）。

        strength 分级依据文献中的交互强度：
          - 0.40-0.50: Strong（直接受体介导，主要驱动通路）
          - 0.25-0.35: Moderate（间接受体或次要通路）
          - 0.10-0.20: Weak（间接多突触或条件性效应）

        文献来源：
          - DA↔5-HT: Olijslagers 2006, Howell & Cunningham 2015
          - DA↔NE: Ranjbar-Slamloo & Fazlali 2020
          - Cortisol→DA/5-HT/NE: Skorzewska 2021, Nasereddin 2024
          - Glu→DA/NE/5-HT: Kadriu 2019
          - GABA→DA/NE/5-HT: Fogaca & Duman 2019
          - Orexin→NE/DA/Hist/ACh: Scammell 2017, Chen 2024
          - ACh→DA/NE: Myslivecek 2021
          - Oxytocin→Cortisol: Takayanagi & Onaka 2022
          - Endocannabinoids→Cortisol: Maldonado 2020
          - Endorphins→DA: Grothusen 2022
          - Adrenaline↔NE: Axelrod & Reisine 1984
        """
        known = [
            # ── 多巴胺 ↔ 血清素 ──────────────────────────────────
            # DA→5-HT: D2受体在DRN 5-HT神经元上，直接抑制
            ("dopamine", "serotonin", "inhibit", 0.08),
            # 5-HT→DA: 净效应弱抑制（5-HT2C抑制占优）
            ("serotonin", "dopamine", "inhibit", 0.06),

            # ── 多巴胺 ↔ 去甲肾上腺素 ──────────────────────────
            # DA→NE: 共享生物合成通路，PFC中协同
            ("dopamine", "norepinephrine", "synergize", 0.10),
            # NE→DA: LC可共释放DA
            ("norepinephrine", "dopamine", "activate", 0.08),

            # ── 血清素 → 去甲肾上腺素 ──────────────────────────
            # 5-HT→NE: 净效应弱抑制
            ("serotonin", "norepinephrine", "inhibit", 0.06),

            # ── 乙酰胆碱 → 内源性大麻素 ────────────────────────
            # ACh→eCB: 胆碱能激活促进eCB合成
            ("acetylcholine", "endocannabinoids", "activate", 0.07),

            # ── 皮质醇 → 单胺类（慢性抑制）──────────────────────
            # 慢性GC暴露下调DA/5-HT/NE
            ("cortisol", "dopamine", "inhibit", 0.10),
            ("cortisol", "serotonin", "inhibit", 0.09),
            ("cortisol", "norepinephrine", "inhibit", 0.08),

            # ── 食欲素 → 觉醒系统 ──────────────────────────────
            # Orexin→NE: 对LC的主要兴奋输入
            ("orexin", "norepinephrine", "activate", 0.12),
            # Orexin→DA: VTA中等兴奋
            ("orexin", "dopamine", "activate", 0.08),
            # Orexin→Histamine: 对TMN的强兴奋输入
            ("orexin", "histamine", "activate", 0.12),
            # Orexin→ACh: BF/LDT兴奋
            ("orexin", "acetylcholine", "activate", 0.10),

            # ── 谷氨酸 → 单胺类 ────────────────────────────────
            # Glu→DA: VTA DA神经元的主要兴奋驱动
            ("glutamate", "dopamine", "activate", 0.12),
            # Glu→NE: LC NE神经元的主要兴奋驱动
            ("glutamate", "norepinephrine", "activate", 0.12),
            # Glu→5-HT: DRN直接兴奋，但间接GABA抵消部分
            ("glutamate", "serotonin", "activate", 0.08),

            # ── GABA → 单胺类 ──────────────────────────────────
            # GABA→DA: VTA GABA能紧张性抑制DA神经元
            ("gaba", "dopamine", "inhibit", 0.12),
            # GABA→NE: LC GABA能抑制
            ("gaba", "norepinephrine", "inhibit", 0.10),
            # GABA→5-HT: DRN局部GABA能抑制
            ("gaba", "serotonin", "inhibit", 0.08),

            # ── 内啡肽 → 多巴胺 ────────────────────────────────
            # Endorphins→DA: MOR抑制VTA GABA中间神经元→去抑制DA (间接)
            ("endorphins", "dopamine", "activate", 0.10),

            # ── 催产素 / 内源性大麻素 → 皮质醇 ─────────────────
            # Oxytocin→Cortisol: 抑制PVN CRF
            ("oxytocin", "cortisol", "inhibit", 0.09),
            # eCB→Cortisol: CB1紧张性抑制HPA轴
            ("endocannabinoids", "cortisol", "inhibit", 0.08),

            # ── 肾上腺素 ↔ 去甲肾上腺素 ────────────────────────
            # 肾上腺素与NE共释放，协同激活
            ("adrenaline", "norepinephrine", "synergize", 0.10),
            # 皮质醇→肾上腺素: PNMT诱导NE→EPI转化
            ("cortisol", "adrenaline", "activate", 0.09),

            # ── 乙酰胆碱 → 单胺类 ──────────────────────────────
            # ACh→DA: nAChR在DA末梢强激活
            ("acetylcholine", "dopamine", "activate", 0.10),
            # ACh→NE: nAChR在LC兴奋NE
            ("acetylcholine", "norepinephrine", "activate", 0.08),

            # ── 新增：文献支持但原版缺失的交互 ──────────────────
            # GABA→内啡肽的调制 (Weak)
            ("gaba", "endorphins", "inhibit", 0.05),
            # 多巴胺→乙酰胆碱：DA D2抑制纹状体ACh
            ("dopamine", "acetylcholine", "inhibit", 0.07),
            # 血清素→催产素：5-HT促进催产素释放
            ("serotonin", "oxytocin", "activate", 0.07),
            # 肾上腺素→皮质醇：急性应激时EPI通过CRF促进皮质醇
            # 强度从0.07提升到0.15：cortisol不再由LLM直接释放，
            # 改由昼夜基线+耦合驱动双通道，需要更强的耦合来传递应激信号
            ("adrenaline", "cortisol", "activate", 0.15),
            # 催产素→血清素：OT促进5-HT释放
            ("oxytocin", "serotonin", "activate", 0.07),
            # 内啡肽→GABA：MOR抑制GABA释放
            ("endorphins", "gaba", "inhibit", 0.08),
            # 食欲素→内啡肽：食欲素促进奖赏通路 (Weak)
            ("orexin", "endorphins", "activate", 0.06),
        ]
        for entry in known:
            if len(entry) == 3:
                src, tgt, rel = entry
                strength = 0.3  # 默认强度
            else:
                src, tgt, rel, strength = entry
            self.edges.append(CouplingEdge(source=src, target=tgt, relation=rel, strength=strength))

    def add_edge(self, source: str, target: str, relation: str, strength: float = 0.3) -> None:
        """添加一条新的耦合边。"""
        self.edges.append(CouplingEdge(source=source, target=target, relation=relation, strength=strength))

    def get_edges(self) -> list[CouplingEdge]:
        """返回所有耦合边的副本。"""
        return list(self.edges)

    def compute_coupling_deltas(self, states: dict[str, NeuromodulatorState]) -> dict[str, float]:
        """根据当前神经调质状态计算各目标神经调质的浓度增量。

        对于每条边，计算源神经调质偏离基线的程度，乘以关系类型和强度，
        累加到目标神经调质的增量中。

        为防止正反馈导致浓度爆炸，对每个目标的累计 delta 进行钳制：
          max_coupling_delta = 0.15（单步耦合效应不超过 ±0.15）

        Args:
            states: 当前所有神经调质的状态字典

        Returns:
            {目标神经调质名称: 浓度增量} 的字典
        """
        deltas: dict[str, float] = {}
        max_coupling_delta = 0.05

        for edge in self.edges:
            source_state = states.get(edge.source)
            target_name = edge.target

            if source_state is None or edge.target not in states:
                continue

            deviation = source_state.concentration - source_state.baseline
            if abs(deviation) < 0.01:
                continue

            delta = edge.strength * deviation

            if edge.relation == "inhibit":
                delta = -delta
            elif edge.relation == "synergize":
                delta = delta * 0.5

            deltas[target_name] = deltas.get(target_name, 0.0) + delta

        # 钳制每个目标的累计耦合增量，防止正反馈爆炸
        for target_name in deltas:
            deltas[target_name] = max(-max_coupling_delta, min(max_coupling_delta, deltas[target_name]))

        return deltas