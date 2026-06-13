"""
神经调质动力学引擎（Neuromodulator Dynamics）。

核心动力学模拟器，负责驱动智能体的"神经化学"状态变化。每个时间步执行：
1. 接收事件触发的释放向量（phasic release）—— 仅事件驱动调质
2. 通过耦合网络计算神经调质间的相互影响
3. 所有神经调质浓度向基线衰减
4. 环境驱动调质的基线由昼夜节律调制（CircadianRhythm）

神经调质驱动源分类：
- 事件驱动（semantic）：由 ReleaseEvaluator 根据事件语义生成释放量
  dopamine, serotonin, norepinephrine, acetylcholine, glutamate,
  oxytocin, endorphins, adrenaline, substance_p, endocannabinoids
- 环境驱动（circadian）：由 CircadianRhythm 根据昼夜节律调制基线
  orexin, mch, cortisol, histamine, gaba

此外，根据神经调质浓度计算心理学的效价-唤醒度（Valence-Arousal）二维情绪坐标：
- 效价（Valence）：愉快-不愉快维度，由多巴胺/血清素/催产素/内啡肽（正向）和皮质醇/P物质（负向）决定
- 唤醒度（Arousal）：激活-平静维度，由去甲肾上腺素/组胺/肾上腺素/食欲素/多巴胺决定
"""

from nan_agent.logging.logger import get_logger

from nan_agent.self_value.circadian import CircadianRhythm, CIRCADIAN_NEUROMODULATORS
from nan_agent.self_value.coupling import CouplingNetwork
from nan_agent.self_value.neuromodulators import Neuromodulator, NeuromodulatorState

logger = get_logger(__name__)


class NeuromodulatorDynamics:
    """神经调质动力学引擎。

    管理所有神经调质的实时状态，驱动浓度变化、耦合计算和情绪坐标计算。
    环境驱动调质的基线由 CircadianRhythm 映射的昼夜节律调制。
    """
    def __init__(
        self,
        neuromodulators: list[Neuromodulator],
        coupling_network: CouplingNetwork,
        circadian: CircadianRhythm | None = None,
    ) -> None:
        self._coupling_network = coupling_network
        self._circadian = circadian or CircadianRhythm()
        self._states: dict[str, NeuromodulatorState] = {}

        for nm in neuromodulators:
            self._states[nm.name] = NeuromodulatorState.from_neuromodulator(nm)

        # 保存环境驱动调质的原始基线（昼夜调制在此基础上叠加）
        self._static_baselines: dict[str, float] = {}
        for name in CIRCADIAN_NEUROMODULATORS:
            state = self._states.get(name)
            if state is not None:
                self._static_baselines[name] = state.baseline

        # 初始应用一次昼夜调制
        self._apply_circadian_modulation()

        logger.info(
            "dynamics_initialized",
            neuromodulator_count=len(self._states),
            coupling_edge_count=len(coupling_network.get_edges()),
            circadian_modulated=list(self._static_baselines.keys()),
        )

    def get_states(self) -> dict[str, NeuromodulatorState]:
        """返回所有神经调质状态的副本。"""
        return dict(self._states)

    def apply_release_vector(self, release: dict[str, float]) -> None:
        """应用事件触发的释放向量。

        分两步：
        1. 对每个神经调质应用阶段性释放（phasic injection）
        2. 通过耦合网络计算并应用交叉影响

        Args:
            release: {神经调质名称: 释放量} 的字典，释放量 0.0-1.0
        """
        logger.debug("apply_release_vector", release=release)

        for name, amount in release.items():
            state = self._states.get(name)
            if state is not None:
                state.apply_phasic(amount)
                logger.debug("phasic_injection", target=name, amount=amount, concentration=state.concentration)
            else:
                logger.warning("unknown_neuromodulator_in_release", name=name)

        deltas = self._coupling_network.compute_coupling_deltas(self._states)

        for target_name, delta in deltas.items():
            state = self._states.get(target_name)
            if state is not None:
                state.apply_coupling_effect(delta)
                logger.debug("coupling_applied", target=target_name, delta=round(delta, 6), concentration=state.concentration)

        logger.debug("release_vector_applied", release_count=len(release), coupling_delta_count=len(deltas))

    def step(self, dt: float = 1.0) -> None:
        """执行一个时间步的衰减 + 昼夜基线调制。

        所有神经调质浓度向其基线衰减。
        环境驱动调质的基线由昼夜节律实时更新。
        """
        logger.debug("step", dt=dt)

        # 先更新昼夜调制基线
        self._apply_circadian_modulation()

        # 然后执行衰减（衰减目标是调制后的基线）
        for state in self._states.values():
            state.apply_decay(dt)

        logger.debug("step_complete", dt=dt)

    def _apply_circadian_modulation(self) -> None:
        """根据当前时间更新环境驱动调质的基线。

        昼夜偏移量叠加到原始静态基线上，形成动态基线。
        浓度如果低于新基线，则上拉到新基线（模拟昼夜驱动的基线漂移）。
        """
        offsets = self._circadian.compute_baseline_offsets()

        for name, offset in offsets.items():
            state = self._states.get(name)
            static_baseline = self._static_baselines.get(name)
            if state is not None and static_baseline is not None:
                # 新基线 = 静态基线 + 昼夜偏移，钳制在 [0.05, 0.95]
                new_baseline = max(0.05, min(0.95, static_baseline + offset))
                state.baseline = new_baseline
                # 如果浓度低于新基线，上拉到新基线
                # 这模拟了昼夜驱动的基线漂移：夜间GABA/MCH基线升高，
                # 即使没有事件释放，浓度也会被基线拉上去
                if state.concentration < new_baseline:
                    state.concentration = new_baseline

    def compute_valence_arousal(self) -> dict[str, float]:
        """计算效价-唤醒度（Valence-Arousal）二维情绪坐标。

        使用偏离基线的偏差（deviation）而非绝对浓度来计算，确保：
        1. 基线状态下 valence ≈ 0.5, arousal ≈ 0.5（中性）
        2. 正向和负向权重对称（总和均为 1.0）
        3. 浓度升高和降低都能产生合理的情绪偏移
        4. 使用 tanh 压缩大偏差，防止极端浓度导致锁死

        Valence 公式（基于基线偏差，tanh 压缩）：
            positive = tanh((DA_dev*0.35 + 5HT_dev*0.25 + OXT_dev*0.20 + END_dev*0.20) * 2.2)
            negative = tanh((CORT_dev*0.45 + SP_dev*0.30 + ADR_dev*0.25) * 2.2)
            valence = 0.5 + 0.5*(positive - negative)

        Arousal 公式（基于基线偏差，tanh 压缩，多源抑制）：
            activating_raw = NE_dev*0.30 + HIS_dev*0.20 + DA_dev*0.30
                           + ADR_dev*0.15 + ORX_dev*0.15
            inhibiting_raw = GABA_dev*0.70 + MCH_dev*0.50
            eCB_disinhibition = max(0, eCB_dev) * 0.20
            net_activation = activating_raw - inhibiting_raw + eCB_disinhibition
            arousal = 0.5 + 0.5*tanh(net_activation * 2.8)

        文献依据：
        - Valence正向: DA(奖励信号,主驱动)、5-HT(情绪稳定)、OXT(社会满足)、END(欣快)
        - Valence负向: Cortisol(应激,主驱动,双通道:昼夜基线+耦合应激)、
                       Substance P(疼痛/不适)、Adrenaline(恐惧)
        - Arousal激活: NE(警觉,主驱动)、Histamine(觉醒,昼夜调制)、DA(动机性激活)、
                       Adrenaline(急性应激)、Orexin(觉醒维持,昼夜调制)
        - Arousal抑制: GABA(时相性偏离→促眠, VLPO通路, 昼夜调制; DeWoskin 2015 PNAS)、
                       MCH(与GABA共释放协同促眠, 昼夜调制; Sapin 2010, Li 2025)
        - Arousal去抑制: eCB(CB1门控GABA释放→去抑制→促觉醒; Kesner 2020)

        Returns:
            {"valence": 0.0-1.0, "arousal": 0.0-1.0}
        """
        import math

        c = {name: state.concentration for name, state in self._states.items()}
        b = {name: state.baseline for name, state in self._states.items()}

        # 计算偏离基线的偏差
        def dev(name: str) -> float:
            return c.get(name, 0.5) - b.get(name, 0.5)

        # Valence: 正向权重总和 = 1.0, 负向权重总和 = 1.0
        # tanh 压缩防止大偏差导致极端值，但用 2x 缩放保留足够的动态范围
        positive_raw = (
            dev("dopamine") * 0.35       # 奖励信号，主驱动
            + dev("serotonin") * 0.25     # 情绪稳定/满足
            + dev("oxytocin") * 0.20      # 社会联结满足
            + dev("endorphins") * 0.20    # 欣快感
        )
        negative_raw = (
            dev("cortisol") * 0.45        # 应激，主负向驱动
            + dev("substance_p") * 0.30    # 疼痛/不适
            + dev("adrenaline") * 0.25     # 恐惧/威胁
        )
        positive = math.tanh(positive_raw * 2.2)
        negative = math.tanh(negative_raw * 2.2)
        valence = 0.5 + 0.5 * (positive - negative)

        # Arousal: 基于偏差的多源激活-抑制模型
        # 激活源: NE(警觉), HIS(觉醒), DA(动机), ADR(应激), ORX(觉醒维持)
        activating_raw = (
            dev("norepinephrine") * 0.30  # 警觉/注意力，主驱动
            + dev("histamine") * 0.20        # 觉醒
            + dev("dopamine") * 0.30         # 动机性激活（与NE并列主驱动）
            + dev("adrenaline") * 0.15       # 急性应激激活
            + dev("orexin") * 0.15           # 觉醒维持
        )
        # 抑制源: GABA(时相性偏离→促眠), MCH(协同促眠)
        # 文献: GABA偏差法(非绝对浓度)基于DeWoskin 2015 PNAS计算模型
        #       MCH 85%共表达GABA,协同促眠(Sapin 2010, Li 2025)
        inhibiting_raw = (
            dev("gaba") * 0.70              # GABA时相性偏离→VLPO促眠通路
            + dev("mch") * 0.50             # MCH协同促眠
        )
        # 去抑制: eCB通过CB1门控GABA释放→去抑制→促觉醒
        # 文献: eCB选择性减少GABA释放(Kesner 2020), 仅正偏离有效
        eCB_disinhibition = max(0.0, dev("endocannabinoids")) * 0.20

        net_activation = activating_raw - inhibiting_raw + eCB_disinhibition
        arousal = 0.5 + 0.5 * math.tanh(net_activation * 2.8)

        valence = max(0.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))

        logger.debug("valence_arousal_computed", valence=round(valence, 4), arousal=round(arousal, 4))

        return {"valence": valence, "arousal": arousal}

    def get_personality_context(self) -> dict:
        """获取当前人格上下文，用于注入 LLM prompt。

        Returns:
            {"neuromodulators": {名称: 浓度}, "valence": 效价, "arousal": 唤醒度}
        """
        v_a = self.compute_valence_arousal()

        return {
            "neuromodulators": {name: state.concentration for name, state in self._states.items()},
            "valence": v_a["valence"],
            "arousal": v_a["arousal"],
        }