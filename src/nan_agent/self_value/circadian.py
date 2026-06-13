"""昼夜节律调制器（Circadian Rhythm Modulator）。

根据真实系统时间计算环境驱动神经调质的昼夜基线调制。

神经调质分为两类驱动源：
1. 事件驱动（semantic）：由 LLM ReleaseEvaluator 根据事件语义生成释放量
   - dopamine, serotonin, norepinephrine, acetylcholine, glutamate,
     oxytocin, endorphins, adrenaline, substance_p, endocannabinoids
2. 环境驱动（circadian）：由昼夜节律程序化计算基线调制
   - orexin: 觉醒维持，白天高夜间低
   - mch: 促眠，夜间高白天低，与 orexin 反相关
   - cortisol: 皮质醇觉醒反应(CAR) + 昼夜基线
   - histamine: 觉醒系统，白天高夜间低
   - gaba: VLPO 促眠通路，夜间基线升高

文献依据：
- Orexin/MCH 反相关：Saper 2010, Scammell 2015
- Cortisol CAR：Clow 2010, Pruessner 1997
- Histamine 昼夜节律：Haas 2008
- GABA-VLPO 昼夜调制：Saper 2005
"""

import math
from datetime import datetime, timezone

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)

# 环境驱动调质名称集合
CIRCADIAN_NEUROMODULATORS = frozenset({
    "orexin", "mch", "cortisol", "histamine", "gaba",
})

# 事件驱动调质名称集合（LLM ReleaseEvaluator 负责）
EVENT_DRIVEN_NEUROMODULATORS = frozenset({
    "dopamine", "serotonin", "norepinephrine", "acetylcholine",
    "glutamate", "oxytocin", "endorphins", "adrenaline",
    "substance_p", "endocannabinoids",
})


def _hour_of_day(dt: datetime | None = None) -> float:
    """返回当前小时数（0.0-24.0），含分钟精度。

    Args:
        dt: 可选的 datetime，默认使用当前 UTC 时间
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.hour + dt.minute / 60.0


class CircadianRhythm:
    """昼夜节律调制器。

    根据真实时间计算环境驱动神经调质的基线调制量。
    每个调质返回一个 baseline_offset（-0.3 ~ +0.3），
    叠加到原始基线上形成昼夜调制后的有效基线。

    调制函数使用余弦/正弦拟合昼夜节律：
    - 觉醒类（orexin, histamine）：白天峰值 ~14:00，夜间谷值 ~02:00
    - 促眠类（mch, gaba夜间分量）：与觉醒类反相
    - 皮质醇：CAR 峰值 ~07:00（觉醒后30min），夜间谷值 ~00:00
    """

    def __init__(self, reference_dt: datetime | None = None):
        """初始化昼夜节律调制器。

        Args:
            reference_dt: 参考时间点，用于测试注入。默认使用实时时钟。
        """
        self._reference_dt = reference_dt

    def _get_hour(self) -> float:
        """获取当前小时数。"""
        return _hour_of_day(self._reference_dt)

    def compute_baseline_offsets(self) -> dict[str, float]:
        """计算所有环境驱动调质的基线偏移量。

        Returns:
            {调质名称: baseline_offset} 字典，offset 范围约 -0.25 ~ +0.25
        """
        hour = self._get_hour()

        offsets = {
            "orexin": self._orexin_offset(hour),
            "mch": self._mch_offset(hour),
            "cortisol": self._cortisol_offset(hour),
            "histamine": self._histamine_offset(hour),
            "gaba": self._gaba_offset(hour),
        }

        logger.debug(
            "circadian_offsets_computed",
            hour=round(hour, 2),
            offsets={k: round(v, 4) for k, v in offsets.items()},
        )

        return offsets

    def _orexin_offset(self, hour: float) -> float:
        """Orexin 觉醒维持：白天高，夜间低。

        峰值 ~14:00，谷值 ~02:00。
        余弦函数：cos(2π(hour - 14) / 24) 在 14:00 = 1，02:00 = -1
        """
        phase = math.cos(2 * math.pi * (hour - 14) / 24)
        return phase * 0.20  # ±0.20 偏移

    def _mch_offset(self, hour: float) -> float:
        """MCH 促眠：夜间高，白天低，与 orexin 反相。

        峰值 ~02:00，谷值 ~14:00。
        文献：MCH 神经元在 REM 睡眠期活跃（Hassani 2009）
        """
        phase = math.cos(2 * math.pi * (hour - 2) / 24)
        return phase * 0.20  # ±0.20 偏移

    def _cortisol_offset(self, hour: float) -> float:
        """Cortisol 皮质醇觉醒反应 + 昼夜基线。

        CAR 峰值 ~07:00（觉醒后30min），夜间谷值 ~00:00-02:00。
        使用偏移余弦：cos(2π(hour - 7) / 24)
        文献：Pruessner 1997, Clow 2010
        """
        phase = math.cos(2 * math.pi * (hour - 7) / 24)
        return phase * 0.15  # ±0.15 偏移（cortisol 波动幅度小于 orexin）

    def _histamine_offset(self, hour: float) -> float:
        """Histamine 觉醒系统：白天高，夜间低。

        峰值 ~14:00，谷值 ~02:00。与 orexin 同相但幅度较小。
        文献：Haas 2008, TMN 神经元放电率昼夜变化
        """
        phase = math.cos(2 * math.pi * (hour - 14) / 24)
        return phase * 0.15  # ±0.15 偏移

    def _gaba_offset(self, hour: float) -> float:
        """GABA VLPO 促眠通路：夜间基线升高。

        峰值 ~02:00，谷值 ~14:00。与 MCH 同相但幅度更小。
        文献：Saper 2005, VLPO GABA 神经元在睡眠期高放电
        """
        phase = math.cos(2 * math.pi * (hour - 2) / 24)
        return phase * 0.10  # ±0.10 偏移（GABA 基线已高，偏移幅度小）

    def get_phase_description(self) -> str:
        """返回当前昼夜阶段的文字描述。"""
        hour = self._get_hour()
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
