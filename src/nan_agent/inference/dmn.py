"""
GoT Spontaneous Growth Strategy — 图思维自发增长策略
------------------------------------------------------
模拟大脑默认模式网络 (DMN) 的自发思维功能，当智能体处于"空闲"状态时，
通过多种锚点选择策略和生成模式，自发产生思想、联想和反思。

核心设计参考：
- Sakana AI CTM (2025): tick-based rhythm，轻量/重量 tick 交替
- Buehler (2025): 自组织临界态，~12% 的"惊奇边"驱动持续创新
- CHIMERA (2025): 创意重组与概念融合
- SYNAPSE (2025): 扩散激活在 LLM 智能体记忆中的应用

架构：
1. GrowthStrategy: 锚点选择策略（结构洞/惊奇追踪/概念融合/心智游走）
2. GenerationMode: 生成模式（自由联想/类比映射/批判反思/假设生成）
3. Tick-based rhythm: 轻量 tick（激活传播+权重更新+结构洞检测）与
   重量 tick（锚点选择+LLM 生成新节点）交替，默认 5:1
4. 与 GoTGraph 深度集成：新节点使用 NodeType.INSIGHT + NodeOrigin.SPONTANEOUS，
   新边使用对应 EdgeType（ANALOGIZES/BRANCH/QUESTIONS 等）

向后兼容：
- DMNGenerator 作为 SpontaneousGrowthStrategy 的别名保留
- should_generate(pool_size) 和 generate(max_nodes) 方法签名不变
"""

import hashlib
import json
import math
import random
import re
from enum import Enum
from typing import Optional, Union

from nan_agent.inference.graph import (
    EDGE_DECAY_RATES,
    EdgeType,
    GoTEdge,
    GoTGraph,
    GoTNode,
    NodeOrigin,
    NodeType,
)
from nan_agent.logging.logger import get_logger
from nan_agent.model.types import MultiModalInput

logger = get_logger(__name__)

# 默认记忆检索查询词
_DEFAULT_QUERY_TERMS = [
    "general reflection on recent experiences",
    "creative thinking and exploration",
    "interesting knowledge connections",
    "personal insight and growth",
    "abstract concept exploration",
    "pattern recognition across domains",
    "novel ideas and associations",
    "philosophical reflection",
    "scientific curiosity",
    "meaningful connections between memories",
]


# ═══════════════════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════════════════


class GrowthStrategy(Enum):
    """锚点选择策略

    决定从图中选择哪个（或哪些）节点作为自发思维的起点。

    策略设计参考：
    - STRUCTURAL_HOLE: Burt (2004) 结构洞理论，连接不同社区的关键位置
    - SURPRISE_TRACKING: Buehler (2025) 惊奇边驱动创新
    - CONCEPTUAL_BLENDING: Fauconnier & Turner 概念融合理论 + CHIMERA (2025)
    - MIND_WANDERING: 小世界网络上的随机游走，模拟心智游走
    """

    STRUCTURAL_HOLE = "structural_hole"      # 选择连接最少的节点（填补图中的空隙）
    SURPRISE_TRACKING = "surprise_tracking"  # 选择惊奇度最高的节点
    CONCEPTUAL_BLENDING = "conceptual_blending"  # 选择语义距离中等的节点对并尝试融合
    MIND_WANDERING = "mind_wandering"        # 从随机节点出发沿边随机游走 2-3 步


class GenerationMode(Enum):
    """生成模式

    决定如何将锚点内容转化为 LLM prompt，以及生成什么样的思维节点。

    模式与锚点策略的对应关系（非强制，由 select_mode 动态决定）：
    - STRUCTURAL_HOLE → FREE_ASSOCIATION（填补空隙需要自由联想）
    - SURPRISE_TRACKING → CRITICAL_REFLECTION（惊奇点值得批判审视）
    - CONCEPTUAL_BLENDING → ANALOGY_MAPPING（概念融合本质是类比）
    - MIND_WANDERING → FREE_ASSOCIATION 或 HYPOTHESIS_GENERATION
    """

    FREE_ASSOCIATION = "free_association"        # 给锚点内容 + "自由联想" prompt
    ANALOGY_MAPPING = "analogy_mapping"          # 给两个节点的结构描述 + "寻找结构相似性" prompt
    CRITICAL_REFLECTION = "critical_reflection"  # 给高置信度结论 + "挑战此结论" prompt
    HYPOTHESIS_GENERATION = "hypothesis_generation"  # 给疑问节点 + "生成可能的假设" prompt


# ═══════════════════════════════════════════════════════════════
# 默认配置
# ═══════════════════════════════════════════════════════════════

# 策略权重默认值：结构洞和惊奇追踪略高，概念融合和心智游走略低
_DEFAULT_STRATEGY_WEIGHTS: dict[GrowthStrategy, float] = {
    GrowthStrategy.STRUCTURAL_HOLE: 0.30,
    GrowthStrategy.SURPRISE_TRACKING: 0.30,
    GrowthStrategy.CONCEPTUAL_BLENDING: 0.20,
    GrowthStrategy.MIND_WANDERING: 0.20,
}

# 生成模式权重默认值
_DEFAULT_MODE_WEIGHTS: dict[GenerationMode, float] = {
    GenerationMode.FREE_ASSOCIATION: 0.35,
    GenerationMode.ANALOGY_MAPPING: 0.20,
    GenerationMode.CRITICAL_REFLECTION: 0.25,
    GenerationMode.HYPOTHESIS_GENERATION: 0.20,
}

# 生成模式到边类型的映射
_MODE_TO_EDGE_TYPE: dict[GenerationMode, EdgeType] = {
    GenerationMode.FREE_ASSOCIATION: EdgeType.BRANCH,
    GenerationMode.ANALOGY_MAPPING: EdgeType.ANALOGIZES,
    GenerationMode.CRITICAL_REFLECTION: EdgeType.CONTRADICT,
    GenerationMode.HYPOTHESIS_GENERATION: EdgeType.QUESTIONS,
}

# 生成模式到节点类型的映射
_MODE_TO_NODE_TYPE: dict[GenerationMode, NodeType] = {
    GenerationMode.FREE_ASSOCIATION: NodeType.INSIGHT,
    GenerationMode.ANALOGY_MAPPING: NodeType.ANALOGY,
    GenerationMode.CRITICAL_REFLECTION: NodeType.CONTRADICTION,
    GenerationMode.HYPOTHESIS_GENERATION: NodeType.QUESTION,
}


# ═══════════════════════════════════════════════════════════════
# SpontaneousGrowthStrategy
# ═══════════════════════════════════════════════════════════════


class SpontaneousGrowthStrategy:
    """GoT 自发增长策略

    当推理节点池空闲时，通过多种策略从图中选择锚点，并使用不同生成模式
    调用 LLM 产生新的思维节点。采用 tick-based rhythm 实现轻量/重量
    tick 交替，避免频繁调用 LLM。

    设计灵感来自大脑默认模式网络 (DMN) 和 Sakana AI CTM (2025) 的
    tick-based 架构：大部分 tick 只做轻量的激活传播和权重更新，
    偶尔执行重量 tick 调用 LLM 生成新节点。

    Attributes:
        _cognition: LLM 推理能力接口
        _hard_memory: 记忆检索接口（可选，用于自由联想模式的后备素材）
        _graph: GoTGraph 推理图实例（可选，无图时退化为旧 DMN 行为）
        _min_pool_size: 触发生成的最小节点池阈值
        _strategy_weights: 各锚点选择策略的权重
        _mode_weights: 各生成模式的权重
        _lightweight_tick_ratio: 轻量 tick 与重量 tick 的比率
        _default_temperature: LLM 默认温度
        _tick_counter: tick 计数器，用于决定当前是轻量还是重量 tick
        _topics: 可选的关注主题列表
        _last_generated: 上一轮生成的思维节点列表
    """

    def __init__(
        self,
        cognition,
        hard_memory=None,
        graph: Optional[GoTGraph] = None,
        min_pool_size: int = 3,
        config: Optional[dict] = None,
        embed_fn=None,
    ):
        """初始化自发增长策略

        Args:
            cognition: LLM 推理能力接口，需提供 infer() 方法
            hard_memory: 记忆检索接口，提供思考素材（可选）
            graph: GoTGraph 推理图实例（可选，无图时退化为旧 DMN 行为）
            min_pool_size: 触发 DMN 的最小节点池阈值
            config: 可选配置字典，支持以下键：
                - strategy_weights: dict[GrowthStrategy, float] 策略权重
                - mode_weights: dict[GenerationMode, float] 模式权重
                - lightweight_tick_ratio: int 轻量/重量 tick 比率（默认 5）
                - default_temperature: float 默认温度（默认 0.9）
            embed_fn: 异步 embedding 函数 (text → list[float])，用于惊奇度计算
        """
        config = config or {}

        self._cognition = cognition
        self._hard_memory = hard_memory
        self._graph = graph
        self._min_pool_size = min_pool_size
        self._embed_fn = embed_fn

        # 策略和模式权重
        self._strategy_weights: dict[GrowthStrategy, float] = config.get(
            "strategy_weights", dict(_DEFAULT_STRATEGY_WEIGHTS)
        )
        self._mode_weights: dict[GenerationMode, float] = config.get(
            "mode_weights", dict(_DEFAULT_MODE_WEIGHTS)
        )

        # Tick-based rhythm 参数
        self._lightweight_tick_ratio: int = config.get(
            "lightweight_tick_ratio", 5
        )
        self._default_temperature: float = config.get(
            "default_temperature", 0.9
        )

        # 内部状态
        self._tick_counter: int = 0
        self._topics: list[str] = []
        self._last_generated: list[GoTNode] = []

    # ═══════════════════════════════════════════════════════════
    # 向后兼容接口
    # ═══════════════════════════════════════════════════════════

    def should_generate(self, pool_size: int) -> bool:
        """Check if spontaneous generation should be triggered.

        Delegates to the strategy: triggers when pool size is below threshold.
        """
        return pool_size < self._min_pool_size

    def set_generation_topics(self, topics: list[str]) -> None:
        """Set focus topics for spontaneous generation.

        These topics will be used as query terms for memory retrieval
        and as hints in the generation prompt.

        Args:
            topics: List of topic strings to focus on.
        """
        self._topics = list(topics)

    def get_last_generated(self) -> list[GoTNode]:
        """Return the list of nodes generated in the last grow/generate call."""
        return list(self._last_generated)

    async def generate(self, max_nodes: int = 2) -> list[GoTNode]:
        """Backward-compatible entry point. Delegates to grow().

        Args:
            max_nodes: Maximum number of nodes to generate.

        Returns:
            List of newly generated GoTNode instances.
        """
        return await self.grow(max_nodes=max_nodes)

    # ═══════════════════════════════════════════════════════════
    # 核心公共方法
    # ═══════════════════════════════════════════════════════════

    async def grow(self, max_nodes: int = 2) -> list[GoTNode]:
        """Main entry point for spontaneous growth.

        Selects an anchor and generation strategy, then calls LLM to produce
        new nodes. If no graph is available, falls back to legacy DMN behavior
        (memory-based free association).

        Args:
            max_nodes: Maximum number of nodes to generate.

        Returns:
            List of newly generated GoTNode instances.
        """
        logger.info("spontaneous_grow_start", max_nodes=max_nodes)

        # 无图时退化为旧 DMN 行为
        if self._graph is None or self._graph.node_count == 0:
            return await self._grow_from_memory_fallback(max_nodes)

        # 选择策略和锚点
        strategy = self.select_strategy()
        anchor = self.select_anchor(strategy)

        if anchor is None:
            logger.info("spontaneous_grow_no_anchor", strategy=strategy.value)
            # 无锚点时也退化为记忆后备
            return await self._grow_from_memory_fallback(max_nodes)

        # 选择生成模式
        mode = self.select_mode(anchor, strategy)

        # 根据模式生成节点
        nodes = await self._generate_by_mode(anchor, mode, strategy, max_nodes)

        # 将新节点和边添加到图中
        for node in nodes:
            self._graph.add_node(node)
            self._add_edges_for_node(anchor, node, mode)

        # 使用 embedding 余弦距离计算实际惊奇度 + 去重
        if self._embed_fn is not None:
            deduped_nodes = []
            for node in nodes:
                try:
                    node.surprise = await self._graph.compute_node_surprise(
                        node, self._embed_fn,
                    )
                    # 冗余检测：惊奇度过低
                    if node.surprise < 0.12:
                        node.mark_pruned(f"dmn_redundant (surprise={node.surprise:.2f})")
                        continue
                except Exception:
                    pass  # embedding 失败时保留节点
                deduped_nodes.append(node)
            nodes = deduped_nodes

        self._last_generated = nodes
        logger.info(
            "spontaneous_grow_complete",
            node_count=len(nodes),
            strategy=strategy.value,
            mode=mode.value,
        )
        return nodes

    def tick(self) -> list[GoTNode]:
        """Execute one tick cycle.

        Lightweight ticks: only do activation propagation, update weights,
        detect structural holes (no LLM call).
        Heavyweight ticks: select anchor + call LLM to generate new nodes.

        Note: heavyweight ticks are async (call grow()), so this method
        returns an empty list for lightweight ticks and must be awaited
        for heavyweight ticks. Use tick_async() for the async version.

        Returns:
            Empty list for lightweight ticks (no new nodes).
        """
        self._tick_counter += 1

        # 判断当前是轻量还是重量 tick
        is_heavyweight = (
            self._tick_counter % (self._lightweight_tick_ratio + 1) == 0
        )

        if is_heavyweight:
            # 重量 tick：重置计数器，返回标记让调用者知道需要异步执行
            logger.debug(
                "spontaneous_tick_heavyweight",
                tick_counter=self._tick_counter,
            )
            # 重量 tick 需要异步调用 grow()，这里返回空列表
            # 调用者应使用 tick_async()
            return []
        else:
            # 轻量 tick：激活传播 + 权重更新 + 结构洞检测
            self._lightweight_tick()
            return []

    async def tick_async(self) -> list[GoTNode]:
        """Execute one tick cycle (async version).

        Lightweight ticks: activation propagation, weight updates, structural
        hole detection (no LLM call).
        Heavyweight ticks: select anchor + call LLM to generate new nodes.

        Returns:
            List of newly generated GoTNode instances (empty for lightweight ticks).
        """
        self._tick_counter += 1

        is_heavyweight = (
            self._tick_counter % (self._lightweight_tick_ratio + 1) == 0
        )

        if is_heavyweight:
            logger.debug(
                "spontaneous_tick_heavyweight",
                tick_counter=self._tick_counter,
            )
            return await self.grow()
        else:
            self._lightweight_tick()
            return []

    def select_strategy(self) -> GrowthStrategy:
        """Select an anchor selection strategy using weighted random.

        Returns:
            The selected GrowthStrategy.
        """
        strategies = list(self._strategy_weights.keys())
        weights = [self._strategy_weights[s] for s in strategies]
        return random.choices(strategies, weights=weights, k=1)[0]

    def select_anchor(
        self, strategy: Optional[GrowthStrategy] = None
    ) -> Union[GoTNode, tuple[GoTNode, GoTNode], None]:
        """Select anchor node(s) based on the given strategy.

        Args:
            strategy: The strategy to use. If None, selects one via select_strategy().

        Returns:
            A single GoTNode for most strategies, a tuple of two GoTNodes for
            CONCEPTUAL_BLENDING, or None if no suitable anchor is found.
        """
        if strategy is None:
            strategy = self.select_strategy()

        if self._graph is None or self._graph.node_count == 0:
            return None

        anchor_map = {
            GrowthStrategy.STRUCTURAL_HOLE: self._structural_hole_anchor,
            GrowthStrategy.SURPRISE_TRACKING: self._surprise_tracking_anchor,
            GrowthStrategy.CONCEPTUAL_BLENDING: self._conceptual_blending_anchor,
            GrowthStrategy.MIND_WANDERING: self._mind_wandering_anchor,
        }

        handler = anchor_map.get(strategy)
        if handler is None:
            return None

        anchor = handler()
        if anchor is not None:
            logger.debug(
                "spontaneous_anchor_selected",
                strategy=strategy.value,
                anchor_type=type(anchor).__name__,
            )
        return anchor

    def select_mode(
        self,
        anchor: Union[GoTNode, tuple[GoTNode, GoTNode]],
        strategy: Optional[GrowthStrategy] = None,
    ) -> GenerationMode:
        """Select a generation mode based on anchor type and context.

        Uses heuristics to map strategy to preferred mode, with weighted
        random fallback.

        Args:
            anchor: The selected anchor (single node or pair).
            strategy: The strategy that selected the anchor.

        Returns:
            The selected GenerationMode.
        """
        # 策略到偏好的生成模式映射
        strategy_mode_preference: dict[GrowthStrategy, GenerationMode] = {
            GrowthStrategy.STRUCTURAL_HOLE: GenerationMode.FREE_ASSOCIATION,
            GrowthStrategy.SURPRISE_TRACKING: GenerationMode.CRITICAL_REFLECTION,
            GrowthStrategy.CONCEPTUAL_BLENDING: GenerationMode.ANALOGY_MAPPING,
            GrowthStrategy.MIND_WANDERING: GenerationMode.FREE_ASSOCIATION,
        }

        # 如果锚点是节点对，优先使用类比映射
        if isinstance(anchor, tuple):
            return GenerationMode.ANALOGY_MAPPING

        # 如果锚点是疑问节点，优先使用假设生成
        if isinstance(anchor, GoTNode) and anchor.type == NodeType.QUESTION:
            return GenerationMode.HYPOTHESIS_GENERATION

        # 如果锚点是高置信度结论，优先使用批判反思
        if isinstance(anchor, GoTNode) and anchor.type == NodeType.CONCLUSION and anchor.confidence > 0.7:
            return GenerationMode.CRITICAL_REFLECTION

        # 根据策略偏好选择，但有一定概率随机选择其他模式
        if strategy and random.random() < 0.7:
            return strategy_mode_preference.get(strategy, GenerationMode.FREE_ASSOCIATION)

        # 加权随机选择
        modes = list(self._mode_weights.keys())
        weights = [self._mode_weights[m] for m in modes]
        return random.choices(modes, weights=weights, k=1)[0]

    # ═══════════════════════════════════════════════════════════
    # 锚点选择策略实现
    # ═══════════════════════════════════════════════════════════

    def _structural_hole_anchor(self) -> Optional[GoTNode]:
        """选择结构洞分数最高的活跃节点（填补图中的结构洞）

        结构洞由 _update_structural_hole_scores() 计算：
        (1 - PageRank) / (1 + degree*0.15) * (1 - neighbor_redundancy)。
        低 PageRank + 邻居间稀疏连接 = 认知图的知识缺口。

        Returns:
            结构洞分数最高的活跃节点，或 None
        """
        active_nodes = self._graph.get_active_nodes()
        if not active_nodes:
            return None

        # 确保分数已计算（通常已在 tick 中调用）
        # 按 structural_hole_score 降序排序
        active_nodes.sort(
            key=lambda n: n.metadata.get("structural_hole_score", 0.0),
            reverse=True,
        )
        # 在前 3 个中随机选择，避免总是选同一个
        candidates = active_nodes[: min(3, len(active_nodes))]
        return random.choice(candidates)

    def _surprise_tracking_anchor(self) -> Optional[GoTNode]:
        """选择惊奇度最高的活跃节点

        参考 Buehler (2025)：~12% 的"惊奇边"驱动了持续创新。
        高惊奇度的节点意味着其内容或关联出乎意料，值得进一步探索。

        Returns:
            惊奇度最高的活跃节点，或 None（图无活跃节点时）
        """
        active_nodes = self._graph.get_active_nodes()
        if not active_nodes:
            return None

        # 过滤出有非零惊奇度的节点
        surprising = [n for n in active_nodes if n.surprise > 0.0]
        if not surprising:
            # 无惊奇节点时退化为结构洞策略
            return self._structural_hole_anchor()

        # 按惊奇度降序排序
        surprising.sort(key=lambda n: n.surprise, reverse=True)
        # 在惊奇度最高的前 3 个中随机选择
        candidates = surprising[: min(3, len(surprising))]
        return random.choice(candidates)

    def _conceptual_blending_anchor(
        self,
    ) -> Optional[tuple[GoTNode, GoTNode]]:
        """选择语义距离中等的节点对用于概念融合

        参考 Fauconnier & Turner 概念融合理论和 CHIMERA (2025)：
        语义距离太近的节点融合没有新意，太远的节点融合缺乏基础，
        中等距离的节点对最有可能产生有价值的创意融合。

        语义距离通过图距离（最短路径长度）近似衡量：
        - 距离 1: 直接相连（太近）
        - 距离 2-3: 中等距离（最佳融合候选）
        - 距离 4+: 太远

        Returns:
            语义距离中等的节点对，或 None（无合适节点对时）
        """
        active_nodes = self._graph.get_active_nodes()
        if len(active_nodes) < 2:
            return None

        # 采样候选节点对（避免 O(n²) 全量计算）
        sample_size = min(20, len(active_nodes))
        sampled = random.sample(active_nodes, sample_size)

        best_pair: Optional[tuple[GoTNode, GoTNode]] = None
        best_score = -1.0

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                a, b = sampled[i], sampled[j]
                dist = self._graph_distance(a.node_id, b.node_id)

                # 距离 2-3 最佳，距离 1 太近，距离 > 4 太远
                if dist == 1:
                    score = 0.2  # 太近，低分
                elif dist in (2, 3):
                    score = 1.0  # 最佳距离
                elif dist == 4:
                    score = 0.5  # 稍远
                else:
                    score = 0.1  # 太远或不可达

                # 加入惊奇度加成
                score += (a.surprise + b.surprise) * 0.2

                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

        return best_pair

    def _mind_wandering_anchor(self) -> Optional[GoTNode]:
        """从随机节点出发沿边随机游走 2-3 步

        模拟心智游走 (mind wandering)：从图中的一个随机节点出发，
        沿边（不区分方向）随机游走 2-3 步，到达的节点即为锚点。
        这种方式倾向于到达图的"边缘"区域，发现被忽视的节点。

        Returns:
            随机游走到达的节点，或 None（图无活跃节点时）
        """
        active_nodes = self._graph.get_active_nodes()
        if not active_nodes:
            return None

        # 随机选择起始节点
        current = random.choice(active_nodes)
        steps = random.randint(2, 3)

        for _ in range(steps):
            # 获取所有相邻节点（父节点 + 子节点）
            neighbors = []
            for parent in self._graph.get_parents(current.node_id):
                if parent.is_active:
                    neighbors.append(parent)
            for child in self._graph.get_children(current.node_id):
                if child.is_active:
                    neighbors.append(child)

            if not neighbors:
                # 死胡同，停在当前节点
                break

            current = random.choice(neighbors)

        return current

    # ═══════════════════════════════════════════════════════════
    # 生成模式实现
    # ═══════════════════════════════════════════════════════════

    async def _generate_by_mode(
        self,
        anchor: Union[GoTNode, tuple[GoTNode, GoTNode]],
        mode: GenerationMode,
        strategy: GrowthStrategy,
        max_nodes: int,
    ) -> list[GoTNode]:
        """根据生成模式调用对应的生成方法

        Args:
            anchor: 锚点（单节点或节点对）
            mode: 生成模式
            strategy: 使用的锚点选择策略
            max_nodes: 最大生成节点数

        Returns:
            新生成的 GoTNode 列表
        """
        mode_handlers = {
            GenerationMode.FREE_ASSOCIATION: self._generate_free_association,
            GenerationMode.ANALOGY_MAPPING: self._generate_analogy_mapping,
            GenerationMode.CRITICAL_REFLECTION: self._generate_critical_reflection,
            GenerationMode.HYPOTHESIS_GENERATION: self._generate_hypothesis,
        }

        handler = mode_handlers.get(mode)
        if handler is None:
            logger.warning(
                "spontaneous_unknown_mode",
                mode=mode.value,
                fallback="free_association",
            )
            return await self._generate_free_association(anchor, max_nodes)

        return await handler(anchor, max_nodes)

    async def _generate_free_association(
        self,
        anchor: Union[GoTNode, tuple[GoTNode, GoTNode]],
        max_nodes: int,
    ) -> list[GoTNode]:
        """自由联想生成：给锚点内容 + "自由联想" prompt

        这是最接近旧 DMN 行为的模式。从锚点内容出发，让 LLM 自由
        联想产生新的思想节点。

        Args:
            anchor: 锚点节点（忽略节点对，只使用第一个）
            max_nodes: 最大生成节点数

        Returns:
            新生成的 GoTNode 列表
        """
        # 处理节点对的情况
        if isinstance(anchor, tuple):
            anchor = anchor[0]

        # 先尝试从图中获取上下文
        context_text = self._build_anchor_context(anchor)

        # 如果有记忆接口，也获取一些记忆作为补充素材
        memories_text = ""
        if self._hard_memory is not None:
            memories = await self._fetch_random_memories()
            if memories:
                mem_lines = [f"[{i}] {m.get('content', '')}" for i, m in enumerate(memories[:5])]
                memories_text = "\n\nAdditional memory fragments:\n" + "\n".join(mem_lines)

        prompt = (
            "You are the spontaneous thought generator of a thinking AI. "
            "Your role is to freely associate from the given anchor thought, "
            "generating creative and unexpected connections.\n\n"
            f"Anchor thought:\n{context_text}\n"
            f"{memories_text}\n\n"
            f"Generate exactly {max_nodes} spontaneous thought(s) by freely "
            "associating from the anchor. Explore unexpected connections, "
            "creative leaps, and novel perspectives.\n\n"
            "Return your response as a JSON object with a 'nodes' key "
            "containing an array of objects, each with a 'content' key "
            "for the thought text.\n"
            'Example: {{"nodes": [{{"content": "spontaneous thought here"}}]}}\n'
            "Return ONLY the JSON object, no other text."
        )

        return await self._call_llm_and_parse(
            prompt, max_nodes, GenerationMode.FREE_ASSOCIATION
        )

    async def _generate_analogy_mapping(
        self,
        anchor: Union[GoTNode, tuple[GoTNode, GoTNode]],
        max_nodes: int,
    ) -> list[GoTNode]:
        """类比映射生成：给两个节点的结构描述 + "寻找结构相似性" prompt

        参考 Gentner (1983) 结构映射理论：类比的核心不是表面相似性，
        而是关系结构的对应。给定两个节点，让 LLM 寻找它们之间
        深层的结构相似性。

        Args:
            anchor: 节点对（必须为 tuple）
            max_nodes: 最大生成节点数

        Returns:
            新生成的 GoTNode 列表
        """
        # 确保有节点对
        if isinstance(anchor, GoTNode):
            # 单节点时，尝试从图中找一个邻居组成对
            neighbors = self._graph.get_children(anchor.node_id) + self._graph.get_parents(anchor.node_id)
            active_neighbors = [n for n in neighbors if n.is_active]
            if not active_neighbors:
                # 无邻居，退化为自由联想
                return await self._generate_free_association(anchor, max_nodes)
            partner = random.choice(active_neighbors)
            node_a, node_b = anchor, partner
        else:
            node_a, node_b = anchor

        # 构建结构描述
        desc_a = self._build_structural_description(node_a)
        desc_b = self._build_structural_description(node_b)

        prompt = (
            "You are the analogy engine of a thinking AI. "
            "Your role is to find deep structural similarities between "
            "two seemingly different concepts.\n\n"
            f"Concept A:\n{desc_a}\n\n"
            f"Concept B:\n{desc_b}\n\n"
            "Find the structural similarity between these two concepts. "
            "Focus on relational patterns, not surface features. "
            "What abstract structure do they share?\n\n"
            f"Generate exactly {max_nodes} analogy node(s) that articulate "
            "the structural mapping between these concepts.\n\n"
            "Return your response as a JSON object with a 'nodes' key "
            "containing an array of objects, each with a 'content' key.\n"
            'Example: {{"nodes": [{{"content": "structural analogy here"}}]}}\n'
            "Return ONLY the JSON object, no other text."
        )

        return await self._call_llm_and_parse(
            prompt, max_nodes, GenerationMode.ANALOGY_MAPPING
        )

    async def _generate_critical_reflection(
        self,
        anchor: Union[GoTNode, tuple[GoTNode, GoTNode]],
        max_nodes: int,
    ) -> list[GoTNode]:
        """批判反思生成：给高置信度结论 + "挑战此结论" prompt

        参考 Popper (1963) 证伪主义：科学进步的核心不是证实，
        而是证伪。对高置信度的结论进行批判性审视，寻找潜在的
        反例、隐藏假设或逻辑漏洞。

        Args:
            anchor: 锚点节点（忽略节点对，只使用第一个）
            max_nodes: 最大生成节点数

        Returns:
            新生成的 GoTNode 列表
        """
        if isinstance(anchor, tuple):
            anchor = anchor[0]

        context_text = self._build_anchor_context(anchor)

        prompt = (
            "You are the critical reflection engine of a thinking AI. "
            "Your role is to challenge conclusions and find hidden "
            "assumptions, logical gaps, or potential counterexamples.\n\n"
            f"Conclusion to challenge:\n{context_text}\n"
            f"Confidence level: {anchor.confidence:.0%}\n\n"
            "Critically examine this conclusion. Consider:\n"
            "- What assumptions does it rely on?\n"
            "- What evidence would contradict it?\n"
            "- What alternative explanations exist?\n"
            "- What are the edge cases or exceptions?\n\n"
            f"Generate exactly {max_nodes} critical reflection node(s) "
            "that challenge or refine this conclusion.\n\n"
            "Return your response as a JSON object with a 'nodes' key "
            "containing an array of objects, each with a 'content' key.\n"
            'Example: {{"nodes": [{{"content": "critical reflection here"}}]}}\n'
            "Return ONLY the JSON object, no other text."
        )

        return await self._call_llm_and_parse(
            prompt, max_nodes, GenerationMode.CRITICAL_REFLECTION
        )

    async def _generate_hypothesis(
        self,
        anchor: Union[GoTNode, tuple[GoTNode, GoTNode]],
        max_nodes: int,
    ) -> list[GoTNode]:
        """假设生成：给疑问节点 + "生成可能的假设" prompt

        对疑问节点生成可能的假设性回答。这些假设不是确定的结论，
        而是需要后续验证的候选解释。

        Args:
            anchor: 锚点节点（忽略节点对，只使用第一个）
            max_nodes: 最大生成节点数

        Returns:
            新生成的 GoTNode 列表
        """
        if isinstance(anchor, tuple):
            anchor = anchor[0]

        context_text = self._build_anchor_context(anchor)

        prompt = (
            "You are the hypothesis generator of a thinking AI. "
            "Your role is to generate possible hypotheses for open "
            "questions, exploring multiple plausible explanations.\n\n"
            f"Question to address:\n{context_text}\n\n"
            "Generate diverse, plausible hypotheses that could answer "
            "this question. Each hypothesis should be:\n"
            "- Internally consistent\n"
            "- Potentially testable or verifiable\n"
            "- Distinct from other hypotheses\n\n"
            f"Generate exactly {max_nodes} hypothesis node(s).\n\n"
            "Return your response as a JSON object with a 'nodes' key "
            "containing an array of objects, each with a 'content' key.\n"
            'Example: {{"nodes": [{{"content": "hypothesis here"}}]}}\n'
            "Return ONLY the JSON object, no other text."
        )

        return await self._call_llm_and_parse(
            prompt, max_nodes, GenerationMode.HYPOTHESIS_GENERATION
        )

    # ═══════════════════════════════════════════════════════════
    # 轻量 tick 实现
    # ═══════════════════════════════════════════════════════════

    def _lightweight_tick(self) -> None:
        """执行轻量 tick：激活传播 + 权重更新 + 结构洞检测

        不调用 LLM，只做图上的本地计算：
        1. 扩散激活传播（沿边传播能量）
        2. 全局激活衰减
        3. 检测结构洞（更新元数据，供后续重量 tick 使用）
        """
        if self._graph is None:
            return

        # 扩散激活传播
        self._graph.spread_activation(lateral_inhibition=0.3)

        # 全局衰减
        self._graph.decay_all()

        # 结构洞检测：计算并缓存每个活跃节点的结构洞分数
        self._update_structural_hole_scores()

        logger.debug(
            "spontaneous_lightweight_tick",
            tick_counter=self._tick_counter,
            active_nodes=self._graph.active_count,
        )

    def _update_structural_hole_scores(self) -> None:
        """更新图中活跃节点的结构洞分数

        混合 PageRank + 约束代理指标：
        - 运行简化无向 PageRank（15 轮迭代）
        - 高 PageRank 节点 = 枢纽，已是认知中心，非结构洞
        - 低 PageRank + 低度数 + 邻居间稀疏连接 = 真正的结构洞
        - 最终分数 = (1 - PageRank) / (1 + degree * 0.15) * (1 - neighbor_redundancy)

        参考 Burt (2004) 约束指标和 Brin & Page (1998) PageRank。
        """
        if self._graph is None:
            return

        active = self._graph.get_active_nodes()
        if not active:
            return

        node_ids = [n.node_id for n in active]
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # ── 1. 构建无向邻接矩阵 ──
        # 把每一个 (parent_id, child_id) 边的两个方向都加入集合
        adjacency: list[set[int]] = [set() for _ in range(n)]
        for node in active:
            src_idx = id_to_idx[node.node_id]
            for cid in node.children_ids:
                if cid in id_to_idx:
                    dst_idx = id_to_idx[cid]
                    adjacency[src_idx].add(dst_idx)
                    adjacency[dst_idx].add(src_idx)
            for pid in node.parent_ids:
                if pid in id_to_idx:
                    dst_idx = id_to_idx[pid]
                    adjacency[src_idx].add(dst_idx)
                    adjacency[dst_idx].add(src_idx)

        # ── 2. 简化 PageRank（无向图，阻尼 0.85，15 轮） ──
        pr = [1.0 / n] * n
        DAMPING = 0.85

        for _ in range(15):
            new_pr = [(1.0 - DAMPING) / n] * n
            for i in range(n):
                neighbors = adjacency[i]
                if neighbors:
                    out_degree = len(neighbors)
                    share = DAMPING * pr[i] / out_degree
                    for j in neighbors:
                        new_pr[j] += share
            pr = new_pr

        # ── 3. 邻居冗余度代理 ──
        # 邻居之间边数越多 → 冗余度越高 → 说明处于紧密社区中心，非结构洞
        neighbor_redundancy: dict[str, float] = {}
        for node in active:
            nb_set = (set(node.parent_ids) | set(node.children_ids)) & set(node_ids)
            nb_list = list(nb_set)
            k = len(nb_list)
            if k < 2:
                neighbor_redundancy[node.node_id] = 0.0
                continue
            edge_count = 0
            pair_count = 0
            for a in range(k):
                for b in range(a + 1, k):
                    pair_count += 1
                    na, nb = nb_list[a], nb_list[b]
                    if (self._graph.get_edge(na, nb) or self._graph.get_edge(nb, na)):
                        edge_count += 1
            neighbor_redundancy[node.node_id] = edge_count / max(pair_count, 1)

        # ── 4. 综合：结构洞分数 ──
        for i, node in enumerate(active):
            degree = len(node.parent_ids) + len(node.children_ids)
            nr = neighbor_redundancy.get(node.node_id, 1.0)
            # (1 - PageRank): 低枢纽性
            # / (1 + degree * 0.15): 度惩罚（轻微，避免孤立节点高分）
            # * (1 - nr): 邻居冗余度惩罚（高冗余 → 已在社区中心）
            hole_score = (1.0 - pr[i]) / (1.0 + degree * 0.15) * (1.0 - nr)
            hole_score = max(0.0, min(1.0, hole_score))
            node.metadata["structural_hole_score"] = round(hole_score, 4)

    # ═══════════════════════════════════════════════════════════
    # 图辅助方法
    # ═══════════════════════════════════════════════════════════

    def _graph_distance(self, node_a_id: str, node_b_id: str) -> int:
        """计算两个节点之间的最短路径长度（BFS）

        Args:
            node_a_id: 起始节点 ID
            node_b_id: 目标节点 ID

        Returns:
            最短路径长度，不可达时返回 999
        """
        if self._graph is None:
            return 999

        if node_a_id == node_b_id:
            return 0

        # BFS 搜索（无向图：同时沿父节点和子节点方向）
        visited = {node_a_id}
        queue = [node_a_id]
        distance = 0

        while queue:
            distance += 1
            next_queue = []
            for nid in queue:
                node = self._graph.get_node(nid)
                if node is None:
                    continue
                # 收集所有邻居（父+子）
                neighbor_ids = list(node.parent_ids) + list(node.children_ids)
                for neighbor_id in neighbor_ids:
                    if neighbor_id == node_b_id:
                        return distance
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_queue.append(neighbor_id)
            queue = next_queue

        return 999  # 不可达

    def _build_anchor_context(self, anchor: GoTNode) -> str:
        """构建锚点节点的上下文描述

        包括节点内容、类型、置信度以及其邻居的摘要。

        Args:
            anchor: 锚点节点

        Returns:
            格式化的上下文文本
        """
        lines = [
            f"[{anchor.type.value}] {anchor.content}",
            f"Confidence: {anchor.confidence:.0%}",
        ]

        if anchor.surprise > 0:
            lines.append(f"Surprise: {anchor.surprise:.2f}")

        # 添加邻居摘要
        if self._graph is not None:
            children = self._graph.get_children(anchor.node_id)
            parents = self._graph.get_parents(anchor.node_id)
            if children:
                child_summaries = [f"  → [{c.type.value}] {c.content}" for c in children[:3]]
                lines.append("Related thoughts (children):")
                lines.extend(child_summaries)
            if parents:
                parent_summaries = [f"  ← [{p.type.value}] {p.content}" for p in parents[:3]]
                lines.append("Related thoughts (parents):")
                lines.extend(parent_summaries)

        return "\n".join(lines)

    def _build_structural_description(self, node: GoTNode) -> str:
        """构建节点的结构描述（用于类比映射）

        包括节点内容、类型、邻居关系模式等结构信息，
        而非仅仅是内容文本。

        Args:
            node: 目标节点

        Returns:
            格式化的结构描述
        """
        lines = [
            f"Type: {node.type.value}",
            f"Content: {node.content}",
            f"Confidence: {node.confidence:.0%}",
        ]

        if self._graph is not None:
            children = self._graph.get_children(node.node_id)
            parents = self._graph.get_parents(node.node_id)
            lines.append(f"Outgoing connections: {len(children)}")
            lines.append(f"Incoming connections: {len(parents)}")

            # 描述邻居的类型分布
            neighbor_types = {}
            for n in children + parents:
                t = n.type.value
                neighbor_types[t] = neighbor_types.get(t, 0) + 1
            if neighbor_types:
                type_dist = ", ".join(f"{t}:{c}" for t, c in neighbor_types.items())
                lines.append(f"Neighbor type distribution: {type_dist}")

            # 描述边的类型分布
            edge_types = {}
            for child in children:
                edge = self._graph.get_edge(node.node_id, child.node_id)
                if edge:
                    et = edge.type.value
                    edge_types[et] = edge_types.get(et, 0) + 1
            for parent in parents:
                edge = self._graph.get_edge(parent.node_id, node.node_id)
                if edge:
                    et = edge.type.value
                    edge_types[et] = edge_types.get(et, 0) + 1
            if edge_types:
                edge_dist = ", ".join(f"{t}:{c}" for t, c in edge_types.items())
                lines.append(f"Edge type distribution: {edge_dist}")

        return "\n".join(lines)

    def _add_edges_for_node(
        self,
        anchor: Union[GoTNode, tuple[GoTNode, GoTNode]],
        new_node: GoTNode,
        mode: GenerationMode,
    ) -> None:
        """为新节点添加到锚点的边

        Args:
            anchor: 锚点（单节点或节点对）
            new_node: 新生成的节点
            mode: 生成模式
        """
        if self._graph is None:
            return

        edge_type = _MODE_TO_EDGE_TYPE.get(mode, EdgeType.BRANCH)

        if isinstance(anchor, tuple):
            # 节点对：两个锚点都连接到新节点
            for a in anchor:
                edge = GoTEdge(
                    source_id=a.node_id,
                    target_id=new_node.node_id,
                    type=edge_type,
                    weight=round(random.uniform(0.6, 0.9), 2),
                )
                try:
                    self._graph.add_edge(edge)
                except ValueError:
                    logger.debug(
                        "spontaneous_edge_add_failed",
                        source=a.node_id,
                        target=new_node.node_id,
                    )
        else:
            # 单锚点
            edge = GoTEdge(
                source_id=anchor.node_id,
                target_id=new_node.node_id,
                type=edge_type,
                weight=round(random.uniform(0.6, 0.9), 2),
            )
            try:
                self._graph.add_edge(edge)
            except ValueError:
                logger.debug(
                    "spontaneous_edge_add_failed",
                    source=anchor.node_id,
                    target=new_node.node_id,
                )

    # ═══════════════════════════════════════════════════════════
    # LLM 调用与解析
    # ═══════════════════════════════════════════════════════════

    async def _call_llm_and_parse(
        self,
        prompt: str,
        max_nodes: int,
        mode: GenerationMode,
    ) -> list[GoTNode]:
        """调用 LLM 并解析返回的 JSON，创建 GoTNode 列表

        # TODO: 未来迁移到两阶段输出格式：
        # 1. 自由形式思考（不限制格式）
        # 2. 使用 extract_tags 解析结构化标签
        # 当前保持 JSON 格式，与 reasoning_loop 的更新保持独立

        Args:
            prompt: LLM prompt
            max_nodes: 最大生成节点数
            mode: 生成模式

        Returns:
            新生成的 GoTNode 列表
        """
        user_input = MultiModalInput()
        user_input.add_text(prompt)

        result = await self._cognition.infer(
            user_input,
            temperature=self._default_temperature,
            max_context_tokens=4096,
        )

        if not result or not result.text:
            logger.warning("spontaneous_empty_llm_response", mode=mode.value)
            return self._parse_nodes("", max_nodes, mode)

        return self._parse_nodes(result.text.strip(), max_nodes, mode)

    def _parse_nodes(
        self, text: str, max_nodes: int, mode: GenerationMode
    ) -> list[GoTNode]:
        """解析 LLM 返回的 JSON，创建 GoTNode 列表

        Args:
            text: LLM 返回的文本
            max_nodes: 最大节点数
            mode: 生成模式（决定节点类型）

        Returns:
            GoTNode 列表
        """
        node_type = _MODE_TO_NODE_TYPE.get(mode, NodeType.INSIGHT)

        try:
            if text:
                sanitized = self._sanitize_llm_output(text)
                data = json.loads(sanitized)
                nodes_data = data.get("nodes", [])
            else:
                nodes_data = []
        except (json.JSONDecodeError, ValueError):
            logger.debug("spontaneous_json_parse_failed", text=text[:200])
            nodes_data = []

        if nodes_data:
            nodes = []
            for nd in nodes_data[:max_nodes]:
                content = nd.get("content", "")
                if content:
                    # 根据生成模式设置不同的置信度范围
                    if mode == GenerationMode.CRITICAL_REFLECTION:
                        confidence = round(random.uniform(0.25, 0.40), 2)
                    elif mode == GenerationMode.HYPOTHESIS_GENERATION:
                        confidence = round(random.uniform(0.30, 0.50), 2)
                    elif mode == GenerationMode.ANALOGY_MAPPING:
                        confidence = round(random.uniform(0.40, 0.60), 2)
                    else:
                        # 自由联想：低置信度，高惊奇度
                        confidence = round(random.uniform(0.35, 0.45), 2)

                    # 惊奇度将在 grow() 中通过 embedding 余弦距离实际计算
                    dmn_node = GoTNode(
                        type=node_type,
                        content=content,
                        confidence=confidence,
                        origin=NodeOrigin.SPONTANEOUS,
                    )
                    dmn_node.inject_activation(random.uniform(0.8, 2.0))  # DMN 自发节点注入激活能
                    nodes.append(dmn_node)

            if nodes:
                return nodes

        return []

    # ═══════════════════════════════════════════════════════════
    # 记忆后备（旧 DMN 行为）
    # ═══════════════════════════════════════════════════════════

    async def _grow_from_memory_fallback(self, max_nodes: int) -> list[GoTNode]:
        """从记忆中获取素材的后备生成模式（旧 DMN 行为）

        当图不可用或无锚点时，退化为旧 DMN 的行为：从记忆中
        随机抓取记忆片段，通过 LLM 生成自发的反思和联想。

        Args:
            max_nodes: 最大生成节点数

        Returns:
            新生成的 GoTNode 列表
        """
        logger.info("spontaneous_grow_memory_fallback")

        if self._hard_memory is None:
            logger.info("spontaneous_no_memory_available")
            self._last_generated = []
            return []

        memories = await self._fetch_random_memories()

        if not memories:
            logger.info("spontaneous_no_memories_fetched")
            self._last_generated = []
            return []

        nodes = await self._generate_from_memories(memories, max_nodes)
        self._last_generated = nodes
        logger.info(
            "spontaneous_memory_fallback_complete",
            node_count=len(nodes),
        )
        return nodes

    async def _fetch_random_memories(self) -> list[dict]:
        """从记忆中随机抓取记忆片段

        使用预设的查询词列表（或自定义主题）从 HardMemory 中
        检索相关记忆，去重后返回。

        Returns:
            记忆字典列表
        """
        query_terms = self._topics if self._topics else _DEFAULT_QUERY_TERMS

        selected = random.sample(query_terms, min(3, len(query_terms)))

        all_results: list[dict] = []
        seen_ids: set[str] = set()

        for term in selected:
            try:
                results = await self._hard_memory.recollect(term, k=5)
                for r in results:
                    # 兼容 dict 和对象两种格式
                    if isinstance(r, dict):
                        rid = r.get("id", "")
                        content = r.get("content", str(r))
                    else:
                        rid = r.id if hasattr(r, "id") else ""
                        content = r.content if hasattr(r, "content") else str(r)
                    if rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        all_results.append({"id": rid, "content": content})
                    elif not rid:
                        # 无 id 的记忆也保留，用内容哈希去重
                        content_key = hashlib.md5(content.encode()).hexdigest()
                        if content_key not in seen_ids:
                            seen_ids.add(content_key)
                            all_results.append({"id": "", "content": content})
            except Exception as e:
                logger.debug(
                    "spontaneous_memory_fetch_term_error",
                    term=term,
                    error=str(e),
                )
                continue

        logger.debug(
            "spontaneous_memories_fetched",
            count=len(all_results),
            terms_used=selected,
        )
        return all_results

    async def _generate_from_memories(
        self, memories: list[dict], max_nodes: int
    ) -> list[GoTNode]:
        """从记忆片段生成自发思维节点（旧 DMN 行为）

        Args:
            memories: 记忆字典列表
            max_nodes: 最大生成节点数

        Returns:
            新生成的 GoTNode 列表
        """
        memory_lines = []
        for i, mem in enumerate(memories[:10]):
            content = mem.get("content", "")
            memory_lines.append(f"[{i}] {content}")

        memories_text = "\n".join(memory_lines)

        topic_hint = ""
        if self._topics:
            topic_hint = (
                f"\nCurrent focus areas: {', '.join(self._topics[:5])}"
            )

        prompt = (
            "You are the Default Mode Network of a thinking AI. "
            "Your role is to spontaneously generate reflections, connections, "
            "and questions based on memories when the mind is idle.\n\n"
            "Available memories:\n"
            f"{memories_text}\n"
            f"{topic_hint}\n\n"
            f"Generate exactly {max_nodes} spontaneous thought node(s). "
            "Each node should be a reflection, creative connection, or "
            "thought-provoking question derived from the memories above. "
            "Be associative and creative - explore unexpected connections "
            "between different memories.\n\n"
            "Return your response as a JSON object with a 'nodes' key "
            "containing an array of objects, each with a 'content' key "
            "for the thought text.\n"
            'Example: {{"nodes": [{{"content": "spontaneous thought here"}}]}}\n'
            "Return ONLY the JSON object, no other text."
        )

        user_input = MultiModalInput()
        user_input.add_text(prompt)

        result = await self._cognition.infer(
            user_input,
            temperature=self._default_temperature,
            max_context_tokens=4096,
        )

        if not result or not result.text:
            logger.warning("spontaneous_empty_llm_response")
            return self._parse_nodes("", max_nodes, GenerationMode.FREE_ASSOCIATION)

        return self._parse_nodes(
            result.text.strip(), max_nodes, GenerationMode.FREE_ASSOCIATION
        )

    def _sanitize_llm_output(self, text: str) -> str:
        """清理 LLM 输出中的非 JSON 内容

        移除 <think> 标签、Markdown 代码块标记等，
        提取最外层的 JSON 对象。

        Args:
            text: 原始 LLM 输出文本

        Returns:
            清理后的文本
        """
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'Thinking Process:.*?(?=\n\{|\n\[|$)', '', text, flags=re.DOTALL)
        text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip())
        text = re.sub(r'\n?\s*```$', '', text.strip())
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start : brace_end + 1]
        return text


# ═══════════════════════════════════════════════════════════════
# 向后兼容别名
# ═══════════════════════════════════════════════════════════════

DMNGenerator = SpontaneousGrowthStrategy
