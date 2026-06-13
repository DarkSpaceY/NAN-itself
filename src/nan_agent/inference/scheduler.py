"""
GoT 调度器 (Scheduler)
-----------------------
从节点池中批量取出推理节点，并行执行推理循环，管理推理吞吐量。
支持基于扩散激活（Spreading Activation）的智能调度。

调度器负责：
1. 从 NodePool 中按优先级 + 激活能量取出节点 → process_batch() / select_next_nodes()
2. 批量并行执行 ReasoningLoop.run() → asyncio.gather
3. 收集结果，将新节点回填到 NodePool → 循环往复
4. 每批次处理后执行扩散激活传播 → graph.spread_activation() + graph.decay_all()
5. 外部任务注入激活能量 → inject_external_task()
6. Tick 驱动的轻量调度循环 → tick()
7. 统计吞吐量指标（节点数、错误数、成功率、处理速率）

调度策略（优先级从高到低）：
- 外部任务节点（CEN 注入）：最高优先级 + 激活能量注入
- 高激活能量节点（扩散激活传播结果）：次高优先级
- 长时间未被处理的节点（last_accessed_at 较旧）：避免饥饿
- 池中默认优先级顺序：兜底策略

参考文献：
- SYNAPSE (2025): 扩散激活在 LLM 智能体记忆中的应用
- Buehler (2025): 自组织临界态与惊奇边
"""

import asyncio
import dataclasses
import time
from datetime import datetime, timezone

from nan_agent.inference.graph import GoTGraph, GoTNode, NodeOrigin, EDGE_DECAY_RATES
from nan_agent.inference.node_pool import NodePool
from nan_agent.inference.reasoning_loop import LoopResult, ReasoningLoop
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class BatchResult:
    """单批次推理结果统计

    Attributes:
        batch_size: 本批次处理的节点数
        new_nodes_count: 产生的新节点数
        pruned_count: 被裁剪的节点数
        action_count: 产生的动作输出数
        errors: 错误信息列表
        elapsed_ms: 批次处理耗时（毫秒）
    """
    batch_size: int
    new_nodes_count: int
    pruned_count: int
    action_count: int
    errors: list[str]
    elapsed_ms: float


@dataclasses.dataclass
class TickResult:
    """单次 tick 调度结果统计

    Attributes:
        nodes_processed: 本次 tick 处理的节点数
        new_nodes: 本次 tick 产生的新节点数
        activation_spread: 是否执行了扩散激活传播
        entropy: 当前图的结构熵
        elapsed_ms: tick 耗时（毫秒）
    """
    nodes_processed: int
    new_nodes: int
    activation_spread: bool
    entropy: float
    elapsed_ms: float


class GoTScheduler:
    """GoT 推理调度器

    管理推理节点队列，批量并行执行推理循环。核心职责：
    - 从 NodePool 获取待处理节点 -> 并行执行推理 -> 收集结果回填
    - 每批次处理后执行扩散激活传播，更新节点激活能量
    - 基于激活能量智能选择下一批待处理节点
    - 支持外部任务注入激活能量
    - Tick 驱动的轻量调度循环
    - 统计吞吐量指标（处理速率、成功率、错误数）
    - 支持单独的节点处理 (process_single) 和周期性任务注册
    """

    # ── 外部任务注入的默认激活能量 ──
    _EXTERNAL_TASK_ACTIVATION: float = 5.0
    # ── 邻居节点注入的激活能量（外部任务注入时扩散给邻居） ──
    _NEIGHBOR_ACTIVATION: float = 2.0
    # ── 激活能量在调度评分中的权重 ──
    _ACTIVATION_WEIGHT: float = 0.4
    # ── 新鲜度（last_accessed_at）在调度评分中的权重 ──
    _RECENCY_WEIGHT: float = 0.2
    # ── 池优先级在调度评分中的权重 ──
    _POOL_PRIORITY_WEIGHT: float = 0.4

    def __init__(
        self,
        cognition,
        graph: GoTGraph,
        pool: NodePool,
        reasoning_loop: ReasoningLoop,
    ):
        self._cognition = cognition
        self.graph = graph
        self.pool = pool
        self._reasoning_loop = reasoning_loop

        # ── 批次统计 ──
        self._total_batches: int = 0
        self._total_nodes_processed: int = 0
        self._total_new_nodes: int = 0
        self._total_action_nodes: int = 0
        self._total_pruned_nodes: int = 0
        self._total_errors: int = 0
        self._total_elapsed_ms: float = 0.0

        # ── Tick 统计 ──
        self._total_ticks: int = 0
        self._total_activation_spreads: int = 0

    # ═══════════════════════════════════════════════════════════
    # 核心调度：基于扩散激活的节点选择
    # ═══════════════════════════════════════════════════════════

    def select_next_nodes(self, batch_size: int = 5) -> list[GoTNode]:
        """Select the best nodes to process next, considering activation energy, recency, and pool priority.

        调度策略（综合评分）：
        1. 外部任务节点（origin=EXTERNAL）获得最高优先级 + 激活能量注入
        2. 高激活能量节点（扩散激活传播结果）获得较高评分
        3. 长时间未被处理的节点（last_accessed_at 较旧）避免饥饿
        4. 池中默认优先级顺序作为兜底

        Args:
            batch_size: 期望选择的节点数量

        Returns:
            按综合评分排序的待处理节点列表
        """
        if self.pool.is_empty():
            return []

        # ── 获取池中所有节点 ──
        all_nodes = list(self.pool.nodes.values())
        if not all_nodes:
            return []

        now = datetime.now(timezone.utc)

        # ── 第一步：对外部任务节点和自发节点注入激活能量 ──
        for node in all_nodes:
            if node.origin == NodeOrigin.EXTERNAL and node.activation < self._EXTERNAL_TASK_ACTIVATION:
                # 外部任务节点获得高激活能量注入
                node.inject_activation(self._EXTERNAL_TASK_ACTIVATION - node.activation)
                logger.debug(
                    "external_task_activation_injected",
                    node_id=node.node_id,
                    activation=node.activation,
                )
            elif node.origin == NodeOrigin.SPONTANEOUS and node.activation < 1.0:
                # 自发节点维持最低激活能，避免陷入零激活循环（微量注入）
                node.inject_activation(0.1)
                logger.debug(
                    "spontaneous_node_activation_boosted",
                    node_id=node.node_id,
                    activation=node.activation,
                )

        # ── 计算每个节点的综合评分 ──
        scored_nodes: list[tuple[GoTNode, float]] = []
        for node in all_nodes:
            score = self._compute_scheduling_score(node, now)
            scored_nodes.append((node, score))

        # ── 按评分降序排列 ──
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # ── 取前 batch_size 个节点 ──
        selected = [node for node, _ in scored_nodes[:batch_size]]

        logger.debug(
            "nodes_selected",
            batch_size=len(selected),
            top_scores=[round(s, 3) for _, s in scored_nodes[:batch_size]],
        )

        return selected

    def _compute_scheduling_score(self, node: GoTNode, now: datetime) -> float:
        """Compute the scheduling score for a single node.

        综合评分 = 激活能量分 × 权重 + 新鲜度分 × 权重 + 池优先级分 × 权重

        Args:
            node: 待评分的节点
            now: 当前时间（用于计算新鲜度）

        Returns:
            综合评分（0.0 ~ 无上限，激活能量可能推高评分）
        """
        # ── 激活能量分：归一化到 0.0-1.0 范围 ──
        # 激活能量上限为 10.0（见 GoTNode.inject_activation）
        activation_score = min(node.activation / 10.0, 1.0)

        # ── 新鲜度分：越久未被访问，分数越高（避免饥饿） ──
        age_seconds = (now - node.last_accessed_at).total_seconds()
        # 使用对数缩放，避免极端值；60秒内为0分，之后逐渐增加
        if age_seconds <= 60.0:
            recency_score = 0.0
        else:
            import math
            recency_score = min(math.log1p(age_seconds - 60.0) / 10.0, 1.0)

        # ── 池优先级分：使用 NodePool 的优先级评分逻辑 ──
        pool_score = self.pool._priority_score(node)
        # 归一化到 0.0-1.0（池优先级范围大约 -100 ~ 1.5）
        pool_score_normalized = max(0.0, min(1.0, (pool_score + 100.0) / 101.5))

        # ── 外部任务节点额外加分 ──
        external_bonus = 0.0
        if node.origin == NodeOrigin.EXTERNAL:
            external_bonus = 1.0

        # ── 结构熵调节：熵过高时偏向收敛，熵过低时偏向发散 ──
        entropy = self.graph.structural_entropy()
        entropy_bonus = 0.0
        if entropy > 0.75:
            # 熵过高：给有子节点的节点加分（适合 merge/prune）
            children = self.graph.get_children(node.node_id)
            if children:
                entropy_bonus = (entropy - 0.75) * 5.0  # 加强收敛倾向
        elif entropy < 0.35:
            # 熵过低：给叶子节点加分（适合 branch 发散）
            children = self.graph.get_children(node.node_id)
            if not children:
                entropy_bonus = (0.35 - entropy) * 5.0  # 加强发散倾向

        # ── 综合评分 ──
        composite = (
            activation_score * self._ACTIVATION_WEIGHT
            + recency_score * self._RECENCY_WEIGHT
            + pool_score_normalized * self._POOL_PRIORITY_WEIGHT
            + external_bonus
            + entropy_bonus
        )

        return composite

    # ═══════════════════════════════════════════════════════════
    # Tick 驱动的轻量调度
    # ═══════════════════════════════════════════════════════════

    async def tick(self, batch_size: int = 5) -> TickResult:
        """Execute one tick of the scheduler: spread activation, select and process a batch.

        Tick 是调度器的"心跳"，每次调用执行一轮轻量调度：
        1. 执行一轮扩散激活传播（graph.spread_activation）
        2. 执行全局激活衰减（graph.decay_all）
        3. 如果池中有活跃节点，选择并处理一个批次
        4. 返回 TickResult 统计

        Args:
            batch_size: 每批次处理的节点数

        Returns:
            TickResult 包含本次 tick 的统计信息
        """
        tick_start = time.perf_counter()
        self._total_ticks += 1

        # ── 1. 全局衰减（每个 tick 衰减所有节点 15%）──
        self.graph.decay_all()
        logger.debug("tick_decay_applied", tick=self._total_ticks)

        # ── 2. 选择并处理批次 ──
        nodes_processed = 0
        new_nodes = 0

        if not self.pool.is_empty():
            # 使用基于激活能量的节点选择
            batch_result = await self.process_batch(batch_size)
            nodes_processed = batch_result.batch_size
            new_nodes = batch_result.new_nodes_count

        # ── 3. 计算结构熵 ──
        entropy = self.graph.structural_entropy()

        tick_end = time.perf_counter()
        elapsed_ms = (tick_end - tick_start) * 1000.0

        logger.info(
            "tick_completed",
            tick=self._total_ticks,
            nodes_processed=nodes_processed,
            new_nodes=new_nodes,
            activation_spread=activation_spread,
            entropy=round(entropy, 4),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return TickResult(
            nodes_processed=nodes_processed,
            new_nodes=new_nodes,
            activation_spread=activation_spread,
            entropy=round(entropy, 4),
            elapsed_ms=round(elapsed_ms, 2),
        )

    # ═══════════════════════════════════════════════════════════
    # 外部任务注入
    # ═══════════════════════════════════════════════════════════

    def inject_external_task(self, node_id: str) -> bool:
        """Inject high activation energy to a node and its neighbors for external task processing.

        当 CEN（TaskAgent）注入外部任务时调用。向目标节点注入高激活能量，
        同时向其邻居节点注入较低激活能量，使相关推理路径被激活。

        Args:
            node_id: 外部任务节点的 ID

        Returns:
            是否成功注入（节点不存在时返回 False）
        """
        node = self.graph.get_node(node_id)
        if node is None:
            logger.warning("inject_external_task_node_not_found", node_id=node_id)
            return False

        # ── 向目标节点注入高激活能量 ──
        node.inject_activation(self._EXTERNAL_TASK_ACTIVATION)
        logger.info(
            "external_task_activation_injected",
            node_id=node_id,
            energy=self._EXTERNAL_TASK_ACTIVATION,
        )

        # ── 向邻居节点注入较低激活能量 ──
        # 获取所有邻居（父节点 + 子节点）
        neighbors = self.graph.get_parents(node_id) + self.graph.get_children(node_id)
        for neighbor in neighbors:
            if neighbor.is_active:
                neighbor.inject_activation(self._NEIGHBOR_ACTIVATION)
                logger.debug(
                    "neighbor_activation_injected",
                    neighbor_id=neighbor.node_id,
                    energy=self._NEIGHBOR_ACTIVATION,
                )

        return True

    # ═══════════════════════════════════════════════════════════
    # 批次处理（保留原有接口，增加扩散激活传播）
    # ═══════════════════════════════════════════════════════════

    async def process_batch(self, batch_size: int = 5) -> BatchResult:
        """Process a batch of nodes from the pool with activation-aware selection.

        与原版 process_batch 的区别：
        - 使用 select_next_nodes() 替代 pool.peek_top_k()，考虑激活能量
        - 批次处理完成后执行扩散激活传播和全局衰减

        Args:
            batch_size: 每批次处理的节点数

        Returns:
            BatchResult 包含本次批次的统计信息
        """
        if self.pool.is_empty():
            return BatchResult(
                batch_size=0,
                new_nodes_count=0,
                pruned_count=0,
                action_count=0,
                errors=[],
                elapsed_ms=0.0,
            )

        # ── 使用基于激活能量的节点选择 ──
        batch = self.select_next_nodes(batch_size)
        actual_size = len(batch)

        if actual_size == 0:
            return BatchResult(
                batch_size=0,
                new_nodes_count=0,
                pruned_count=0,
                action_count=0,
                errors=[],
                elapsed_ms=0.0,
            )

        start_time = time.perf_counter()

        # ── 扩散激活传播（仅传播本次 batch 节点的激活，避免旧节点残留扩散）──
        batch_ids = {n.node_id for n in batch}
        self.graph.spread_activation(source_node_ids=batch_ids)

        # ── 消耗被选中节点的激活能量 ──
        for node in batch:
            # 保存原始激活值到 metadata，供 reasoning_loop 读取
            node.metadata["_activation_before_consume"] = node.activation
            consumed = node.consume_activation()
            if consumed > 0:
                logger.debug(
                    "node_activation_consumed",
                    node_id=node.node_id,
                    consumed_energy=round(consumed, 3),
                )

        results = await asyncio.gather(
            *[self._run_node(node) for node in batch],
            return_exceptions=True,
        )

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0

        new_nodes_count = 0
        pruned_count = 0
        action_count = 0
        errors: list[str] = []

        for i, result in enumerate(results):
            node = batch[i]

            if isinstance(result, BaseException):
                error_msg = f"node {node.node_id}: {type(result).__name__}: {result}"
                errors.append(error_msg)
                logger.warning("batch_node_exception", node_id=node.node_id, error=str(result))
                node.mark_pruned(f"scheduler_exception: {type(result).__name__}")
                self.pool.remove(node.node_id)
                continue

            if result is None:
                self.pool.remove(node.node_id)
                continue

            if result.error:
                errors.append(f"node {result.node_id}: {result.error}")
                logger.warning(
                    "batch_node_logic_error",
                    node_id=result.node_id,
                    error=result.error,
                )
                node.mark_pruned(f"logic_error: {result.error}")

            new_node_list = result.new_nodes if isinstance(result.new_nodes, list) else []
            for new_node in new_node_list:
                if isinstance(new_node, GoTNode):
                    self.pool.add(new_node)
                    new_nodes_count += 1

            if result.pruned:
                pruned_count += 1

            action_outputs = getattr(result, "action_outputs", []) or []
            action_count += len(action_outputs)

            self.pool.remove(node.node_id)

        # ── 批次后全局衰减 ──
        self.graph.decay_all()

        # ── 更新统计 ──
        self._total_batches += 1
        self._total_nodes_processed += actual_size
        self._total_new_nodes += new_nodes_count
        self._total_action_nodes += action_count
        self._total_pruned_nodes += pruned_count
        self._total_errors += len(errors)
        self._total_elapsed_ms += elapsed_ms

        logger.info(
            "batch_processed",
            batch_size=actual_size,
            new_nodes=new_nodes_count,
            pruned=pruned_count,
            actions=action_count,
            errors=len(errors),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return BatchResult(
            batch_size=actual_size,
            new_nodes_count=new_nodes_count,
            pruned_count=pruned_count,
            action_count=action_count,
            errors=errors,
            elapsed_ms=round(elapsed_ms, 2),
        )

    async def _run_node(self, node: GoTNode) -> LoopResult | None:
        """Run a single reasoning node through the reasoning loop."""
        try:
            return await self._reasoning_loop.run(node)
        except Exception as e:
            logger.warning("node_run_exception", node_id=node.node_id, error=str(e))
            return LoopResult(
                node_id=node.node_id,
                new_nodes=[],
                status="failed",
                error=f"{type(e).__name__}: {e}",
            )

    # ═══════════════════════════════════════════════════════════
    # 保留的原有接口
    # ═══════════════════════════════════════════════════════════

    async def process_single(self, node_id: str) -> LoopResult | None:
        """Process a single node by ID."""
        node = self.graph.get_node(node_id)
        if node is None:
            logger.warning("process_single_node_not_found", node_id=node_id)
            return None

        return await self._run_node(node)

    def get_queue_depth(self) -> int:
        """Return the current number of nodes in the pool."""
        return self.pool.size()

    def get_throughput_stats(self) -> dict:
        """Return throughput statistics including tick and activation spread counts."""
        if self._total_nodes_processed == 0:
            return {
                "total_batches": 0,
                "total_nodes_processed": 0,
                "total_new_nodes": 0,
                "total_action_nodes": 0,
                "total_pruned_nodes": 0,
                "total_errors": 0,
                "nodes_per_second": 0.0,
                "total_elapsed_ms": 0.0,
                "success_rate": 0.0,
                "total_ticks": 0,
                "total_activation_spreads": 0,
            }

        success_count = self._total_nodes_processed - self._total_errors
        success_rate = success_count / max(self._total_nodes_processed, 1)

        nodes_per_second = 0.0
        if self._total_elapsed_ms > 0:
            nodes_per_second = (self._total_nodes_processed / self._total_elapsed_ms) * 1000.0

        return {
            "total_batches": self._total_batches,
            "total_nodes_processed": self._total_nodes_processed,
            "total_new_nodes": self._total_new_nodes,
            "total_action_nodes": self._total_action_nodes,
            "total_pruned_nodes": self._total_pruned_nodes,
            "total_errors": self._total_errors,
            "nodes_per_second": round(nodes_per_second, 2),
            "total_elapsed_ms": round(self._total_elapsed_ms, 2),
            "success_rate": round(success_rate, 4),
            "total_ticks": self._total_ticks,
            "total_activation_spreads": self._total_activation_spreads,
        }
