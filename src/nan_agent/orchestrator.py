"""
进化编排器模块 - EvolutionOrchestrator

本模块实现了 NAN-Agent 的后台进化系统，负责定期执行记忆巩固、
软记忆学习、GoT 引擎健康检查等周期性任务。这是 Agent 实现"自我进化"
能力的核心组件。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
进化循环的每个周期执行以下步骤：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 记忆巩固 (Memory Consolidation)
     - 将 HardMemory 中的短期记忆整合为长期记忆结构（memscenes）
     - 搜索经验树中可用的探索-搜索候选

  2. 软记忆学习周期 (SoftMemory Learning Cycle)
     - 根据记忆单元数量触发 LoRA 适配器的训练/更新
     - 当活跃适配器数 ≥ 3 时，尝试合并生成 epoch 级合并适配器

  3. 人格巩固 (Personality Solidification)
     - 固化当前的人格状态

  4. 经验蒸馏 (Experience Distillation)
     - 当经验案例 ≥ 5 时，对经验进行蒸馏提炼
     - 移除低于成熟度阈值的过期经验

  5. 状态过期清理
     - 清理 StateStore 中的过期状态

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用方式：
    orchestrator = EvolutionOrchestrator(
        cognition=cognition,
        hard_memory=hard_memory,
        self_value=self_value,
        soft_memory=soft_memory,
        got_engine=got_engine,
    )
    # 启动后台进化循环（默认每 3600 秒一个周期）
    task = asyncio.ensure_future(
        orchestrator.start_evolution_loop(interval_seconds=3600)
    )

相关文件：
- main.py：NANAgent 在 start() 中创建 EvolutionOrchestrator 并启动后台任务
- hard_memory/interface.py：HardMemory 提供记忆巩固接口
- soft_memory/interface.py：SoftMemory 提供学习周期接口
- self_value/interface.py：SelfValue 提供人格评估接口
- inference/engine.py：GoTEngine 提供健康检查接口
"""

import asyncio
import time
from datetime import datetime, timezone

from nan_agent.logging.logger import Timer, get_logger, log_event

logger = get_logger(__name__)


class EvolutionOrchestrator:
    """
    进化编排器 —— Agent 自我进化的调度中心

    以固定间隔运行进化循环，协调记忆巩固、学习、经验蒸馏等周期性任务。
    设计上所有子任务均通过 _safe_call 执行，单个任务失败不会中断整个循环。

    Attributes:
        _cognition: 认知引擎引用
        _hard_memory: 硬记忆系统引用
        _self_value: 自我价值系统引用
        _soft_memory: 软记忆系统引用
        _got_engine: GoT 推理引擎引用（可选）
        _action_room: 动作空间引用（可选，当前未在进化循环中使用）
        _running (bool): 进化循环运行标记
        _start_time (float): 启动时间戳
        _evolution_count (int): 已执行的进化周期数
        _evolution_stats (dict): 进化统计信息
    """

    def __init__(
        self,
        cognition,
        hard_memory=None,
        self_value=None,
        soft_memory=None,
        got_engine=None,
        action_room=None,
    ):
        """
        初始化进化编排器。

        Args:
            cognition: 认知引擎（Cognition 实例）
            hard_memory: 硬记忆系统（HardMemory 实例），可选
            self_value: 自我价值系统（SelfValue 实例），可选
            soft_memory: 软记忆系统（SoftMemory 实例），可选
            got_engine: GoT 推理引擎（GoTEngine 实例），可选
            action_room: 动作空间（ActionRoom 实例），可选
        """
        self._cognition = cognition
        self._hard_memory = hard_memory
        self._self_value = self_value
        self._soft_memory = soft_memory
        self._got_engine = got_engine
        self._action_room = action_room

        self._running = False
        self._start_time = 0.0

        self._evolution_count = 0

        # 进化统计追踪
        self._evolution_stats = {
            "version": "0.1.0",
            "total_consolidations": 0,
            "total_learning_cycles": 0,
            "total_got_checks": 0,
            "total_memscenes": 0,
            "last_consolidation": None,
            "last_learning": None,
        }

    async def start_evolution_loop(self, interval_seconds=3600):
        """
        启动后台进化循环。

        在 while 循环中每隔 interval_seconds 秒执行一次进化步骤。
        循环会持续运行直到 _running 被设为 False（通过 stop() 方法或外部设置）。

        Args:
            interval_seconds: 进化周期间隔（秒），默认 3600 秒（1 小时）
        """
        self._running = True
        self._start_time = time.time()
        log_event(logger, "evolution_loop_started", interval_seconds=interval_seconds)

        while self._running:
            try:
                # ── 阶段 1: 记忆巩固（Dream 三阶段流水线） ──
                if self._hard_memory is not None:
                    with Timer(logger, "memory_consolidate", warn_threshold_ms=30000):
                        dream_stats = await self._safe_call(self._hard_memory.dream)
                    self._evolution_stats["total_consolidations"] += 1
                    if dream_stats:
                        self._evolution_stats["total_memscenes"] += dream_stats.get("facts_extracted", 0)
                    self._evolution_stats["last_consolidation"] = (
                        datetime.now(timezone.utc).isoformat()
                    )

                # ── 阶段 2: 软记忆学习周期 ──
                if self._soft_memory is not None:
                    memcell_count = (
                        self._hard_memory.total_count if self._hard_memory else 0
                    )
                    with Timer(logger, "learning_cycle", warn_threshold_ms=60000):
                        await self._safe_call(
                            self._soft_memory.run_learning_cycle, memcell_count
                        )
                    self._evolution_stats["total_learning_cycles"] += 1
                    self._evolution_stats["last_learning"] = (
                        datetime.now(timezone.utc).isoformat()
                    )

                # 输出进化统计到日志
                stats = self.get_evolution_stats()
                logger.info("evolution_stats", **stats)

            except Exception as e:
                logger.exception("evolution_loop_error", error=str(e))

            # 等待下一个周期
            await asyncio.sleep(interval_seconds)

        log_event(logger, "evolution_loop_stopped")

    def stop(self):
        """
        请求停止进化循环。

        将 _running 设为 False，当前循环周期结束后退出。
        """
        self._running = False
        logger.info("evolution_orchestrator_stop_requested")

    # ── 手动触发接口 ────────────────────────────────────────────

    async def trigger_consolidation(self):
        """
        手动触发一次记忆巩固（Dream 三阶段流水线）。

        Returns:
            HardMemory.dream() 的结果（统计信息字典），
            如果 hard_memory 不可用则返回 None
        """
        if self._hard_memory is None:
            return None
        dream_stats = await self._hard_memory.dream()
        self._evolution_stats["total_consolidations"] += 1
        if dream_stats:
            self._evolution_stats["total_memscenes"] += dream_stats.get("facts_extracted", 0)
        self._evolution_stats["last_consolidation"] = (
            datetime.now(timezone.utc).isoformat()
        )
        return dream_stats

    async def trigger_learning(self):
        """
        手动触发一次软记忆学习。

        Returns:
            SoftMemory.run_learning_cycle() 的结果，
            如果 soft_memory 不可用则返回 None
        """
        if self._soft_memory is None:
            return None
        memcell_count = self._hard_memory.total_count if self._hard_memory else 0
        result = await self._soft_memory.run_learning_cycle(memcell_count)
        self._evolution_stats["total_learning_cycles"] += 1
        self._evolution_stats["last_learning"] = (
            datetime.now(timezone.utc).isoformat()
        )
        return result

    async def trigger_got_check(self):
        """
        手动触发一次 GoT 引擎健康检查。

        Returns:
            GoTEngine.health_check() 的结果，
            如果 got_engine 不可用则返回 None
        """
        if self._got_engine is None:
            return None
        self._evolution_stats["total_got_checks"] += 1
        return await self._got_engine.health_check()

    # ── 统计查询 ────────────────────────────────────────────────

    def get_evolution_stats(self) -> dict:
        """
        获取进化统计信息。

        汇总所有子系统的最新状态，包括：
        - GoT 引擎节点/边/池大小
        - 情绪效价/唤醒度（Valence/Arousal）
        - MBTI 人格类型
        - LoRA 适配器状态（活跃数/总数）
        - 记忆单元总数
        - 经验树节点数
        - 运行时长

        Returns:
            包含完整进化统计的字典
        """
        got_stats = None
        if self._got_engine is not None:
            try:
                got_stats = self._got_engine.get_stats()
            except Exception as e:
                logger.warning("orchestrator_got_stats_failed", error=str(e))

        va = {"valence": 0.0, "arousal": 0.0}
        if self._self_value is not None:
            try:
                va = self._self_value.get_valence_arousal()
            except Exception as e:
                logger.debug("orchestrator_valence_arousal_failed", error=str(e))

        personality_type = None
        if self._self_value is not None and self._self_value.profile is not None:
            try:
                personality_type = self._self_value.profile.mbti
            except Exception as e:
                logger.debug("orchestrator_mbti_failed", error=str(e))

        active_count = 0
        total_count = 0
        if self._soft_memory is not None:
            try:
                active = self._soft_memory.get_active_adaptors()
                total = self._soft_memory.list_adaptors()
                active_count = len(active) if active else 0
                total_count = len(total) if total else 0
            except Exception as e:
                logger.debug("orchestrator_adaptors_failed", error=str(e))

        total_memcells = 0
        if self._hard_memory is not None:
            try:
                total_memcells = self._hard_memory.total_count
            except Exception:
                pass

        uptime = time.time() - self._start_time if self._start_time > 0 else 0.0

        exp_info = {}
        if self._hard_memory is not None:
            try:
                exp_info = {
                    "skill_count": self._hard_memory.skill_count,
                }
            except Exception:
                pass

        return {
            "version": self._evolution_stats["version"],
            "total_memcells": total_memcells,
            "got_nodes": got_stats.total_nodes if got_stats else 0,
            "got_edges": got_stats.total_edges if got_stats else 0,
            "pool_size": got_stats.pool_size if got_stats else 0,
            "action_queue_size": got_stats.action_queue_size if got_stats else 0,
            "memscene_count": self._evolution_stats.get("total_memscenes", 0),
            "active_adaptors": active_count,
            "total_adaptors": total_count,
            "valence": va.get("valence", 0.0),
            "arousal": va.get("arousal", 0.0),
            "personality_type": personality_type,
            "last_consolidation": self._evolution_stats.get("last_consolidation"),
            "last_learning": self._evolution_stats.get("last_learning"),
            "uptime_seconds": uptime,
            **exp_info,
        }

    # ── 完整进化步骤 ────────────────────────────────────────────

    async def run_evolution_step(self):
        """
        执行一个完整的进化步骤（所有阶段）。

        这是一个聚合操作，在一次调用中执行记忆巩固、学习、人格固化、
        适配器合并、经验蒸馏、清理等所有阶段。适合手动触发完整进化。

        各阶段：
        1. HardMemory 记忆巩固
        2. SoftMemory 学习周期 + 人格固化
        3. 适配器合并（如果活跃适配器 ≥ 3）
        4. 经验蒸馏（如果经验案例 ≥ 5）
        5. 低成熟度经验淘汰
        6. StateStore 过期清理

        Returns:
            包含各阶段结果的字典，含 "consolidation", "learning", "stats" 等键
        """
        with Timer(logger, "evolution_step", warn_threshold_ms=60000):
            results = {}

            # 阶段 1: 记忆巩固（Dream 三阶段流水线）
            if self._hard_memory is not None:
                results["consolidation"] = await self._safe_call(
                    self._hard_memory.dream
                )
                self._evolution_stats["total_consolidations"] += 1
                if results["consolidation"]:
                    self._evolution_stats["total_memscenes"] += results["consolidation"].get("facts_extracted", 0)
                self._evolution_stats["last_consolidation"] = (
                    datetime.now(timezone.utc).isoformat()
                )

            # 阶段 2: 软记忆学习 + 人格固化
            if self._soft_memory is not None:
                memcell_count = (
                    self._hard_memory.total_count if self._hard_memory else 0
                )
                results["learning"] = await self._safe_call(
                    self._soft_memory.run_learning_cycle, memcell_count
                )
                self._evolution_stats["total_learning_cycles"] += 1
                self._evolution_stats["last_learning"] = (
                    datetime.now(timezone.utc).isoformat()
                )

                # 人格固化
                if self._self_value is not None:
                    solidified = self._self_value.solidify_personality()
                    log_event(logger, "personality_solidified", **solidified)

                # 适配器合并：活跃适配器 ≥ 3 时触发
                active = self._soft_memory.get_active_adaptors()
                if len(active) >= 3:
                    adaptor_names = [a.label for a in active]
                    try:
                        merged = await self._soft_memory._peft_manager.merge_and_save(
                            names=adaptor_names,
                            output_name=f"merged_epoch_{self._evolution_count}",
                            output_path=f"data/adaptors/merged_epoch_{self._evolution_count}",
                        )
                        log_event(logger, "adaptors_merged", name=merged.label)
                    except Exception as e:
                        logger.debug("adaptors_merge_skipped", error=str(e))

            self._evolution_count += 1
            results["stats"] = self.get_evolution_stats()

            # 阶段 3: 状态过期清理
            if self._hard_memory is not None and self._hard_memory.state_store is not None:
                try:
                    self._hard_memory.state_store.expire()
                except Exception:
                    pass

            # 阶段 4: Exp 淘汰（效用追踪）
            if self._hard_memory is not None:
                try:
                    pruned = await self._hard_memory._skill_store.prune(min_uses=5, min_success_rate=0.3)
                    if pruned:
                        logger.info("skills_pruned", count=len(pruned))
                except Exception as e:
                    logger.debug("skill_prune_failed", error=str(e))

            return results

    # ── 内部工具方法 ────────────────────────────────────────────

    async def _safe_call(self, fn, *args, **kwargs):
        """
        安全调用：执行函数并静默捕获异常。

        确保单个子任务的失败不会导致整个进化循环崩溃。
        支持同步函数和异步函数。

        Args:
            fn: 要调用的函数（同步或异步）
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数返回值，如果执行失败则返回 None
        """
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        except Exception as e:
            logger.warning(
                "safe_call_failed",
                function=fn.__name__ if hasattr(fn, "__name__") else str(fn),
                error=str(e),
            )
            return None