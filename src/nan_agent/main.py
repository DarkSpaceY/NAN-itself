"""
NAN-Agent 主入口模块 - NANAgent 类

本模块是 NAN-Agent 的顶层编排器，负责创建、配置、连接所有子系统，
并管理 Agent 的完整生命周期。它是整个项目的"大脑皮层"——
协调认知、记忆、价值、行动各模块协同工作。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
架构概览（子系统依赖关系）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ConfigLoader ──► NANAgent ──► EventBus
                        │
         ┌──────────────┼──────────────────────────┐
         │              │                           │
    OllamaProvider  LifecycleManager         ActionRoom
         │                                    │
    ┌────┴────────┬────────┬────────┐    WebSearch
    │             │        │        │    CodeExecutor
  Cognition  VectorStore GraphStore StateStore  BlobStore
    │             │
    ├─ HardMemory ─┤  (双向引用)
    │              │
    ├─ SelfValue ──┤  (双向引用)
    │              │
    ├─ SoftMemory ─┤  (双向引用, 含 PEFTManager)
    │
    └─ GoTEngine ──┤  (图推理引擎)
         │
    EvolutionOrchestrator (定期后台任务)
         │
    NANRepl (交互式命令行)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
初始化流程（initialize 方法）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 配置加载        ConfigLoader 从 YAML 文件加载所有配置
  2. 日志初始化      setup_logger 配置日志级别和输出
  3. 模型提供器      OllamaProvider 连接本地 Ollama 服务
  4. 存储层          VectorStore, GraphStore, StateStore, BlobStore
  5. PEFT 管理器     LoRA 适配器管理（最大激活数可配置）
  6. Cognition       认知引擎（核心推理组件）
  7. HardMemory      硬记忆系统（向量索引 + 事实超图 + 经验树）
  8. SelfValue       自我价值系统（MBTI 人格 + 情绪动力学）
  9. SoftMemory      软记忆系统（LoRA 微调 + 学习周期）
 10. ActionRoom      动作空间（代码执行 + 网页搜索 + MCP 工具集成）
 11. GoTEngine       Graph of Thought 图推理引擎
 12. NANRepl         交互式命令行界面
 13. Evolution       进化编排器启动后台定期任务

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
运行流程（start 方法）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  初始化 → 生命周期启动 → 信号处理注册 → 启动 GoT 推理循环（异步）
  → 启动 Evolution 进化循环（异步后台） → 进入 REPL 交互界面

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用方式：
    # 命令行启动
    python -m nan_agent.main

    # 编程方式
    agent = NANAgent(config_path="./my_config.yaml")
    asyncio.run(agent.start())
"""

import asyncio
import time
from typing import Any, Dict, Optional

from nan_agent.action_room.code_executor import CodeExecutor
from nan_agent.action_room.interface import ActionRoom
from nan_agent.action_room.web_search import WebSearch
from nan_agent.cli import NANRepl
from nan_agent.config.loader import ConfigLoader
from nan_agent.event_bus import EventBus
from nan_agent.hard_memory.interface import HardMemory
from nan_agent.inference.engine import GoTEngine
from nan_agent.lifecycle import LifecycleManager
from nan_agent.logging.logger import Timer, get_logger, log_event, new_correlation_id, setup_logger
from nan_agent.model.cognition import Cognition
from nan_agent.model.ollama_provider import OllamaProvider
from nan_agent.model.peft_manager import PEFTManager
from nan_agent.self_value.interface import SelfValue
from nan_agent.soft_memory.interface import SoftMemory
from nan_agent.storage.blob_store import BlobStore
from nan_agent.storage.graph_store import GraphStore
from nan_agent.storage.state_store import StateStore
from nan_agent.storage.vector_store import VectorStore

logger = get_logger(__name__)


class NANAgent:
    """
    NAN-Agent 主类 —— 所有子系统的顶层编排器

    该类是整个项目的入口点，负责：
    - 从配置文件加载所有参数
    - 按正确顺序创建和连接各子系统
    - 管理 Agent 的启动、运行和关闭生命周期
    - 暴露各子系统的访问接口（property）

    Attributes:
        config (dict): 从 YAML 加载的完整配置
        event_bus (EventBus): 全局事件总线，各组件通过它松耦合通信
        _lifecycle (LifecycleManager): 生命周期状态机管理器
        _provider (OllamaProvider): Ollama 模型提供器
        _vector_store (VectorStore): 向量存储（ChromaDB）
        _graph_store (GraphStore): 图存储
        _state_store (StateStore): 状态存储（SQLite）
        _blob_store (BlobStore): 二进制大对象存储
        _peft_manager (PEFTManager): LoRA 参数高效微调管理器
        _cognition (Cognition): 认知引擎
        _hard_memory (HardMemory): 硬记忆系统（长期记忆）
        _self_value (SelfValue): 自我价值系统（人格/情绪）
        _soft_memory (SoftMemory): 软记忆系统（可塑性学习）
        _action_room (ActionRoom): 动作空间（工具执行环境）
        _got_engine (GoTEngine): Graph of Thought 推理引擎
        _repl (NANRepl): 交互式命令行界面
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 NANAgent 实例。

        此阶段只完成配置加载和基础对象创建，不执行任何异步操作。
        所有子系统的实际初始化在 initialize() 方法中完成。

        Args:
            config_path: 用户配置文件路径（可选）。
                如果为 None，使用 config/defaults.yaml 中的默认配置。
        """
        # ── 配置加载 ──
        loader = ConfigLoader(user_config_path=config_path)
        self.config = loader.load()

        # ── 事件总线（全局神经系统） ──
        self.event_bus = EventBus()

        # ── 生命周期管理器 ──
        self._lifecycle = LifecycleManager(event_bus=self.event_bus)

        # ── 子系统引用（initialize 中填充） ──
        self._provider: Optional[OllamaProvider] = None
        self._vector_store: Optional[VectorStore] = None
        self._graph_store: Optional[GraphStore] = None
        self._state_store: Optional[StateStore] = None
        self._blob_store: Optional[BlobStore] = None
        self._peft_manager: Optional[PEFTManager] = None
        self._cognition: Optional[Cognition] = None
        self._hard_memory: Optional[HardMemory] = None
        self._self_value: Optional[SelfValue] = None
        self._soft_memory: Optional[SoftMemory] = None
        self._action_room: Optional[ActionRoom] = None
        self._got_engine: Optional[GoTEngine] = None
        self._repl: Optional[NANRepl] = None

        # ── 运行时状态追踪 ──
        self._start_time: float = 0.0
        self._got_task: Optional[asyncio.Task] = None  # GoT 推理循环的后台任务
        self._orchestrator = None  # EvolutionOrchestrator 实例
        self._evolution_task: Optional[asyncio.Task] = None  # 进化循环后台任务

    # ── 子系统访问属性 ──────────────────────────────────────────

    @property
    def cognition(self) -> Optional[Cognition]:
        """认知引擎（核心推理组件）。"""
        return self._cognition

    @property
    def hard_memory(self) -> Optional[HardMemory]:
        """硬记忆系统（向量索引 + 事实超图 + 经验树）。"""
        return self._hard_memory

    @property
    def self_value(self) -> Optional[SelfValue]:
        """自我价值系统（MBTI 人格 + 情绪动力学）。"""
        return self._self_value

    @property
    def soft_memory(self) -> Optional[SoftMemory]:
        """软记忆系统（LoRA 适配器 + 学习周期）。"""
        return self._soft_memory

    @property
    def action_room(self) -> Optional[ActionRoom]:
        """动作空间（代码执行 + 网页搜索 + MCP 工具集成）。"""
        return self._action_room

    @property
    def got_engine(self) -> Optional[GoTEngine]:
        """Graph of Thought 图推理引擎。"""
        return self._got_engine

    # ── 初始化 ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        初始化所有子系统（异步）。

        这是 Agent 启动的核心流程，按依赖关系顺序初始化各组件。
        每个组件的初始化都通过 Timer 记录耗时，超过阈值会发出警告。

        初始化顺序遵循依赖关系：
        - 先创建底层服务（存储、模型提供器）
        - 再创建认知层（Cognition）
        - 然后是记忆层（HardMemory、SelfValue、SoftMemory）
        - 最后是上层服务（ActionRoom、GoTEngine、NANRepl）
        """
        with Timer(logger, "agent_initialize", warn_threshold_ms=30000):
            # ── 1. 日志系统 ──
            log_config = self.config.get("logging", {})
            setup_logger(
                level=log_config.get("level", "INFO"),
                output=log_config.get("format", log_config.get("output", "console")),
                file=log_config.get("file"),
                module_levels=log_config.get("module_levels"),
            )
            new_correlation_id(prefix="session")
            log_event(logger, "logging_initialized", level=log_config.get("level", "INFO"))

            # ── 2. 模型提供器（Ollama） ──
            model_config = self.config.get("model", {})
            self._provider = OllamaProvider(
                base_url=model_config.get("base_url", "http://localhost:11434"),
                model_name=model_config.get("model_name", "gemma4:26b"),
                timeout=model_config.get("timeout", 120.0),
            )
            log_event(logger, "provider_initialized")

            # ── 3. 存储层（向量、图、状态、Blob） ──
            memory_config = self.config.get("memory", {})
            vs_config = memory_config.get("vector_store", {})
            self._vector_store = VectorStore(
                persist_directory=vs_config.get("persist_directory", "./data/chroma"),
            )

            self._graph_store = GraphStore()

            ss_config = memory_config.get("state_store", {})
            self._state_store = StateStore(
                db_path=ss_config.get("db_path", "./data/state.db"),
            )

            bs_config = memory_config.get("blob_store", {})
            self._blob_store = BlobStore(
                base_dir=bs_config.get("base_path", "./data/blobs"),
            )

            # ── 4. PEFT 管理器（LoRA 参数高效微调） ──
            self._peft_manager = PEFTManager(
                model=None,
                max_active=self.config.get("soft_memory", {}).get("lora", {}).get("max_active", 4),
            )

            # ── 5. 认知引擎（Cognition） ──
            # 注意：Cognition 初始化时硬记忆、自我价值、软记忆引用均为 None，
            # 后续各组件创建后会通过 setter 反向注入
            model_temperature = model_config.get("temperature", 0.7)
            model_top_p = model_config.get("top_p", 0.9)

            self._cognition = Cognition(
                provider=self._provider,
                hard_memory=None,
                self_value=None,
                soft_memory=None,
                default_temperature=model_temperature,
                default_top_p=model_top_p,
            )
            log_event(logger, "cognition_initialized")

            # ── 6. 硬记忆系统（HardMemory） ──
            # 依赖：VectorStore, OllamaProvider, Cognition, StateStore, BlobStore
            self._hard_memory = HardMemory(
                vector_store=self._vector_store,
                ollama_provider=self._provider,
                cognition=self._cognition,
                state_store=self._state_store,
                blob_store=self._blob_store,
            )
            log_event(logger, "hard_memory_initialized")
            await self._hard_memory.initialize()
            self._cognition.hard_memory = self._hard_memory  # 反向注入

            # ── 7. 自我价值系统（SelfValue） ──
            # 依赖：Cognition, HardMemory
            self._self_value = SelfValue(
                cognition=self._cognition,
                hard_memory=self._hard_memory,
                soft_memory=None,  # SoftMemory 尚未创建
            )
            log_event(logger, "self_value_initialized")
            self._cognition.self_value = self._self_value  # 反向注入

            # 初始化 MBTI 人格类型（默认 INTJ）
            sv_config = self.config.get("self_value", {})
            self._self_value.initialize(mbti=sv_config.get("mbti", "INTJ"))

            # ── 8. 软记忆系统（SoftMemory） ──
            # 依赖：Cognition, HardMemory, PEFTManager
            soft_memory_config = dict(self.config.get("soft_memory", {}))
            soft_memory_config.setdefault("ollama_model", model_config.get("model_name", "gemma4:26b"))

            self._soft_memory = SoftMemory(
                cognition=self._cognition,
                hard_memory=self._hard_memory,
                peft_manager=self._peft_manager,
                config=soft_memory_config,
            )
            log_event(logger, "soft_memory_initialized")
            self._cognition.soft_memory = self._soft_memory  # 反向注入
            self._self_value.soft_memory = self._soft_memory  # 反向注入

            # ── 9. 动作空间（ActionRoom） ──
            # 包含代码执行器、网页搜索、MCP 工具集成、文件系统监控等
            self._action_room = ActionRoom(
                event_bus=self.event_bus,
                config=self.config,
                lazy_init=False,  # 立即初始化所有工具
            )
            log_event(logger, "action_room_initialized")

            # 连接 MCP 服务器（例如 GitNexus 等外部工具）
            await self._action_room.connect_mcp()

            # 注册文件系统变更监听器
            # 文件系统事件不再直接写入记忆，通过 Cognition 统一入口
            if self._action_room.filesystem is not None:
                log_event(logger, "filesystem_watcher_registered")

            # ── 10. GoT 推理引擎（Graph of Thought） ──
            inference_config = self.config.get("inference", {}).get("goto", {})

            # 创建 GoT 引擎使用的工具实例
            _got_web_search = WebSearch(
                max_results=inference_config.get("search_max_results", 10),
                rate_limit=inference_config.get("search_rate_limit", 1.0),
            )
            _got_code_executor = CodeExecutor(
                default_timeout=inference_config.get("code_timeout", 30.0),
                max_output_chars=inference_config.get("code_max_output", 100_000),
            )

            self._got_engine = GoTEngine(
                cognition=self._cognition,
                hard_memory=self._hard_memory,
                self_value=self._self_value,
                soft_memory=self._soft_memory,
                web_search=_got_web_search,
                code_executor=_got_code_executor,
                config=inference_config,
                task_agent_factory=self._create_task_agent_for_got,
                embedding_provider=self._provider.embed,
            )
            log_event(logger, "got_engine_initialized")
            self._got_engine.set_event_bus(self.event_bus)

            # 如果存在之前的图快照，尝试导入恢复
            import os
            graph_path = "data/graph_snapshots/final.json"
            if os.path.exists(graph_path):
                try:
                    self._got_engine.import_graph(graph_path)
                    logger.info("graph_imported", path=graph_path)
                except Exception as e:
                    logger.warning("graph_import_failed", path=graph_path, error=str(e))

            # ── 11. REPL 交互界面 ──
            self._repl = NANRepl(
                agent=self,
                event_bus=self.event_bus,
            )
            log_event(logger, "repl_initialized")

            log_event(logger, "nan_agent_initialized")

    def _create_task_agent_for_got(self):
        """
        GoT 引擎的任务代理工厂方法。

        GoT 引擎在执行推理图时会为每个活跃节点创建短生命周期的
        TaskAgent 实例来执行具体任务。此方法作为工厂回调注入 GoTEngine。

        Returns:
            TaskAgent: 一个新创建的 TaskAgent 实例，配置了当前的认知/记忆/动作系统
        """
        from nan_agent.task_agent.agent import TaskAgent
        return TaskAgent(
            cognition=self._cognition,
            hard_memory=self._hard_memory,
            self_value=self._self_value,
            soft_memory=self._soft_memory,
            action_room=self._action_room,
            got_engine=self._got_engine,
        )

    # ── 启动与运行 ──────────────────────────────────────────────

    async def start(self) -> None:
        """
        启动 Agent 的完整运行流程。

        执行顺序：
        1. initialize() — 初始化所有子系统
        2. 生命周期管理器初始化
        3. 注册系统信号处理器（SIGINT/SIGTERM 优雅关闭）
        4. 关联感知层到 GoT 引擎
        5. 启动 GoT 推理循环（异步后台任务）
        6. 启动 Evolution 进化编排器（定期后台任务）
        7. 进入 REPL 交互界面（阻塞，直到用户退出）

        任何步骤失败或用户中断，都会触发 shutdown() 优雅关闭。
        """
        await self.initialize()
        self._start_time = time.time()

        with Timer(logger, "agent_start", warn_threshold_ms=10000):
            await self._lifecycle.initialize()

            if not self._lifecycle.is_running:
                logger.error("lifecycle_failed_to_reach_running_state")
                return

            # 注册 SIGINT/SIGTERM 信号处理
            self._lifecycle.setup_signal_handlers(self._lifecycle)
            log_event(logger, "signal_handlers_registered")

            # 将 ActionRoom 的感知接口注入 GoT 引擎
            if self._action_room and self._action_room.perception is not None:
                self._got_engine.set_perception(self._action_room.perception)

            # 启动 GoT 推理循环（异步后台任务）
            self._got_task = asyncio.create_task(
                self._got_engine.run_loop()
            )

            # 启动 Evolution 进化编排器（默认每 3600 秒执行一次进化步骤）
            from nan_agent.orchestrator import EvolutionOrchestrator
            self._orchestrator = EvolutionOrchestrator(
                hard_memory=self._hard_memory,
                soft_memory=self._soft_memory,
                cognition=self._cognition,
                self_value=self._self_value,
            )
            self._evolution_task = asyncio.ensure_future(
                self._orchestrator.start_evolution_loop(interval_seconds=3600)
            )
            log_event(logger, "evolution_orchestrator_started", interval=3600)

            log_event(logger, "nan_agent_started", components="all")

        # 进入 REPL 交互循环（阻塞）
        try:
            await self._repl.start()
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt_received")
        finally:
            await self.shutdown()

    # ── 状态查询 ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """
        获取 Agent 的完整状态信息。

        返回包含组件状态、记忆统计、推理引擎统计、人格信息等综合状态字典。
        用于 CLI 命令（/status）和监控面板展示。

        Returns:
            包含以下字段的字典：
            - agent_name, agent_version: Agent 标识
            - uptime_seconds: 运行时长
            - components: 各组件是否已初始化的布尔值字典
            - memory_stats: 记忆系统统计（总记忆单元数、活跃适配器数）
            - engine_stats: GoT 引擎统计（节点数、边数、处理步骤等）
            - personality_context: 人格上下文
            - lifecycle_state: 生命周期状态名称
            - healthy: 所有组件是否就绪
        """
        uptime = time.time() - self._start_time if self._start_time > 0 else 0.0

        # 组件就绪状态
        component_status = {
            "provider": self._provider is not None,
            "vector_store": self._vector_store is not None,
            "graph_store": self._graph_store is not None,
            "state_store": self._state_store is not None,
            "blob_store": self._blob_store is not None,
            "peft_manager": self._peft_manager is not None,
            "cognition": self._cognition is not None,
            "hard_memory": self._hard_memory is not None,
            "self_value": self._self_value is not None,
            "soft_memory": self._soft_memory is not None,
            "action_room": self._action_room is not None,
            "got_engine": self._got_engine is not None,
            "repl": self._repl is not None,
        }

        # 记忆系统统计
        memory_stats = {}
        if self._hard_memory is not None:
            memory_stats["total_memcells"] = self._hard_memory.total_count

        if self._soft_memory is not None and hasattr(self._soft_memory, '_hotswap') and self._soft_memory._hotswap:
            try:
                active_adaptors = self._soft_memory._hotswap.list_active()
                memory_stats["active_adaptors"] = len(active_adaptors) if active_adaptors else 0
            except Exception as e:
                logger.debug("status_active_adaptors_failed", error=str(e))

        # GoT 引擎统计
        engine_stats = {}
        if self._got_engine is not None:
            stats = self._got_engine.get_stats()
            engine_stats = {
                "total_nodes": stats.get("total_nodes", 0),
                "active_nodes": stats.get("active_nodes", 0),
                "total_edges": stats.get("total_edges", 0),
                "pool_size": stats.get("node_pool_size", stats.get("pool_size", 0)),
                "action_queue_size": stats.get("action_queue_size", stats.get("queue_depth", 0)),
                "steps_run": stats.get("steps_run", 0),
                "nodes_processed": stats.get("nodes_processed", 0),
                "pruned_nodes": stats.get("pruned_nodes", 0),
                "dmn_generated": stats.get("dmn_generated", 0),
                "uptime_seconds": stats.get("uptime_seconds", 0),
            }

        if self._got_engine is not None:
            try:
                summary = self._got_engine.get_graph_summary()
                if summary:
                    engine_stats["graph_summary"] = summary
            except Exception as e:
                logger.debug("status_graph_summary_failed", error=str(e))
        personality_context = {}
        if self._self_value is not None:
            try:
                personality_context = self._self_value.get_personality_context()
            except Exception as e:
                logger.debug("status_personality_context_failed", error=str(e))

        # 健康状态：所有组件都已初始化才算健康
        healthy = all(component_status.values())

        return {
            "agent_name": self.config.get("agent", {}).get("name", "NAN-Agent"),
            "agent_version": self.config.get("agent", {}).get("version", "0.1.0"),
            "uptime_seconds": uptime,
            "components": component_status,
            "memory_stats": memory_stats,
            "engine_stats": engine_stats,
            "personality_context": personality_context,
            "lifecycle_state": self._lifecycle.state.name if self._lifecycle else "unknown",
            "healthy": healthy,
        }

    # ── 手动触发操作 ────────────────────────────────────────────

    async def trigger_consolidation(self):
        """
        手动触发记忆巩固。

        Returns:
            EvolutionOrchestrator.trigger_consolidation() 的返回结果
        """
        if self._orchestrator:
            return await self._orchestrator.trigger_consolidation()

    async def trigger_learning(self):
        """
        手动触发软记忆学习周期。

        Returns:
            EvolutionOrchestrator.trigger_learning() 的返回结果
        """
        if self._orchestrator:
            return await self._orchestrator.trigger_learning()

    async def trigger_got_check(self):
        """
        手动触发 GoT 引擎健康检查。

        Returns:
            EvolutionOrchestrator.trigger_got_check() 的返回结果
        """
        if self._orchestrator:
            return await self._orchestrator.trigger_got_check()

    # ── 关闭 ────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        优雅关闭 Agent，按相反顺序释放所有资源。

        关闭顺序：
        1. 停止 GoT 推理循环并取消后台任务
        2. 导出当前图快照（用于下次恢复）
        3. 停止 Evolution 进化循环并取消后台任务
        4. 断开 MCP 连接并关闭 ActionRoom
        5. 关闭 HardMemory
        6. 关闭 OllamaProvider
        7. 生命周期管理器 shutdown
        8. 清空所有子系统引用
        """
        log_event(logger, "nan_agent_shutdown_start")

        with Timer(logger, "agent_shutdown", warn_threshold_ms=10000):
            # 停止 GoT 引擎
            if self._got_task is not None:
                await self._got_engine.stop()
                self._got_task.cancel()
                try:
                    await self._got_task
                except asyncio.CancelledError:
                    pass
                self._got_task = None

            # 导出图快照（断点续传）
            if self._got_engine is not None:
                try:
                    import os
                    os.makedirs("data/graph_snapshots", exist_ok=True)
                    self._got_engine.export_graph("data/graph_snapshots/final.json")
                    logger.info("graph_exported")
                except Exception as e:
                    logger.warning("graph_export_failed", error=str(e))
            if self._orchestrator is not None:
                self._orchestrator._running = False
            if self._evolution_task is not None:
                self._evolution_task.cancel()
                try:
                    await self._evolution_task
                except asyncio.CancelledError:
                    pass
                self._evolution_task = None

            # 断开 MCP 并关闭 ActionRoom
            if self._action_room is not None:
                try:
                    await self._action_room.disconnect_mcp()
                except Exception as e:
                    logger.warning("mcp_disconnect_error", error=str(e))
                try:
                    await self._action_room.shutdown()
                except Exception as e:
                    logger.warning("action_room_shutdown_error", error=str(e))

            # 关闭 HardMemory
            if self._hard_memory is not None:
                try:
                    await self._hard_memory.close()
                except Exception as e:
                    logger.warning("hard_memory_close_error", error=str(e))

            # 关闭模型提供器
            if self._provider is not None:
                try:
                    await self._provider.close()
                except Exception as e:
                    logger.warning("provider_close_error", error=str(e))

            # 生命周期管理器关闭
            await self._lifecycle.shutdown()

            # 清空引用（帮助 GC）
            self._provider = None
            self._cognition = None
            self._hard_memory = None
            self._self_value = None
            self._soft_memory = None
            self._action_room = None
            self._got_engine = None
            self._repl = None

            self._start_time = 0.0

            log_event(logger, "nan_agent_shutdown_complete")


def main() -> None:
    """
    NAN-Agent 命令行入口函数。

    创建 NANAgent 实例并启动异步事件循环。
    通过 KeyboardInterrupt（Ctrl+C）可优雅退出。
    """
    agent = NANAgent()
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nNAN-Agent terminated by user.")


if __name__ == "__main__":
    main()