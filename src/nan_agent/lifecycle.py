"""
生命周期管理模块 - LifecycleManager

本模块实现了 NAN-Agent 的生命周期状态机，管理 Agent 从初始化到关闭的
完整状态转换流程。

状态机设计：
    UNINITIALIZED ──► INITIALIZING ──► RUNNING ──► STOPPING ──► STOPPED
                           │               │            │
                           └──► ERROR ◄────┘            └──► ERROR

    此外 RUNNING 和 PAUSED 之间可以相互转换。

状态转换表（_VALID_TRANSITIONS）：
    当前状态           ──►  允许的目标状态
    ─────────────────────────────────────────
    UNINITIALIZED     ──►  INITIALIZING
    INITIALIZING      ──►  RUNNING, ERROR
    RUNNING           ──►  PAUSED, STOPPING
    PAUSED            ──►  RUNNING, STOPPING
    STOPPING          ──►  STOPPED, ERROR
    STOPPED           ──►  (终态，无后续转换)
    ERROR             ──►  UNINITIALIZED (可重置)

核心功能：
- 状态机验证：每次状态转换前检查合法性，非法转换抛出 LifecycleError
- 生命周期钩子：支持注册带有优先级的钩子函数，在 init/stop 阶段按序执行
- 信号处理：自动注册 SIGINT/SIGTERM 信号处理器用于优雅关闭
- 事件通知：通过 EventBus 广播状态变更事件（lifecycle.initializing 等）
- 钩子超时保护：每个钩子执行有独立的超时限制（默认 30 秒）

使用方式：
    manager = LifecycleManager(event_bus=event_bus)
    manager.setup_signal_handlers(manager)  # 注册系统信号处理
    await manager.initialize()              # INIT → RUNNING
    # ... Agent 运行期间 ...
    await manager.shutdown()                # RUNNING → STOPPED

与 EventBus 的集成：
    LifecycleManager 通过 EventBus 发出以下事件：
    - lifecycle.initializing (INITIALIZING 状态)
    - lifecycle.running     (RUNNING 状态)
    - lifecycle.stopping    (STOPPING 状态)
    - lifecycle.stopped     (STOPPED 状态)

相关文件：
- main.py：NANAgent 创建 LifecycleManager 实例并调用其生命周期方法
- event_bus.py：EventBus 提供事件分发能力
- exceptions.py：LifecycleError 定义非法状态转换异常
"""

import asyncio
import signal
from enum import Enum, auto
from typing import Optional

from nan_agent.exceptions import LifecycleError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


class LifecycleState(Enum):
    """
    生命周期状态枚举

    定义了 Agent 在其生命周期中可能处于的所有状态。
    使用 auto() 自动分配枚举值。

    States:
        UNINITIALIZED: 初始状态，Agent 尚未初始化
        INITIALIZING:  正在初始化各组件（模型、存储、记忆系统等）
        RUNNING:       正常运行状态，可接受用户交互和任务处理
        PAUSED:        暂停状态（当前版本未广泛使用）
        STOPPING:      正在关闭各组件
        STOPPED:       完全停止，终态
        ERROR:         错误状态，可从 INITIALIZING 或 STOPPING 进入
    """
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


# 合法的状态转换映射
# 键：当前状态，值：允许跳转到的目标状态集合
_VALID_TRANSITIONS = {
    LifecycleState.UNINITIALIZED: {LifecycleState.INITIALIZING},
    LifecycleState.INITIALIZING: {LifecycleState.RUNNING, LifecycleState.ERROR},
    LifecycleState.RUNNING: {LifecycleState.PAUSED, LifecycleState.STOPPING},
    LifecycleState.PAUSED: {LifecycleState.RUNNING, LifecycleState.STOPPING},
    LifecycleState.STOPPING: {LifecycleState.STOPPED, LifecycleState.ERROR},
    LifecycleState.STOPPED: set(),
    LifecycleState.ERROR: {LifecycleState.UNINITIALIZED},
}


class LifecycleManager:
    """
    NAN-Agent 生命周期管理器

    负责管理 Agent 的状态机转换、生命周期钩子的注册与执行、
    以及系统信号的优雅处理。

    Attributes:
        _state (LifecycleState): 当前生命周期状态
        _hooks (list): 已注册的生命周期钩子列表
        _event_bus (EventBus | None): 关联的事件总线，用于广播状态变更
        _phase_timeout (float): 单个钩子的执行超时时间（秒），默认 30.0
    """

    def __init__(self, event_bus=None, phase_timeout: float = 30.0):
        """
        初始化生命周期管理器。

        Args:
            event_bus: 可选的事件总线实例，用于广播状态变更事件
            phase_timeout: 每个生命周期钩子的超时时间（秒），默认 30 秒
        """
        self._state = LifecycleState.UNINITIALIZED
        self._hooks = []
        self._event_bus = event_bus
        self._phase_timeout = phase_timeout

    @property
    def state(self) -> LifecycleState:
        """返回当前生命周期状态。"""
        return self._state

    @property
    def is_running(self) -> bool:
        """检查 Agent 是否处于 RUNNING 状态。"""
        return self._state == LifecycleState.RUNNING

    async def initialize(self) -> None:
        """
        执行初始化阶段，将状态从 UNINITIALIZED 转换为 RUNNING。

        流程：
        1. 验证状态转换合法性（UNINITIALIZED → INITIALIZING）
        2. 设置状态为 INITIALIZING 并触发 lifecycle.initializing 事件
        3. 按优先级升序执行所有 on_init 钩子
        4. 设置状态为 RUNNING 并触发 lifecycle.running 事件

        Raises:
            LifecycleError: 如果当前状态不允许进入 INITIALIZING
        """
        self._validate_transition(LifecycleState.INITIALIZING)
        self._state = LifecycleState.INITIALIZING
        await self._emit_event("lifecycle.initializing")
        logger.info("lifecycle_initializing")

        hooks = sorted(self._hooks, key=lambda h: h.priority)
        await self._run_phase("init", hooks, "on_init")

        self._state = LifecycleState.RUNNING
        await self._emit_event("lifecycle.running")
        logger.info("lifecycle_running")

    async def start(self) -> None:
        """启动 Agent（等同于 initialize）。"""
        await self.initialize()

    async def stop(self) -> None:
        """
        执行停止阶段，将状态从 RUNNING/PAUSED 转换为 STOPPED。

        流程：
        1. 验证状态转换合法性（→ STOPPING）
        2. 设置状态为 STOPPING 并触发 lifecycle.stopping 事件
        3. 按优先级降序执行所有 on_stop 钩子（与 init 顺序相反）
        4. 设置状态为 STOPPED 并触发 lifecycle.stopped 事件

        Raises:
            LifecycleError: 如果当前状态不允许进入 STOPPING
        """
        self._validate_transition(LifecycleState.STOPPING)
        self._state = LifecycleState.STOPPING
        await self._emit_event("lifecycle.stopping")
        logger.info("lifecycle_stopping")

        hooks = sorted(self._hooks, key=lambda h: -h.priority)
        await self._run_phase("stop", hooks, "on_stop")

        self._state = LifecycleState.STOPPED
        await self._emit_event("lifecycle.stopped")
        logger.info("lifecycle_stopped")

    async def shutdown(self) -> None:
        """
        安全关闭：处理各种当前状态并执行 stop。

        如果处于 PAUSED 状态，先恢复到 RUNNING 再停止。
        如果已经 STOPPED 或 UNINITIALIZED，直接返回。
        """
        if self._state in (LifecycleState.STOPPED, LifecycleState.UNINITIALIZED):
            return
        if self._state == LifecycleState.PAUSED:
            self._state = LifecycleState.RUNNING
        await self.stop()

    def _validate_transition(self, target: LifecycleState) -> None:
        """
        验证状态转换是否合法。

        Args:
            target: 目标状态

        Raises:
            LifecycleError (E802): 如果当前状态不允许转换到目标状态
        """
        if target not in _VALID_TRANSITIONS.get(self._state, set()):
            raise LifecycleError(
                f"Invalid state transition: {self._state.name} -> {target.name}",
                error_code="E802",
                details={
                    "current": self._state.name,
                    "target": target.name,
                },
            )

    async def _run_phase(
        self,
        phase: str,
        hooks: list,
        attr: str,
    ) -> None:
        """
        按顺序执行指定阶段的所有钩子。

        每个钩子执行有独立超时限制（self._phase_timeout）。
        钩子执行失败不会中断其他钩子的执行（异常被记录日志后继续）。

        Args:
            phase: 阶段标识字符串，如 "init"、"stop"
            hooks: 按优先级排序后的钩子列表
            attr: 钩子对象上要调用的方法名，如 "on_init"、"on_stop"
        """
        for hook in hooks:
            handler = getattr(hook, attr, None)
            if handler is None:
                continue
            try:
                await asyncio.wait_for(
                    self._invoke_handler(handler, phase),
                    timeout=self._phase_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "lifecycle_hook_timeout",
                    hook=hook.name,
                    phase=phase,
                    timeout=self._phase_timeout,
                )
            except Exception:
                logger.exception(
                    "lifecycle_hook_error",
                    hook=hook.name,
                    phase=phase,
                )

    @staticmethod
    async def _invoke_handler(handler, phase):
        """
        调用单个钩子处理函数。

        支持异步函数（直接 await）和同步函数（在线程池中执行），
        确保同步函数不会阻塞事件循环。

        Args:
            handler: 钩子处理函数（同步或异步）
            phase: 阶段标识字符串
        """
        if asyncio.iscoroutinefunction(handler):
            await handler()
        else:
            await asyncio.to_thread(handler)

    async def _emit_event(self, event_name: str) -> None:
        """
        通过 EventBus 广播生命周期事件。

        如果未设置 event_bus，则静默跳过。

        Args:
            event_name: 事件名称，如 "lifecycle.running"
        """
        if self._event_bus is None:
            return
        try:
            await self._event_bus.emit(event_name, state=self._state)
        except Exception:
            logger.exception("lifecycle_event_emit_error", event=event_name)

    @staticmethod
    def setup_signal_handlers(
        manager: "LifecycleManager",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """
        注册系统信号处理器，实现优雅关闭。

        为 SIGINT（Ctrl+C）和 SIGTERM 注册处理函数。
        收到信号后，如果 Agent 尚未处于 STOPPING/STOPPED 状态，
        则触发 manager.shutdown() 进行优雅关闭。

        Args:
            manager: LifecycleManager 实例
            loop: 事件循环（可选，默认使用当前运行的事件循环）
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()

        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info("signal_received", signal=sig_name, signum=signum)
            if manager._state in (
                LifecycleState.STOPPING,
                LifecycleState.STOPPED,
            ):
                return
            asyncio.ensure_future(manager.shutdown(), loop=loop)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)