"""
NAN-Agent REPL（交互式命令行界面）

提供完整的交互式 REPL 循环，整合仪表盘实时渲染、命令处理、
任务执行和会话管理。支持历史记录持久化、度量指标采集和事件监听。

主要组件：
- ReplConfig: REPL 配置，包括提示符、历史文件路径、自动保存等
- NANRepl: REPL 主类，管理整个交互循环的生命周期
"""

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from nan_agent.cli.commands import CommandRegistry, CommandResult
from nan_agent.cli.dashboard import DashboardRenderer
from nan_agent.cli.display import Display, NANDisplay
from nan_agent.cli.session import SessionManager
from nan_agent.event_bus import EventBus
from nan_agent.logging.logger import Timer, get_logger, log_event
from nan_agent.task_agent.agent import TaskAgent

logger = get_logger(__name__)


@dataclass
class ReplConfig:
    """REPL 配置数据类。

    Attributes:
        prompt: 输入提示符，默认 ">>> "
        history_file: 历史记录持久化文件路径，默认 "./data/repl_history.json"
        max_history: 内存中保留的最大历史记录数，默认 1000
        auto_save: 是否自动保存会话和历史记录，默认 True
        welcome_message: 启动欢迎消息
    """
    prompt: str = ">>> "
    history_file: str = "./data/repl_history.json"
    max_history: int = 1000
    auto_save: bool = True
    welcome_message: str = "Welcome to NAN-Agent v2.0"


WELCOME_BANNER = r"""
╔══════════════════════════════════════════════╗
║                                              ║
║   ███╗   ██╗ █████╗ ███╗   ██╗              ║
║   ████╗  ██║██╔══██╗████╗  ██║              ║
║   ██╔██╗ ██║███████║██╔██╗ ██║              ║
║   ██║╚██╗██║██╔══██║██║╚██╗██║              ║
║   ██║ ╚████║██║  ██║██║ ╚████║              ║
║   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝              ║
║                                              ║
║        NAN-Agent v2.0 — Self-Evolving        ║
║                                              ║
╚══════════════════════════════════════════════╝
"""

GOODBYE_MESSAGE = "\nGoodbye! NAN-Agent shutting down."


class NANRepl:
    """NAN-Agent 交互式 REPL 主类。

    管理整个交互循环的生命周期，包括：
    - 仪表盘实时渲染（神经调节剂、GoT 引擎状态、事件流）
    - 命令处理（以 "/" 前缀开头）
    - 任务执行（普通文本输入，委托给 TaskAgent）
    - 会话管理（多会话切换、持久化）
    - 历史记录加载/保存
    - 度量指标后台采集
    - 事件总线监听

    Args:
        agent: NAN-Agent 实例（可选），提供 cognition、memory、self_value 等子系统
        session_manager: 会话管理器（可选），默认创建新实例
        event_bus: 事件总线（可选），默认创建新实例
        config: REPL 配置（可选），默认使用 ReplConfig() 默认值
    """

    def __init__(
        self,
        agent=None,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[EventBus] = None,
        config: Optional[ReplConfig] = None,
    ):
        self._agent = agent
        self._event_bus = event_bus or EventBus()
        self._config = config or ReplConfig()
        self._sessions = session_manager or SessionManager()
        self._commands = CommandRegistry.build_default_registry(
            agent=agent,
            session_manager=self._sessions,
        )
        self._running = False
        self._history: list[str] = []
        self._dashboard = DashboardRenderer()
        self._display = NANDisplay()
        self._metrics_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._load_history()

    async def start(self) -> None:
        """启动 REPL 主循环。

        执行流程：
        1. 显示欢迎横幅和提示信息
        2. 创建初始会话
        3. 启动后台度量采集任务（_collect_metrics_loop）
        4. 启动后台事件监听任务（_watch_events_loop）
        5. 进入交互循环：
           - 渲染仪表盘
           - 读取用户输入
           - 若为命令（/ 前缀），交由 CommandRegistry 处理
           - 若为普通文本，委托给 TaskAgent 执行
           - 将交互记录保存到当前会话
        6. 退出时执行清理（_cleanup）

        退出条件：用户输入 /exit 或按 Ctrl+D
        """
        self._running = True

        print(self.get_welcome_banner())
        print(self._config.welcome_message)
        print('Type /help for available commands.\n')

        self._sessions.new_session()
        log_event(logger, "repl_session_created")

        logger.info("repl_started")

        await self._event_bus.emit("repl.started")

        self._metrics_task = asyncio.create_task(self._collect_metrics_loop())
        self._event_task = asyncio.create_task(self._watch_events_loop())

        while self._running:
            try:
                self._dashboard.render()
                self._dashboard.set_last_output("")

                user_input = await self.read_input(self.get_prompt())
                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                self.add_to_history(user_input)
                log_event(logger, "repl_user_input", length=len(user_input))

                if user_input.startswith("/"):
                    result = await self.process_command(user_input)
                    if result.exit_requested:
                        self._running = False
                        print(GOODBYE_MESSAGE)
                        break
                    if result.success and result.output:
                        output_text = str(result.output)
                        print(self._render_output(result.output))
                        self._dashboard.set_last_output(output_text[:80])
                    continue

                session = self._sessions.get_current()
                if session:
                    session.add_message(role="user", content=user_input)

                try:
                    response = await self.process_task(user_input)
                    print(self._render_output(response))
                    self._dashboard.set_last_output(response[:80] if isinstance(response, str) else str(response)[:80])
                except Exception as e:
                    response = f"Task execution failed: {e}"
                    print(Display.format_error(response))

                if session:
                    session.add_message(role="agent", content=response)

                if self._config.auto_save:
                    self._sessions.save_current()

            except KeyboardInterrupt:
                print("\nInterrupted. Type /exit to quit or Ctrl+D to exit.")
            except EOFError:
                print(GOODBYE_MESSAGE)
                break
            except Exception:
                logger.exception("repl_error")
                print(Display.format_error("An internal error occurred. Check the logs for details."))

        await self._cleanup()

        logger.info("repl_stopped")

    async def _collect_metrics_loop(self) -> None:
        """后台度量采集循环（每 1.5 秒执行一次）。

        采集以下指标并更新仪表盘快照：
        - 神经调节剂浓度：从 self_value.dynamics 获取各调节剂的当前浓度
        - 情感效价/唤醒度：从 self_value.get_valence_arousal() 获取
        - GoT 引擎统计：总节点数、活跃池大小、已执行步骤、DMN 生成数、剪枝数、动作队列
        - 最近活跃节点：从 GoT 图中获取最近 3 个活跃节点的内容摘要
        - 硬记忆总数：从 hard_memory.total_count 获取

        所有采集操作均被 try/except 包裹，确保单个指标采集失败不影响其他指标。
        """
        while self._running:
            try:
                if self._agent is None:
                    await asyncio.sleep(2)
                    continue

                agent = self._agent

                neuromodulators: dict[str, float] = {}
                valence = 0.0
                arousal = 0.0
                if agent.self_value:
                    sv = agent.self_value
                    try:
                        va = sv.get_valence_arousal()
                        valence = va.get("valence", 0.0)
                        arousal = va.get("arousal", 0.0)
                    except Exception:
                        pass
                    try:
                        states = sv.dynamics.get_states()
                        neuromodulators = {
                            name: state.concentration
                            for name, state in states.items()
                        }
                    except Exception:
                        pass

                got_stats = None
                got_recent: list[str] = []
                if agent.got_engine:
                    try:
                        got_stats = agent.got_engine.get_stats()
                    except Exception:
                        pass
                    try:
                        nodes = agent.got_engine.graph.get_active_nodes()
                        got_recent = [
                            node.content[:60]
                            for node in list(nodes)[-3:]
                        ]
                    except Exception:
                        pass

                memory_total = 0
                if agent.hard_memory:
                    memory_total = getattr(agent.hard_memory, "total_count", 0)

                self._dashboard.update_snapshot(
                    neuromodulators=neuromodulators,
                    valence=valence,
                    arousal=arousal,
                    got_stats=got_stats,
                    got_recent_nodes=got_recent,
                    memory_total=memory_total,
                )

            except Exception:
                pass

            await asyncio.sleep(1.5)

    async def _watch_events_loop(self) -> None:
        """后台事件监听循环。

        订阅事件总线上的所有事件（"*"），将事件名称和简要详情
        追加到仪表盘事件环形缓冲区中。
        """
        import json

        async def on_any_event(event_name, *args, **kwargs):
            detail_parts = []
            for arg in args:
                if isinstance(arg, dict):
                    detail_parts.append(json.dumps(arg, default=str)[:60])
                elif isinstance(arg, str) and len(arg) < 60:
                    detail_parts.append(arg)
            detail = " ".join(detail_parts)[:100]
            self._dashboard.events.append(event_name, detail)

        self._event_bus.on("*")(on_any_event)

        while self._running:
            try:
                await asyncio.sleep(10)
            except Exception:
                pass

    async def _cleanup(self) -> None:
        """清理资源：取消后台任务、保存会话和历史记录、发送停止事件。"""
        for task in [self._metrics_task, self._event_task]:
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._config.auto_save:
            self._sessions.save_current()
        self.save_history()
        await self._event_bus.emit("repl.stopped")

    def stop(self) -> None:
        """设置运行标志为 False，触发主循环退出。"""
        self._running = False
        logger.info("repl_stop_requested")

    async def read_input(self, prompt: str) -> Optional[str]:
        """异步读取用户输入，使用 run_in_executor 避免阻塞事件循环。"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, lambda: input(prompt))
        except (EOFError, KeyboardInterrupt):
            return None

    async def process_command(self, input_text: str) -> CommandResult:
        """处理以 "/" 开头的命令输入，委托给 CommandRegistry 执行。"""
        with Timer(logger, "repl_process_command", warn_threshold_ms=5000):
            result = await self._commands.handle(input_text)

            if result is False:
                return CommandResult(success=False, message="Not a command")

            if result.output == "__EXIT__":
                return CommandResult(success=True, exit_requested=True)

            return result

    async def process_task(self, input_text: str) -> str:
        """处理普通文本输入（非命令），创建 TaskAgent 实例执行任务。"""
        with Timer(logger, "repl_process_task", warn_threshold_ms=30000):
            task_agent = self._create_task_agent()

            try:
                task_result = await task_agent.run(input_text)
                if task_result.success:
                    return task_result.result or "Task completed successfully."
                else:
                    return f"Task failed: {task_result.error or 'Unknown error'}"
            finally:
                task_agent.destroy()

    def _create_task_agent(self) -> TaskAgent:
        """创建 TaskAgent 实例，复用当前 agent 的子系统（cognition、memory 等）。"""
        if self._agent is None:
            return TaskAgent(cognition=None)

        return TaskAgent(
            cognition=self._agent.cognition,
            hard_memory=self._agent.hard_memory,
            self_value=self._agent.self_value,
            soft_memory=self._agent.soft_memory,
            action_room=self._agent.action_room,
            got_engine=self._agent.got_engine,
        )

    def add_to_history(self, input_text: str) -> None:
        """将输入添加到历史记录，超出 max_history 时自动截断。"""
        self._history.append(input_text)
        if len(self._history) > self._config.max_history:
            self._history = self._history[-self._config.max_history:]

    def save_history(self) -> None:
        """将历史记录持久化到 JSON 文件。"""
        if not self._config.history_file:
            return
        try:
            path = Path(self._config.history_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._history, indent=2))
        except Exception as e:
            logger.warning("save_history_failed", error=str(e))

    def _load_history(self) -> None:
        """从 JSON 文件加载历史记录。"""
        if not self._config.history_file:
            return
        try:
            path = Path(self._config.history_file)
            if path.exists():
                data = json.loads(path.read_text())
                if isinstance(data, list):
                    self._history = data[-self._config.max_history:]
        except Exception as e:
            logger.warning("load_history_failed", error=str(e))

    def get_welcome_banner(self) -> str:
        """返回去除首尾空白的 ASCII 艺术欢迎横幅。"""
        return WELCOME_BANNER.strip()

    def _render_output(self, result) -> str:
        """根据结果类型智能选择渲染格式。

        渲染策略：
        - dict: 若含 status/component 键则用状态面板，否则用 JSON 格式化
        - list[dict]: 提取键作为表头，渲染为表格（最多 20 行）
        - list: 渲染为项目符号列表（最多 20 项）
        - str: 超过 500 字符则截断；含换行且 <500 字符则用代码块；否则直接输出
        - bool: 成功/警告提示
        - int/float: 渲染为进度条（值映射到 0-100）
        - 其他: str() 转换
        """
        if isinstance(result, dict):
            if "status" in result or "component" in result:
                return self._display.format_status_panel(
                    {k: str(v) for k, v in result.items()},
                    title=result.get("title", "Status"),
                )
            return self._display.format_json(result)
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            headers = list(result[0].keys())
            rows = [[str(r.get(h, "")) for h in headers] for r in result[:20]]
            return self._display.format_table(headers, rows)
        elif isinstance(result, list):
            return self._display.format_list([str(r) for r in result[:20]])
        elif isinstance(result, str):
            if len(result) > 500:
                return self._display.truncate(result, 500)
            if "\n" in result and len(result) < 500:
                return self._display.format_code(result)
            return result
        elif isinstance(result, bool):
            return self._display.format_success("Success") if result else self._display.format_warning("Failed")
        elif isinstance(result, (int, float)):
            return self._display.format_progress(min(int(result), 100), 100, "Progress")
        return str(result)

    def _display_got_status(self, stats: dict) -> str:
        """以表格形式渲染 GoT 引擎状态统计。"""
        headers = ["Metric", "Value"]
        rows = [[k, str(v)] for k, v in stats.items()]
        return self._display.format_table(headers, rows)

    def _display_memory_stats(self, memory_info: dict) -> str:
        """以表格形式渲染记忆系统统计信息。"""
        headers = ["Metric", "Value"]
        rows = [[k, str(v)] for k, v in memory_info.items()]
        return self._display.format_table(headers, rows)

    def _display_session_list(self, sessions: list) -> str:
        """以表格形式渲染会话列表（支持 dict 和对象两种格式）。"""
        if not sessions:
            return self._display.format_warning("No sessions available")
        headers = ["ID", "Name", "Messages", "Created"]
        rows = []
        for s in sessions[:20]:
            if isinstance(s, dict):
                rows.append([
                    str(s.get("session_id", s.get("id", ""))),
                    str(s.get("name", "")),
                    str(s.get("message_count", s.get("messages", 0))),
                    str(s.get("created_at", "")),
                ])
            else:
                rows.append([
                    str(getattr(s, "session_id", "")),
                    str(getattr(s, "name", "")),
                    str(getattr(s, "message_count", 0)),
                    str(getattr(s, "created_at", "")),
                ])
        return self._display.format_table(headers, rows)

    def _display_health_check(self, status: dict) -> str:
        """以表格形式渲染组件健康检查状态。"""
        headers = ["Component", "Status"]
        rows = [[k, str(v)] for k, v in status.items()]
        return self._display.format_table(headers, rows)

    def get_prompt(self) -> str:
        """返回当前提示符，如果有活跃会话则加上 session_id 前缀。"""
        session = self._sessions.get_current()
        if session:
            return f"[{session.session_id}] {self._config.prompt}"
        return self._config.prompt