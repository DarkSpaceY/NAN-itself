"""
NAN-Agent 命令系统

提供可扩展的命令注册、匹配和执行框架。所有命令以 "/" 前缀输入，
支持别名（aliases）、分类（category）和模糊搜索。
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Command:
    """单条命令的定义。

    Attributes:
        name: 命令名称（不含 "/" 前缀），如 "help"
        description: 命令描述，用于 /help 显示
        handler: 异步处理函数，接收 (args, user_input, **kwargs)，返回字符串或任意值
        aliases: 命令别名列表，如 exit 的别名 ["quit", "q"]
        usage: 用法说明字符串，如 "/exit | /quit | /q"
        category: 命令分类，用于分组显示，如 "system"、"session"、"memory"、"skills"
    """
    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    aliases: list[str] = field(default_factory=list)
    usage: str = ""
    category: str = "general"

    def matches(self, user_input: str) -> bool:
        """检查用户输入是否匹配此命令（支持主名称和别名）。"""
        text = user_input.strip()
        if not text.startswith("/"):
            return False
        cmd_name = text.split()[0][1:]
        return cmd_name == self.name or cmd_name in self.aliases

    async def execute(self, user_input: str = "", **kwargs) -> Any:
        """解析用户输入中的参数，调用 handler 执行命令。"""
        parts = user_input.strip().split()
        args = parts[1:] if len(parts) > 1 else []
        return await self.handler(args=args, user_input=user_input, **kwargs)


@dataclass
class CommandResult:
    """命令执行结果。

    Attributes:
        success: 命令是否执行成功
        output: 命令的文本输出
        error: 错误信息（失败时）
        data: 附加的结构化数据
        message: 简短消息
        exit_requested: 是否请求退出 REPL
    """
    success: bool = True
    output: str = ""
    error: str = ""
    data: Any = None
    message: str = ""
    exit_requested: bool = False


class CommandRegistry:
    """命令注册表，管理所有已注册命令的匹配、执行和查询。

    通过 build_default_registry() 静态方法创建包含所有内置命令的默认实例。
    """

    def __init__(self):
        self._commands: dict[str, Command] = {}

    def register(self, command: Command):
        """注册一条命令，以命令名称为键。"""
        self._commands[command.name] = command

    async def handle(self, user_input: str, **context) -> bool | CommandResult:
        """处理用户输入：匹配命令并执行，返回 CommandResult 或 False（非命令输入）。"""
        text = user_input.strip()
        if not text.startswith("/"):
            return False

        cmd_name = text.split()[0][1:]
        cmd = self._find_command(cmd_name)
        if cmd:
            try:
                output = await cmd.execute(user_input=text, **context)
                if isinstance(output, str):
                    return CommandResult(output=output, success=True)
                return CommandResult(output=str(output), success=True)
            except Exception as e:
                logger.error("command_failed", command=cmd_name, error=str(e))
                return CommandResult(output="", success=False, error=str(e))

        return CommandResult(
            output="",
            success=False,
            error=f"Unknown command: {cmd_name}. Type /help for available commands.",
        )

    def _find_command(self, name: str) -> Optional[Command]:
        """按名称或别名查找命令。"""
        cmd = self._commands.get(name)
        if cmd:
            return cmd
        for c in self._commands.values():
            if name in c.aliases:
                return c
        return None

    def list_commands(self) -> list[dict]:
        """列出所有已注册命令的元信息。"""
        return [
            {
                "name": c.name,
                "description": c.description,
                "aliases": c.aliases,
                "usage": c.usage,
                "category": c.category,
            }
            for c in self._commands.values()
        ]

    def list_by_category(self) -> dict[str, list[dict]]:
        """按分类分组列出所有命令。"""
        result: dict[str, list[dict]] = {}
        for c in self._commands.values():
            cat = c.category
            if cat not in result:
                result[cat] = []
            result[cat].append({
                "name": c.name,
                "description": c.description,
                "aliases": c.aliases,
                "usage": c.usage,
            })
        return result

    def search_commands(self, query: str) -> list[dict]:
        """按关键词搜索命令（匹配名称和描述）。"""
        q = query.lower()
        results = []
        for c in self._commands.values():
            if q in c.name.lower() or q in c.description.lower():
                results.append({
                    "name": c.name,
                    "description": c.description,
                    "category": c.category,
                })
        return results

    @staticmethod
    def build_default_registry(agent=None, session_manager=None) -> "CommandRegistry":
        """构建包含所有内置命令的默认 CommandRegistry。

        内置命令按分类组织：
        - 系统类 (system): /help, /status, /config, /exit, /clear, /abort, /commands
        - 会话类 (session): /new, /switch, /sessions, /session_load, /session_export,
                           /session_delete, /session_rename
        - 记忆类 (memory): /memory
        - 技能类 (skills): /skills
        - 推理类 (system): /level, /got, /goto
        - 工具类 (system): /tools, /tool_stats, /quota, /multimodal

        Args:
            agent: NAN-Agent 实例，提供各子系统（cognition、memory 等）的访问入口
            session_manager: 会话管理器，提供会话 CRUD 操作

        Returns:
            包含所有内置命令的 CommandRegistry 实例
        """
        registry = CommandRegistry()
        # 闭包引用，供内部 handler 函数访问外部参数
        _agent = agent
        _session_manager = session_manager

        # ================================================================
        # 系统命令 (system)
        # ================================================================

        async def help_handler(**kwargs):
            """显示所有已注册命令的名称、描述和别名。"""
            lines = ["Available commands:"]
            for c in registry._commands.values():
                aliases_str = ""
                if c.aliases:
                    aliases_str = f" (aliases: {', '.join('/' + a for a in c.aliases)})"
                lines.append(f"  /{c.name:<10} - {c.description}{aliases_str}")
            return "\n".join(lines)

        registry.register(Command(
            name="help",
            description="Show available commands",
            handler=help_handler,
            usage="/help",
            category="system",
        ))

        async def status_handler(**kwargs):
            """显示 Agent 运行状态，包括 GoT 节点数、记忆条目数、人格类型等。"""
            agent_obj = kwargs.get("agent", _agent)
            if agent_obj and hasattr(agent_obj, "get_status"):
                status = agent_obj.get_status()
                if isinstance(status, dict):
                    lines = ["NAN-Agent Status:"]
                    for k, v in status.items():
                        lines.append(f"  {k}: {v}")
                    return "\n".join(lines)
                return str(status)
            return "Agent not initialized"

        registry.register(Command(
            name="status",
            description="Show agent status (GoT nodes, memory count, personality, etc.)",
            handler=status_handler,
            usage="/status",
            category="system",
        ))

        # ================================================================
        # 记忆命令 (memory)
        # ================================================================

        async def memory_handler(**kwargs):
            """搜索硬记忆或显示记忆统计。无参数时显示总条目数，带参数时执行语义搜索。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            if agent_obj and agent_obj.hard_memory:
                hm = agent_obj.hard_memory
                if args:
                    query = " ".join(args)
                    if hasattr(hm, "recollect"):
                        results = await hm.recollect(query, k=5)
                        if results:
                            lines = [f"Memory search results for '{query}':"]
                            for i, r in enumerate(results, 1):
                                lines.append(f"  {i}. {str(r)[:120]}")
                            return "\n".join(lines)
                    return f"No results found for '{query}'"
                total = getattr(hm, "total_count", 0)
                return f"Total MemCells: {total}"
            return "Memory system not available"

        registry.register(Command(
            name="memory",
            description="Search memory or show stats",
            handler=memory_handler,
            usage="/memory [query]",
            category="memory",
        ))

        # ================================================================
        # 会话命令 (session)
        # ================================================================

        registry.register(Command(
            name="sessions",
            description="List all sessions",
            handler=lambda **kw: (
                (_session_manager.list_sessions()
                 if _session_manager is not None
                 else "No session manager available")
            ),
            usage="/sessions",
            category="session",
        ))

        async def session_load_handler(**kwargs):
            """按 ID 加载已保存的会话到内存。"""
            sm = kwargs.get("session_manager", _session_manager)
            args = kwargs.get("args", [])
            if sm:
                if not args:
                    return "Usage: /session_load <session_id>"
                session = sm.load_session(args[0])
                if session:
                    return f"Loaded session: [{session.session_id}] {session.name}"
                return f"Session not found: {args[0]}"
            return "Session manager not available"

        registry.register(Command(
            name="session_load",
            description="Load a saved session by ID",
            handler=session_load_handler,
            usage="/session_load <session_id>",
            category="session",
        ))

        async def session_export_handler(**kwargs):
            """将会话导出为 JSON 格式字符串。"""
            sm = kwargs.get("session_manager", _session_manager)
            args = kwargs.get("args", [])
            if sm:
                if not args:
                    return "Usage: /session_export <session_id>"
                result = sm.export_session(args[0])
                if result:
                    return json.dumps(result, indent=2, ensure_ascii=False)
                return f"Session not found: {args[0]}"
            return "Session manager not available"

        registry.register(Command(
            name="session_export",
            description="Export a session as JSON",
            handler=session_export_handler,
            usage="/session_export <session_id>",
            category="session",
        ))

        async def new_handler(**kwargs):
            """创建新会话并自动设为当前活跃会话。"""
            sm = kwargs.get("session_manager", _session_manager)
            if sm:
                session = sm.new_session()
                return f"New session created: [{session.session_id}] {session.name}"
            return "Session manager not available"

        registry.register(Command(
            name="new",
            description="Start a new session",
            handler=new_handler,
            usage="/new",
            category="session",
        ))

        async def switch_handler(**kwargs):
            """切换到指定 ID 的会话。无参数时列出所有可用会话。"""
            sm = kwargs.get("session_manager", _session_manager)
            args = kwargs.get("args", [])
            if not sm:
                return "Session manager not available"
            if not args:
                sessions = sm.list_sessions()
                if not sessions:
                    return "No sessions to switch to"
                lines = ["Usage: /switch <session_id>", "", "Available sessions:"]
                for s in sessions:
                    lines.append(f"  [{s.session_id}] {s.name}")
                return "\n".join(lines)
            sid = args[0]
            session = sm.get_session(sid)
            if session:
                sm.set_current(sid)
                return f"Switched to session: [{session.session_id}] {session.name}"
            return f"Session not found: {sid}"

        registry.register(Command(
            name="switch",
            description="Switch to another session",
            handler=switch_handler,
            usage="/switch <session_id>",
            category="session",
        ))

        # ================================================================
        # 技能命令 (skills)
        # ================================================================

        async def skills_handler(**kwargs):
            """搜索技能树。无参数时列出所有技能树，带参数时执行关键词搜索。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            if agent_obj and agent_obj.action_room:
                ar = agent_obj.action_room
                if hasattr(ar, "skill_trees") and ar.skill_trees:
                    st = ar.skill_trees
                    if args:
                        query = " ".join(args)
                        if hasattr(st, "search"):
                            results = st.search(query)
                            if results:
                                lines = [f"Skills matching '{query}':"]
                                for s in results:
                                    name = s.get("name", s) if isinstance(s, dict) else str(s)
                                    desc = s.get("description", "") if isinstance(s, dict) else ""
                                    entry = f"  - {name}"
                                    if desc:
                                        entry += f": {desc}"
                                    lines.append(entry)
                                return "\n".join(lines)
                        return f"No skills found for '{query}'"
                    if hasattr(st, "list_trees"):
                        trees = st.list_trees()
                        return f"Skill trees: {', '.join(trees)}" if trees else "No skill trees available"
            return "Skill system not available"

        registry.register(Command(
            name="skills",
            description="Search skill trees",
            handler=skills_handler,
            usage="/skills [query]",
            category="skills",
        ))

        # ================================================================
        # 配置命令 (system)
        # ================================================================

        async def config_handler(**kwargs):
            """查看或搜索配置项。无参数时列出顶层配置键，带参数时查询指定键值。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            if agent_obj and hasattr(agent_obj, "config"):
                config = agent_obj.config
                if args:
                    key = args[0]
                    parts = key.split(".")
                    value = config
                    for p in parts:
                        if isinstance(value, dict):
                            value = value.get(p)
                        else:
                            value = None
                            break
                    if value is not None:
                        return f"{key} = {value}"
                    return f"Config key not found: {key}"
                return "Available top-level config keys:\n  " + "\n  ".join(config.keys())
            return "Config not available"

        registry.register(Command(
            name="config",
            description="Show or search configuration",
            handler=config_handler,
            usage="/config [key]",
            category="system",
        ))

        async def exit_handler(**kwargs):
            """请求退出 REPL 循环，返回特殊标记 '__EXIT__'。"""
            return "__EXIT__"

        registry.register(Command(
            name="exit",
            description="Exit the agent",
            handler=exit_handler,
            aliases=["quit", "q"],
            usage="/exit | /quit | /q",
            category="system",
        ))

        async def clear_handler(**kwargs):
            """清空终端屏幕。"""
            os.system("clear" if os.name != "nt" else "cls")
            return "Screen cleared"

        registry.register(Command(
            name="clear",
            description="Clear the screen",
            handler=clear_handler,
            usage="/clear",
            category="system",
        ))

        # ================================================================
        # 多模态 & 工具统计命令
        # ================================================================

        registry.register(Command(
            name="multimodal",
            description="Process multimodal input (image/audio). Usage: /multimodal <text description>",
            handler=lambda text="", **kw: _agent.run_multi_modal(
                __import__("nan_agent.model.types", fromlist=["MultiModalInput"]).MultiModalInput()
            ) if _agent is not None else "No agent available",
        ))

        async def tool_stats_handler(**kwargs):
            """显示工具使用统计。无参数时列出所有工具，带参数时查询指定工具。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            if agent_obj and agent_obj.action_room:
                registry = agent_obj.action_room.registry
                tool_name = args[0] if args else ""
                if tool_name:
                    stats = registry.get_statistics(tool_name)
                    if stats:
                        return f"Tool: {tool_name}\n  calls={stats.calls}, success={stats.successful_calls}, fail={stats.failed_calls}, avg_time={stats.average_execution_time_ms:.2f}ms"
                    return f"No statistics for tool: {tool_name}"
                all_stats = registry.get_all_statistics()
                if not all_stats:
                    return "No tool statistics available"
                lines = ["Tool Statistics:"]
                for name, s in sorted(all_stats.items()):
                    lines.append(f"  {name}: calls={s.calls}, success={s.successful_calls}, fail={s.failed_calls}, avg_time={s.average_execution_time_ms:.2f}ms")
                return "\n".join(lines)
            return "No registry available"

        registry.register(Command(
            name="tool_stats",
            description="Show tool usage statistics. Usage: /tool_stats [tool_name]",
            handler=tool_stats_handler,
            usage="/tool_stats [tool_name]",
            category="system",
        ))

        registry.register(Command(
            name="commands",
            description="List or search all available commands. Usage: /commands [search_term]",
            handler=lambda query="", **kw: "\n".join(
                f"/{c['name']}: {c['description']}"
                for c in (registry.search_commands(query) if query else registry.list_commands())
            ),
            category="system",
        ))

        # ================================================================
        # GoT 推理命令
        # ================================================================

        async def got_handler(**kwargs):
            """GoT 图操作：status 显示统计，mermaid 导出 Mermaid 图。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            action = args[0] if args else "status"
            if agent_obj and agent_obj.got_engine:
                engine = agent_obj.got_engine
                if action == "status":
                    return str(engine.get_stats())
                elif action == "mermaid":
                    return engine.export_mermaid()
                else:
                    return f"Unknown GoT action: {action}. Available: status, mermaid"
            return "GoT engine not available"

        registry.register(Command(
            name="got",
            description="GoT graph operations: status/export/mermaid",
            handler=got_handler,
            usage="/got [status|mermaid]",
            category="system",
        ))

        async def abort_handler(**kwargs):
            """中止当前正在执行的 Agent 任务，可选提供中止原因。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            reason = args[0] if args else ""
            if agent_obj and hasattr(agent_obj, 'abort'):
                agent_obj.abort(reason)
                return "Task aborted"
            return "Agent not available"

        registry.register(Command(
            name="abort",
            description="Abort the current agent task",
            handler=abort_handler,
            usage="/abort [reason]",
            category="system",
        ))

        async def goto_handler(**kwargs):
            """将任务委托给 GoT 引擎执行，由 GoT 引擎进行图推理。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            text = " ".join(args) if args else ""
            if agent_obj and hasattr(agent_obj, 'delegate_to_got') and agent_obj.delegate_to_got:
                return str(agent_obj.delegate_to_got(text))
            return "GoT engine not available"

        registry.register(Command(
            name="goto",
            description="Delegate task to GoT engine. Usage: /goto <task description>",
            handler=goto_handler,
            usage="/goto <task description>",
            category="system",
        ))

        async def level_handler(**kwargs):
            """扩散模式信息。当前版本由模型自主选择 hot/cold 模式，不再支持手动覆盖。"""
            agent_obj = kwargs.get("agent", _agent)
            if agent_obj and agent_obj.got_engine:
                return "Diffusion mode is now autonomously chosen by the model (hot/cold). No manual level override available."
            return "GoT engine not available"

        registry.register(Command(
            name="level",
            description="Diffusion mode is autonomously chosen by the model (hot/cold).",
            handler=level_handler,
            usage="/level [0-4|auto]",
            aliases=["lv"],
            category="system",
        ))

        # ================================================================
        # 会话管理命令（续）
        # ================================================================

        async def session_delete_handler(**kwargs):
            """按 ID 删除指定会话（内存和文件均删除）。"""
            sm = kwargs.get("session_manager", _session_manager)
            args = kwargs.get("args", [])
            if sm:
                if not args:
                    return "Usage: /session_delete <session_id>"
                result = sm.delete_session(args[0])
                return str(result)
            return "Session manager not available"

        registry.register(Command(
            name="session_delete",
            description="Delete a session by ID",
            handler=session_delete_handler,
            usage="/session_delete <session_id>",
            category="session",
        ))

        async def session_rename_handler(**kwargs):
            """重命名指定会话。需要提供会话 ID 和新名称。"""
            sm = kwargs.get("session_manager", _session_manager)
            args = kwargs.get("args", [])
            if sm:
                if len(args) < 2:
                    return "Usage: /session_rename <session_id> <new_name>"
                result = sm.rename_session(args[0], " ".join(args[1:]))
                return str(result)
            return "Session manager not available"

        registry.register(Command(
            name="session_rename",
            description="Rename a session. Usage: /session_rename <id> <new_name>",
            handler=session_rename_handler,
            usage="/session_rename <session_id> <new_name>",
            category="session",
        ))

        # ================================================================
        # 工具 & 系统状态命令
        # ================================================================

        async def tools_handler(**kwargs):
            """列出可用工具。无参数时列出所有工具，带参数时按分类筛选。"""
            agent_obj = kwargs.get("agent", _agent)
            args = kwargs.get("args", [])
            category = args[0] if args else ""
            if agent_obj and agent_obj.action_room:
                ar = agent_obj.action_room
                if not category:
                    return str(ar.list_all_tools_with_metadata())
                return str(ar.list_tools_by_category(category))
            return "No action room available"

        registry.register(Command(
            name="tools",
            description="List available tools. Usage: /tools [category]",
            handler=tools_handler,
            usage="/tools [category]",
            category="system",
        ))

        async def comp_status_handler(**kwargs):
            """显示 ActionRoom 各组件的健康状态。"""
            agent_obj = kwargs.get("agent", _agent)
            if agent_obj and agent_obj.action_room:
                ar = agent_obj.action_room
                return str(ar.get_component_status())
            return "No action room available"

        registry.register(Command(
            name="status",
            description="Show component health status",
            handler=comp_status_handler,
            usage="/status",
            category="system",
        ))

        async def quota_handler(**kwargs):
            """显示文件系统配额信息。"""
            agent_obj = kwargs.get("agent", _agent)
            if agent_obj and agent_obj.action_room:
                ar = agent_obj.action_room
                return str(ar.get_filesystem_quota())
            return "No action room available"

        registry.register(Command(
            name="quota",
            description="Show filesystem quota info",
            handler=quota_handler,
            usage="/quota",
            category="system",
        ))

        return registry