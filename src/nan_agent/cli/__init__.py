"""
NAN-Agent CLI 模块

提供命令行交互界面（REPL）、会话管理、命令系统、仪表盘渲染和显示格式化等核心功能。
主要组件：
- NANRepl: 交互式 REPL 主循环，整合仪表盘、命令处理和任务执行
- Session/SessionManager: 会话持久化与多会话管理
- Command/CommandRegistry: 可扩展的命令系统，支持别名、分类和搜索
- DashboardRenderer: 实时仪表盘，展示神经调节剂、GoT 引擎状态和事件流
- Display: 静态格式化工具，支持 JSON、表格、状态面板、进度条等输出格式
"""

from nan_agent.cli.repl import NANRepl, ReplConfig
from nan_agent.cli.session import Session, SessionManager
from nan_agent.cli.commands import Command, CommandResult, CommandRegistry
from nan_agent.cli.display import Display

__all__ = [
    "NANRepl",
    "ReplConfig",
    "Session",
    "SessionManager",
    "Command",
    "CommandResult",
    "CommandRegistry",
    "Display",
]