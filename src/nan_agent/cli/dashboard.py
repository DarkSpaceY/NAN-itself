"""
NAN-Agent 实时仪表盘

提供终端实时仪表盘渲染，展示神经调节剂状态、GoT（Graph of Thoughts）引擎
指标、最近事件流等关键运行时信息。基于 Rich 库实现终端 UI。

主要组件：
- DashboardSnapshot: 仪表盘数据快照，携带所有需要渲染的指标
- EventRingBuffer: 线程安全的事件环形缓冲区
- DashboardRenderer: 仪表盘渲染器，整合各面板并输出到终端
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.layout import Layout
from rich.console import Console


@dataclass
class DashboardSnapshot:
    """仪表盘数据快照，保存某一时刻的完整运行时状态。

    Attributes:
        neuromodulators: 神经调节剂名称 → 浓度（0.0~1.0）
        valence: 情感效价（0.0~1.0，越高越积极）
        arousal: 情感唤醒度（0.0~1.0，越高越兴奋）
        got_nodes: GoT 引擎总节点数
        got_pool: GoT 活跃节点池大小
        got_steps: GoT 已执行步骤数
        got_dmn: DMN（默认模式网络）生成的节点数
        got_pruned: 已剪枝节点数
        got_actions: 待执行动作队列大小
        got_recent_nodes: 最近活跃节点内容摘要列表
        memory_total: 硬记忆总条目数
        timestamp: 快照时间戳
    """
    neuromodulators: dict[str, float] = field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0
    got_nodes: int = 0
    got_pool: int = 0
    got_steps: int = 0
    got_dmn: int = 0
    got_pruned: int = 0
    got_actions: int = 0
    got_recent_nodes: list[str] = field(default_factory=list)
    memory_total: int = 0
    timestamp: float = field(default_factory=time.time)


class EventRingBuffer:
    """线程安全的事件环形缓冲区，用于存储最近的系统事件。

    Args:
        max_size: 最大事件数量，超出后自动丢弃最旧的事件
    """
    def __init__(self, max_size: int = 100):
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, event_name: str, detail: str = "") -> None:
        """追加一条事件（线程安全）。"""
        entry = {
            "timestamp": time.time(),
            "event": event_name,
            "detail": detail[:120],
        }
        with self._lock:
            self._buffer.append(entry)

    def get_since(self, seconds: float) -> list[dict]:
        """获取最近 N 秒内的事件，按时间倒序返回。"""
        cutoff = time.time() - seconds
        with self._lock:
            items = [e for e in self._buffer if e["timestamp"] >= cutoff]
        return list(reversed(items))


def build_neuromodulator_panel(snapshot: DashboardSnapshot) -> Panel:
    """构建神经调节剂面板，显示各神经调节剂浓度条形图及效价/唤醒度。

    面板布局：
    - 上半部分：各神经调节剂的名称、浓度条形图和数值
      - 条形图颜色根据浓度自动变化：>0.7 绿色、>0.4 黄色、>0.2 暗黄色、≤0.2 暗色
      - 最多显示 15 个调节剂，按浓度降序排列
    - 下半部分：情感效价（Valence）和唤醒度（Arousal）数值
      - 效价 >0.5 绿色（积极），≤0.5 红色（消极）
      - 唤醒度 >0.5 黄色（高唤醒），≤0.5 暗色（低唤醒）

    Args:
        snapshot: 仪表盘数据快照，包含 neuromodulators、valence、arousal 等字段
    """
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", width=6)
    table.add_column(width=18)
    table.add_column(style="dim", width=6, justify="right")

    sorted_nms = sorted(
        snapshot.neuromodulators.items(), key=lambda x: x[1], reverse=True
    )

    for name, conc in sorted_nms[:15]:
        bar_len = int(conc * 18)
        color = _bar_color(conc)
        bar = Text("█" * bar_len, style=color)
        table.add_row(name.upper()[:6], bar, f"{conc:.2f}")

    va_text = Text()
    va_text.append("Valence: ", style="bold")
    va_text.append(f"{snapshot.valence:.2f}", style="green" if snapshot.valence > 0.5 else "red")
    va_text.append("  Arousal: ", style="bold")
    va_text.append(f"{snapshot.arousal:.2f}", style="yellow" if snapshot.arousal > 0.5 else "dim")

    content = Table.grid()
    content.add_row(table)
    content.add_row()
    content.add_row(va_text)

    return Panel(content, title="[bold]Neuromodulators[/]", border_style="blue")


def _bar_color(concentration: float) -> str:
    """根据浓度值返回对应的 Rich 颜色样式。"""
    if concentration > 0.7:
        return "green"
    elif concentration > 0.4:
        return "yellow"
    elif concentration > 0.2:
        return "dim yellow"
    return "dim"


def build_got_panel(snapshot: DashboardSnapshot) -> Panel:
    """构建 GoT 引擎面板，显示节点统计、DMN 生成、剪枝和最近活跃节点。

    面板布局：
    - 上半部分：2×3 网格统计表
      - Total Nodes / DMN Generated
      - Active Pool / Actions Queued
      - Pruned / Steps Run
    - 下半部分：最近活跃节点列表（最多 3 个，每个截断至 60 字符）
    """
    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_column(style="cyan")
    stats_table.add_column(justify="right")
    stats_table.add_column(style="cyan")
    stats_table.add_column(justify="right")

    stats_table.add_row("Total Nodes", str(snapshot.got_nodes), "DMN Generated", str(snapshot.got_dmn))
    stats_table.add_row("Active Pool", str(snapshot.got_pool), "Actions Queued", str(snapshot.got_actions))
    stats_table.add_row("Pruned", str(snapshot.got_pruned), "Steps Run", str(snapshot.got_steps))

    content = Table.grid(padding=(0, 1))
    content.add_row(stats_table)

    if snapshot.got_recent_nodes:
        content.add_row()
        content.add_row(Text("Recent Nodes:", style="bold dim"))
        for node_text in snapshot.got_recent_nodes[:3]:
            content.add_row(Text(f"  {node_text[:60]}", style="dim"))

    return Panel(content, title="[bold]GoT Engine[/]", border_style="magenta")


def build_event_timeline(events: list[dict]) -> Panel:
    """构建事件时间线面板，显示最近 30 秒内的事件列表。

    面板以三列表格展示：时间（HH:MM:SS）、事件名称、详情（截断至 50 字符）。
    最多显示 5 条事件，无事件时显示占位提示。
    """
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Time", width=10, style="dim")
    table.add_column("Event", width=28)
    table.add_column("Detail", style="dim")

    for e in events[:5]:
        ts = datetime.fromtimestamp(e["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
        table.add_row(ts, e["event"], e.get("detail", "")[:50])

    if not events:
        table.add_row("--", "No events yet", "--")

    return Panel(table, title="[bold]Events[/] (last 30s)", border_style="green")


class DashboardRenderer:
    """仪表盘渲染器，负责整合各面板并输出到终端。

    通过 update_snapshot() 更新数据，render() 将当前状态渲染为 Rich 布局。
    """

    def __init__(self):
        self._console = Console()
        self._snapshot = DashboardSnapshot()
        self._events = EventRingBuffer(max_size=100)
        self._last_output: str = ""

    @property
    def snapshot(self) -> DashboardSnapshot:
        """当前仪表盘数据快照。"""
        return self._snapshot

    @property
    def events(self) -> EventRingBuffer:
        """事件环形缓冲区。"""
        return self._events

    def update_snapshot(
        self,
        neuromodulators: dict[str, float],
        valence: float,
        arousal: float,
        got_stats,
        got_recent_nodes: list[str],
        memory_total: int = 0,
    ) -> None:
        """更新仪表盘数据快照。

        从 Agent 各子系统提取最新指标，构建新的 DashboardSnapshot 实例。
        got_stats 参数期望为具有以下属性的对象：
        total_nodes, pool_size, steps_run, dmn_generated, pruned_nodes, action_queue_size

        Args:
            neuromodulators: 神经调节剂名称 → 浓度映射
            valence: 情感效价（0.0~1.0）
            arousal: 情感唤醒度（0.0~1.0）
            got_stats: GoT 引擎统计对象（可为 None）
            got_recent_nodes: 最近活跃节点内容摘要列表
            memory_total: 硬记忆总条目数
        """
        got_nodes = got_stats.total_nodes if got_stats else 0
        got_pool = got_stats.pool_size if got_stats else 0
        got_steps = got_stats.steps_run if got_stats else 0
        got_dmn = got_stats.dmn_generated if got_stats else 0
        got_pruned = got_stats.pruned_nodes if got_stats else 0
        got_actions = got_stats.action_queue_size if got_stats else 0

        self._snapshot = DashboardSnapshot(
            neuromodulators=neuromodulators,
            valence=valence,
            arousal=arousal,
            got_nodes=got_nodes,
            got_pool=got_pool,
            got_steps=got_steps,
            got_dmn=got_dmn,
            got_pruned=got_pruned,
            got_actions=got_actions,
            got_recent_nodes=got_recent_nodes,
            memory_total=memory_total,
        )

    def set_last_output(self, text: str) -> None:
        """设置最近一次输出文本，显示在仪表盘头部。"""
        self._last_output = text

    def render(self) -> None:
        """渲染仪表盘到终端。

        布局结构：
        ┌─────────────────────────────────────────────────┐
        │ header: NAN-Agent v2.0 + 最近输出摘要          │
        ├──────────────────────┬──────────────────────────┤
        │ left:                │ right:                   │
        │ Neuromodulators 面板 │ GoT Engine 面板          │
        ├──────────────────────┴──────────────────────────┤
        │ events: 事件时间线（最近 30 秒）                │
        └─────────────────────────────────────────────────┘
        """
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="events", size=8),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        header_text = Text()
        header_text.append("NAN-Agent v2.0", style="bold white on blue")
        if self._last_output:
            header_text.append(f"  {self._last_output[:80]}", style="dim")
        layout["header"].update(Panel(header_text, border_style="blue"))

        layout["left"].update(build_neuromodulator_panel(self._snapshot))
        layout["right"].update(build_got_panel(self._snapshot))
        layout["events"].update(
            build_event_timeline(self._events.get_since(30))
        )

        self._console.clear()
        self._console.print(layout)
