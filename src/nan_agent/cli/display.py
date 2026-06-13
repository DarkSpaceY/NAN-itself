"""
NAN-Agent 显示格式化工具

提供静态格式化方法，用于将各种数据结构转换为终端友好的字符串输出。
支持 JSON、表格、列表、状态面板、代码块、进度条、错误/警告/成功提示等格式。
"""

import json
from typing import Any


class Display:
    """终端显示格式化工具类，所有方法均为静态方法。

    提供以下格式化能力：
    - JSON 缩进格式化
    - 文本表格（自动列宽对齐）
    - 项目符号列表
    - 状态面板（带标题和分隔线）
    - Markdown 代码块
    - 错误/警告/成功提示（带 Unicode 符号标记）
    - 进度条（30 字符宽度，百分比显示）
    - 文本截断（超出部分用 "..." 替代）
    """

    @staticmethod
    def format_json(data: Any) -> str:
        """将数据格式化为缩进的 JSON 字符串。"""
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def format_table(headers: list[str], rows: list[list[str]]) -> str:
        """将表头和行数据格式化为对齐的文本表格。

        列宽自动根据表头和各列最长单元格内容计算，列间用 " | " 分隔，
        表头下方添加分隔线。
        """
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        lines = []
        header_line = " | ".join(
            h.ljust(col_widths[i]) for i, h in enumerate(headers)
        )
        lines.append(header_line)
        lines.append("-" * len(header_line))

        for row in rows:
            padded = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    padded.append(str(cell).ljust(col_widths[i]))
                else:
                    padded.append(str(cell))
            lines.append(" | ".join(padded))

        return "\n".join(lines)

    @staticmethod
    def format_list(items: list[str], bullet: str = "\u2022") -> str:
        """将字符串列表格式化为带项目符号的列表。"""
        return "\n".join(f"{bullet} {item}" for item in items)

    @staticmethod
    def format_status_panel(status: dict, title: str = "Status") -> str:
        """将状态字典格式化为带标题的状态面板。"""
        lines = ["", "=" * 60, f"  {title}", "=" * 60]
        max_key_len = max(len(k) for k in status.keys()) if status else 0
        for key, value in status.items():
            lines.append(f"  {key.ljust(max_key_len + 2)} {value}")
        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def format_code(code: str, language: str = "") -> str:
        """将代码包装为 Markdown 代码块格式。"""
        return f"```{language}\n{code}\n```"

    @staticmethod
    def format_error(error: str) -> str:
        """将错误信息格式化为带 ✗ 标记的错误提示。"""
        return f"\n  \u2717 Error: {error}\n"

    @staticmethod
    def format_warning(warning: str) -> str:
        """将警告信息格式化为带 ⚠ 标记的警告提示。"""
        return f"\n  \u26a0 Warning: {warning}\n"

    @staticmethod
    def format_success(message: str) -> str:
        """将成功信息格式化为带 ✓ 标记的成功提示。"""
        return f"\n  \u2713 {message}\n"

    @staticmethod
    def format_progress(current: int, total: int, label: str = "") -> str:
        """格式化进度条字符串，形如 [====>     ] 50%。

        进度条宽度固定 30 字符，current/total 比值映射到填充长度。
        比值被钳制在 [0, 1] 范围内，避免越界。
        """
        bar_width = 30
        ratio = min(max(current / max(total, 1), 0), 1)
        filled = int(ratio * bar_width)
        bar = "=" * filled + ">" + " " * max(bar_width - filled - 1, 0)
        pct = int(ratio * 100)
        label_part = f" {label} " if label else " "
        return f"[{bar}]{label_part}{pct}%"

    @staticmethod
    def truncate(text: str, max_len: int = 200) -> str:
        """截断文本到指定长度，超出部分用 "..." 替代。"""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."


NANDisplay = Display  # 向后兼容别名，保留旧代码的 NANDisplay 引用