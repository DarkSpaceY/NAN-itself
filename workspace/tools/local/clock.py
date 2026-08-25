# @tool

"""Clock: 第一个 workspace 本地工具示例。保存后自动被 ProviderRuntime 发现。"""

from __future__ import annotations

import datetime

from src.nan_itself.tools.facade import (
    LocalToolProvider,
    tool,
)


class ClockTools(LocalToolProvider):
    id = "clock"

    @tool(description="返回当前本地时间（ISO 格式）。")
    def now(self) -> str:
        return datetime.datetime.now().isoformat(
            timespec="seconds"
        )

    @tool(description="计算两个整数的和。")
    def add(self, a: int, b: int) -> int:
        return a + b
