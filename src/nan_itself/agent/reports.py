"""
Subagent report delivery.

One message format (`[Subagent Report]`), two trigger times:

    step-collect : children finished since the previous step are
                   injected immediately, parent keeps reasoning
    late buffer  : parent's turn already ended -> parked here and
                   delivered at the start of the next main turn
"""

from __future__ import annotations

import asyncio
from collections import deque

from .model import (
    ChildSubagent,
)


REPORT_PREFIX = "[Subagent Report]"

# TASK_PREVIEW_LIMIT = 200


class LateReportBuffer:
    """
    Parking lot for reports whose parent turn already ended.
    """

    def __init__(self) -> None:
        self._items: deque[str] = deque()

    def park(self, reports: list[str]) -> None:
        self._items.extend(reports)

    def drain(self) -> list[str]:
        items = list(self._items)

        self._items.clear()

        return items

    def has_pending(self) -> bool:
        return bool(self._items)


async def collect_finished_children(
    children: list[ChildSubagent],
) -> list[str]:
    """
    Format and mark every child that finished since last call.
    """
    reports: list[str] = []

    for child in children:
        if child.reported:
            continue

        if not child.handle.done:
            continue

        child.reported = True

        reports.append(
            await format_child_report(child)
        )

    return reports


async def format_child_report(
    child: ChildSubagent,
) -> str:
    try:
        result = await child.handle.wait()

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        body = (
            "status: failed\n"
            f"error: {exc}"
        )

    else:
        body = (
            "status: completed\n"
            + (result.content or "")
        )

    task_preview = child.task

    # if len(task_preview) > TASK_PREVIEW_LIMIT:
    #     task_preview = (
    #         task_preview[
    #             :TASK_PREVIEW_LIMIT
    #         ]
    #         + "..."
    #     )

    return (
        f"{REPORT_PREFIX}\n"
        f"id: {child.id}\n"
        f"task: {task_preview}\n"
        + body
    )
