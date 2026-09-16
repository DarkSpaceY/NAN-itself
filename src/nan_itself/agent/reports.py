"""
Subagent report formatting.

When a spawned Subagent finishes, its final report is formatted
here and parked into the Inbox by the engine, so the next turn's
observation carries it.
"""

from __future__ import annotations

import asyncio

from .model import (
    ChildSubagent,
)


REPORT_PREFIX = "[Subagent Report]"


def report_record_name(
    report: str,
) -> str:
    """
    UI record name for a report: 'report · <task>' when the
    report text carries a task line, else 'report'.
    """
    for line in report.splitlines():
        if line.startswith("task: "):
            return f"report · {line[6:]}"

    return "report"


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

    return (
        f"{REPORT_PREFIX}\n"
        f"id: {child.id}\n"
        f"task: {child.task}\n"
        + body
    )
