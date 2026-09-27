"""
Subagent report formatting.

When a spawned Subagent finishes, its final report is formatted
here. Delivery is exactly one level up: every agent harvests its
finished children's reports at its own next turn boundary and
folds them into that turn's observation. A child that has not
reported by the time its parent finishes loses its report
(dropped with a warning).
"""

from __future__ import annotations


REPORT_TAG = "subagent_report"


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


def format_child_report(
    *,
    agent_id: str,
    task: str | None,
    status: str,
    body: str,
) -> str:
    """
    Wrap one finished agent's report for delivery to its parent.

    `status` is "completed" with the report body as `body`, or
    "failed" with `body="error: <reason>"`.
    """
    return (
        f"<{REPORT_TAG}>\n"
        f"id: {agent_id}\n"
        f"task: {task}\n"
        f"status: {status}\n"
        f"{body}\n"
        f"</{REPORT_TAG}>"
    )
