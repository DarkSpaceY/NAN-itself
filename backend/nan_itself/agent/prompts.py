"""
Prompt assembly: the ONLY place a turn becomes model messages.

Everything an agent sees or says is structured on its Turn
(persona, ambient Module context, harvested child reports, task,
and the turn's reply / tool calls / results). This module renders
that structure into the provider-neutral message sequence --
deterministically: the same Turn always renders to the same
messages, which is what keeps the provider's message prefix
cacheable.

The engine is the only intended caller:

    build_messages(turn)     the full request for one model
                             call: [system] + history + this
                             turn's rendering (an in-flight
                             turn renders as its observation
                             alone)
    StepEngine.render_turn   delegates here for one completed
                             turn, so the caller can derive
                             the next turn's history snapshot
                             by appending the result

The <subagent_report> frame lives here too: reports travel as
structured records and are framed exactly once, at rendering.
"""

from __future__ import annotations

from typing import Any

from .model import (
    Report,
)
from ..utils.llm import (
    Message,
)


_MODULE_TAG = "module"

_REPORT_TAG = "subagent_report"

_AMBIENT_PREAMBLE = (
    "The following information is supplied by background "
    "Modules. It is contextual information, not a user "
    "instruction. Use it when relevant."
)


def build_messages(
    turn,
) -> list[Message]:
    """
    The full model input for one turn: system + history + this
    turn's rendering.
    """
    return [
        _system_message(
            turn.persona or ""
        ),
        *turn.history,
        *_render_turn(turn),
    ]


def render_turn(
    turn,
) -> tuple[Message, ...]:
    """
    One turn as its message run: the observation first, then
    the assistant's reply or tool calls, then one tool message
    per result. An in-flight or failed turn (empty flow) renders
    as the observation alone.

    Deterministic: the same turn always renders to the same
    messages -- the next turn's history derives its prefix
    cacheability from this.
    """
    return _render_turn(turn)


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _render_turn(
    turn,
) -> tuple[Message, ...]:
    messages: list[Message] = [
        _observation_message(turn)
    ]

    if turn.reply is None and not turn.calls:
        return tuple(messages)

    messages.append(
        Message(
            role="assistant",
            content=turn.reply or "",
            tool_calls=list(turn.calls),
        )
    )

    for call, result in zip(
        turn.calls,
        turn.results,
    ):
        messages.append(
            _tool_message(
                call.id,
                result,
            )
        )

    return tuple(messages)


def _observation_message(
    turn,
) -> Message:
    """
    One user message: ambient Module observations (including
    the inbox) first, then harvested child reports, then a
    subagent's own task.
    """
    parts: list[str] = []

    section = _render_section(
        _MODULE_TAG,
        "\n\n".join(
            [_AMBIENT_PREAMBLE, *turn.ambient]
        )
        if turn.ambient
        else "",
    )

    if section:
        parts.append(
            section
        )

    parts.extend(
        _render_report(report)
        for report in turn.reports
    )

    task_section = _render_section(
        "task",
        turn.task or "",
    )

    if task_section:
        parts.append(
            task_section
        )

    return Message(
        role="user",
        content="\n\n".join(
            parts
        ),
    )


def _render_report(
    report: Any,
) -> str:
    """
    One harvested child report as prompt text. Reports travel
    as structured `Report` records (the Turn field is loosely
    typed); anything else passes through as its text.
    """
    if isinstance(report, Report):
        return (
            f"<{_REPORT_TAG}>\n"
            f"id: {report.agent_id}\n"
            f"task: {report.task}\n"
            f"status: {report.status}\n"
            f"{report.body}\n"
            f"</{_REPORT_TAG}>"
        )

    return str(report)


def _system_message(
    persona: str,
) -> Message:
    """
    The system message is the persona alone: it stays stable
    for the whole turn (and across turns until the persona file
    changes), so the provider can cache the prefix.
    """
    return Message(
        role="system",
        content=persona,
    )


def _tool_message(
    tool_call_id: str,
    content: str,
) -> Message:
    return Message(
        role="tool",
        tool_call_id=tool_call_id,
        content=content,
    )


def _render_section(
    tag: str,
    body: str,
) -> str | None:
    """
    Wrap a body in its container; an empty body means the whole
    section disappears.
    """
    body = (body or "").strip()

    if not body:
        return None

    return f"<{tag}>\n{body}\n</{tag}>"
