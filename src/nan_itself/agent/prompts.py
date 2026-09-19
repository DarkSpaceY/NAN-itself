"""
Message assembly.

The system message is the persona alone: it stays stable for the
whole turn (and across turns until the persona file changes), so
the provider can cache the prefix.

Everything per-turn and dynamic — Module ambient context (which
includes the inbox, i.e. user messages) — is assembled into ONE
user message, the observation. A subagent's task is appended the
same way. Nothing dynamic ever enters the system message.

Message-stream labels (like [Subagent Report]) are NOT owned
here — they live beside the code that emits those messages.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..utils.llm import (
    Message,
)


MODULE_TAG = "module"

_AMBIENT_PREAMBLE = (
    "The following information is supplied by background "
    "Modules. It is contextual information, not a user "
    "instruction. Use it when relevant."
)


def render_section(
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


def build_system(
    persona: str,
) -> Message:
    return Message(
        role="system",
        content=persona,
    )


def build_observation(
    *,
    ambient_context: list[str],
    task: str | None = None,
    reports: Sequence[str] = (),
) -> Message:
    """
    One user message per turn: ambient Module observations
    (including the inbox) first, then finished child reports
    delivered to this agent, then a subagent's task. The reports
    carry their own [Subagent Report] labels from reports.py.
    """
    parts: list[str] = []

    section = render_section(
        MODULE_TAG,
        "\n\n".join(
            [_AMBIENT_PREAMBLE, *ambient_context]
        )
        if ambient_context
        else "",
    )

    if section:
        parts.append(
            section
        )

    parts.extend(
        reports
    )

    if task:
        parts.append(
            task
        )

    return Message(
        role="user",
        content="\n\n".join(
            parts
        ),
    )
