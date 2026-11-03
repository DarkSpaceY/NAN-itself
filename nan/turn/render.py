"""
Rendering: the ONLY place a turn becomes model messages.

Deterministic by contract: rendering depends only on the Turn, so
the same Turn yields the same bytes. A frozen turn keeps its
rendered run.
"""

from __future__ import annotations

from dataclasses import replace

from nan.llm.types import Message
from nan.turn.model import INTERRUPTED, Turn


def render_turn(turn: Turn) -> tuple[Message, ...]:
    """
    One turn as its message run: the observation (injections,
    inbox, task), then the assistant's reply or calls, then one
    tool message per call -- paired by call id, so a call with no
    result renders as interrupted. An in-flight turn renders as
    its observation alone.
    """
    messages: list[Message] = [_observation_message(turn)]

    # observed / injected: the model has not answered yet, so the
    # turn renders as its observation alone.
    if turn.phase in ("observed", "injected"):
        return tuple(messages)

    messages.append(
        Message(
            role="assistant",
            content=turn.reply or "",
            thinking=list(turn.thinking),
            tool_calls=list(turn.verb_calls),
        )
    )

    results = {result.call_id: result for result in turn.verb_results}

    for call in turn.verb_calls:
        result = results.get(call.id)

        messages.append(
            Message(
                role="tool",
                tool_call_id=call.id,
                content=result.output if result is not None else INTERRUPTED,
            )
        )

    return tuple(messages)


def freeze_turn(turn: Turn) -> Turn:
    """
    Render once and freeze: the returned Turn carries its
    message run forever.
    """
    if turn.rendered is not None:
        return turn

    return replace(turn, rendered=render_turn(turn))


def _observation_message(turn: Turn) -> Message:
    """
    The unified observation schema. Three ordered sections, each
    omitted entirely when empty:

        <delta>      system state changes since last turn
        <injection>  service contributions this turn
        <inbox>      messages, tagged by the SENDER'S role
    """
    sections = [
        section
        for section in (
            _delta_section(turn),
            _injection_section(turn),
            _inbox_section(turn),
        )
        if section is not None
    ]

    return Message(
        role="user",
        content="\n\n".join(sections) or "(no input this turn)",
    )


def _delta_section(turn: Turn) -> str | None:
    """System state changes, grouped by section in first-seen order.
    A pure function of turn.delta: same list, same bytes."""
    grouped: dict[str, list[str]] = {}

    for event in turn.delta:
        grouped.setdefault(event.section, []).append(event.line)

    if not grouped:
        return None

    inner = [f"<{section}>{'\n'.join(lines)}</{section}>" for section, lines in grouped.items()]

    return "<delta>\n" + "\n".join(inner) + "\n</delta>"


def _injection_section(turn: Turn) -> str | None:
    """Service contributions; a failed service keeps its error."""
    if not turn.injections:
        return None

    services: list[str] = []

    for injection in turn.injections:
        body = injection.text if injection.ok else f"[service error] {injection.text}"

        services.append(f'<service id="{injection.service_id}">\n{body}\n</service>')

    return "<injection>\n" + "\n\n".join(services) + "\n</injection>"


def _inbox_section(turn: Turn) -> str | None:
    """Inbox messages, tagged by the sender's role."""
    if not turn.inbox:
        return None

    messages: list[str] = []

    for message in turn.inbox:
        attrs = f' id="{message.source}"' if message.kind == "subagent" else ""

        if message.event != "send":
            attrs += f' event="{message.event}"'

        messages.append(f"<{message.kind}{attrs}>\n{message.text}\n</{message.kind}>")

    return "<inbox>\n" + "\n\n".join(messages) + "\n</inbox>"
