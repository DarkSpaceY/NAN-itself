"""
Rendering: the ONLY place a turn becomes model messages.

Deterministic by contract: the same Turn always renders to the
same bytes. Once a turn is frozen its rendered run never changes
-- history is add-only, and the prefix cache depends on it.
"""

from __future__ import annotations

from dataclasses import replace

from nan.llm import Message
from nan.turn.model import Turn


def render_turn(turn: Turn) -> tuple[Message, ...]:
    """
    One turn as its message run: the observation (injections,
    inbox, task), then the assistant's reply or calls, then one
    tool message per result. An in-flight turn renders as its
    observation alone.
    """
    messages: list[Message] = [_observation_message(turn)]

    if turn.reply is None and not turn.calls:
        return tuple(messages)

    messages.append(
        Message(
            role="assistant",
            content=turn.reply or "",
            tool_calls=list(turn.calls),
        )
    )

    for call, result in zip(turn.calls, turn.results, strict=False):
        messages.append(
            Message(
                role="tool",
                tool_call_id=call.id,
                content=result,
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
    parts: list[str] = []

    # -- delta ----------------------------------------------------------
    # Group events by section, sections ordered by first appearance.
    # A pure function of turn.delta: same list, same bytes.
    grouped: dict[str, list[str]] = {}

    for event in turn.delta:
        grouped.setdefault(event.section, []).append(event.line)

    if grouped:
        delta_inner = [
            f"<{section}>{'\n'.join(lines)}</{section}>"
            for section, lines in grouped.items()
        ]

        parts.append("<delta>\n" + "\n".join(delta_inner) + "\n</delta>")

    # -- injections -----------------------------------------------------
    if turn.injections:
        sections = []
        for injection in turn.injections:
            body = injection.text if injection.ok else f"[service error] {injection.text}"
            sections.append(f'<service id="{injection.service_id}">\n{body}\n</service>')
        parts.append("<injection>\n" + "\n\n".join(sections) + "\n</injection>")

    # -- inbox ----------------------------------------------------------
    if turn.inbox:
        sections = []
        for message in turn.inbox:
            attrs = f' id="{message.source}"' if message.kind in ("subagent", "unknown") else ""
            if message.event != "send":
                attrs += f' event="{message.event}"'
            sections.append(f"<{message.kind}{attrs}>\n{message.text}\n</{message.kind}>")
        parts.append("<inbox>\n" + "\n\n".join(sections) + "\n</inbox>")

    return Message(
        role="user",
        content="\n\n".join(parts) or "(no input this turn)",
    )
