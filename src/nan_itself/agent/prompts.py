"""
System prompt assembly.

Structural sections are XML-tagged containers; a section whose
body is empty is omitted entirely. Message-stream labels (like
[Subagent Report]) are NOT owned here — they live beside the code
that emits those messages.

Tags:
    <skills>             active protocol + switchable catalog
                         (subagents only)
    <running_subagents>  this execution's still-running children,
                         rebuilt every step
    <module>             background Module observations (ambient
                         context; the future memory module will
                         surface here exactly the same way)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.llm import (
    Message,
)

if TYPE_CHECKING:
    from .model import (
        ChildSubagent,
    )
    from ..skills import (
        Skill,
    )


SKILLS_TAG = "skills"

MODULE_TAG = "module"

RUNNING_SUBAGENTS_TAG = "running_subagents"

# RUNNING_TASK_PREVIEW_LIMIT = 80

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


def build_messages(
    *,
    persona: str,
    skill_section: str | None,
    ambient_context: list[str],
    current: list[Message],
    running_subagents: str | None = None,
) -> list[Message]:
    system_parts: list[str] = []

    # L1 persona: who the agent is. Always present.
    system_parts.append(
        persona
    )

    # L2 skills (subagents only), then this execution's running
    # children, then ambient Module observations.
    for section in (
        render_section(SKILLS_TAG, skill_section),
        render_section(RUNNING_SUBAGENTS_TAG, running_subagents),
        render_section(
            MODULE_TAG,
            "\n\n".join(
                [_AMBIENT_PREAMBLE, *ambient_context]
            )
            if ambient_context
            else "",
        ),
    ):
        if section:
            system_parts.append(
                "\n\n" + section
            )

    return [
        Message(
            role="system",
            content="\n".join(system_parts),
        ),
        *current,
    ]


def format_skill_section(
    active_skill: "Skill | None",
    catalog_metadatas,
) -> str:
    """
    The <skills> body: what is currently active plus what can be
    switched to.

    The Main Agent never receives this section: Skills do not
    exist in its world.
    """
    catalog = [
        f"- {metadata.name}: {metadata.description}"
        for metadata in catalog_metadatas
    ]

    parts: list[str] = []

    if active_skill is not None:
        parts.append(
            f"Active skill: {active_skill.name}"
        )

        parts.append(active_skill.instructions)

    else:
        parts.append(
            "No skill is currently active."
        )

    if catalog:
        parts.append(
            "Switch at any step by calling "
            "activate_skill with one of these names:"
        )

        parts.extend(catalog)

    return "\n".join(parts)


def render_running_subagents(
    children: list["ChildSubagent"],
) -> str:
    """
    One line per child that has not delivered its report yet.

    Callers pass the post-collection remainder: everything here
    is genuinely still in flight.
    """
    lines: list[str] = []

    for child in children:
        task = child.task

        # if len(task) > RUNNING_TASK_PREVIEW_LIMIT:
        #     task = task[:RUNNING_TASK_PREVIEW_LIMIT] + "..."

        lines.append(
            f"- id: {child.id} | task: {task}"
        )

    return "\n".join(lines)
