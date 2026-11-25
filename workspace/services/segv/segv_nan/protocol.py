"""NAN payloads -> the core's turn record and inject query.

Everything protocol-specific lives here: the rendered run, the injection
blocks embedded in user observations, the inbox that seeds recall, and
the llm counters the ledger shows.
"""

from __future__ import annotations

import re
from typing import Any

from segv.record import TurnRecord

_INJECTION_RE = re.compile(r"<injection(?:\s[^>]*)?>.*?</injection>", re.DOTALL)


def strip_injections(text: str) -> str:
    """Remove every injected service block from a user observation."""
    return _INJECTION_RE.sub("", text).strip()


def messages(rendered: Any) -> list[dict[str, Any]]:
    """The message run, with injections removed from user text."""
    out: list[dict[str, Any]] = []

    if not isinstance(rendered, list):
        return out

    for message in rendered:
        if not isinstance(message, dict):
            continue

        item = dict(message)

        content = item.get("content")

        if item.get("role") == "user" and isinstance(content, str):
            item["content"] = strip_injections(content)

        out.append(item)

    return out


def _int(value: Any) -> int:
    try:
        return int(value or 0)

    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def _headline(body: dict[str, Any]) -> str:
    reply = str(body.get("reply") or "")

    headline = reply.strip().splitlines()[0] if reply.strip() else ""

    if not headline:
        inbox = body.get("inbox") or []

        if inbox and isinstance(inbox[-1], dict):
            headline = str(inbox[-1].get("text") or "")

    return headline[:160]


def turn_record(body: dict[str, Any]) -> TurnRecord | None:
    """Map one NAN turn; None when it has no usable identity."""
    agent_id = str(body.get("agent_id") or "")

    started_at = _float(body.get("started_at"))

    if not agent_id or started_at is None:
        return None

    llm = body.get("llm") or {}

    usage = llm.get("usage") or {}

    results = body.get("results") or []

    thinking = body.get("thinking") or []

    reply = str(body.get("reply") or "")

    death = body.get("death")

    death_reason = None

    if isinstance(death, dict):
        death_reason = str(death.get("reason") or "") or None

    return TurnRecord(
        agent_id=agent_id,
        started_at=started_at,
        ended_at=_float(body.get("ended_at")),
        messages=messages(body.get("rendered")),
        summary=_headline(body),
        stats={
            "tokens": _int(usage.get("prompt_tokens"))
            + _int(usage.get("completion_tokens")),
            "retries": _int(llm.get("retries")),
            "calls": len(body.get("calls") or []),
            "errors": sum(
                1 for item in results if isinstance(item, dict) and not item.get("output")
            ),
            "reply_chars": len(reply),
            "thinking_chars": sum(
                len(str(block.get("text") or ""))
                for block in thinking
                if isinstance(block, dict)
            ),
        },
        epoch=_int(body.get("epoch")),
        depth=_int(body.get("depth")),
        parent_id=body.get("parent_id"),
        origin=str(body.get("origin") or "normal"),
        death=death_reason,
        llm_started_at=_float(llm.get("started_at")),
        llm_ended_at=_float(llm.get("ended_at")),
        raw=body,
    )


def inject_request(body: dict[str, Any]) -> tuple[str, int, str]:
    """The cache's inputs: agent, epoch, and the inbox text as the query."""
    agent_id = str(body.get("agent_id") or "").strip()

    epoch = _int(body.get("epoch"))

    parts = [
        str(message.get("text") or "").strip()
        for message in body.get("inbox") or []
        if isinstance(message, dict)
    ]

    query = "\n".join(part for part in parts if part)[-2000:].strip()

    return agent_id, epoch, query
