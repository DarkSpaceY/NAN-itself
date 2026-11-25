"""Turn -> chunks.

A chunk is the unit fact stores: two texts, two vectors, one pointer.
Ingestion prefers the whole turn in one chunk. Only when the message run
does not fit do we cut, in this order: message elements, top-level
sections, inner list elements, then paragraphs and sentences.

Messages arrive already cleaned by the adapter: injected service blocks
and system messages never reach this layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import ChunkConfig
from .text import SENTENCE_RE

ROLES = ("user", "assistant", "tool")

_TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w:.-]*)((?:\s[^<>]*?)?)(/?)\s*>")


@dataclass
class Part:
    """One message element, or one piece of a long element."""

    role: str
    body: str
    header: str

    @property
    def text(self) -> str:
        return self.header + self.body


@dataclass
class ChunkDraft:
    parts: list[Part] = field(default_factory=list)
    cue: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(part.text for part in self.parts)

    @property
    def roles(self) -> list[str]:
        return [part.role for part in self.parts]


def iso_timestamp(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = 0.0

    return datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_sections(text: str) -> list[str]:
    """Split at top-level XML-ish blocks; whitespace between blocks drops."""
    pieces: list[str] = []
    depth = 0
    start = 0

    for match in _TAG_RE.finditer(text):
        closing = bool(match.group(1))
        self_closing = bool(match.group(4))

        if depth == 0:
            if match.start() > start and text[start : match.start()].strip():
                pieces.append(text[start : match.start()])

            if closing or self_closing:
                pieces.append(match.group(0))
                start = match.end()
                continue

            depth = 1
            start = match.start()

        elif self_closing:
            continue

        elif closing:
            depth -= 1

            if depth == 0:
                pieces.append(text[start : match.end()])
                start = match.end()

        else:
            depth += 1

    if depth != 0:
        # Unbalanced markup: the tag split is unsafe, keep the text whole.
        return [text] if text.strip() else []

    if start < len(text) and text[start:].strip():
        pieces.append(text[start:])

    return pieces


def _unwrap(block: str) -> str | None:
    match = _TAG_RE.match(block)

    if match is None or match.group(1) or match.group(4):
        return None

    close = f"</{match.group(2)}>"
    stripped = block.rstrip()

    if not stripped.endswith(close):
        return None

    return stripped[match.end() : len(stripped) - len(close)]


def _split_plain(text: str, limit: int) -> list[str]:
    out: list[str] = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()

        if paragraph:
            out.extend(_split_sentences(paragraph, limit))

    return out


def _split_sentences(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    out: list[str] = []
    buffer = ""

    for sentence in SENTENCE_RE.split(text):
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > limit:
            if buffer:
                out.append(buffer)
                buffer = ""

            out.extend(sentence[i : i + limit] for i in range(0, len(sentence), limit))

        elif len(buffer) + len(sentence) + 1 > limit:
            out.append(buffer)
            buffer = sentence

        else:
            buffer = f"{buffer} {sentence}".strip() if buffer else sentence

    if buffer:
        out.append(buffer)

    return out


def _split_block(block: str, limit: int) -> list[str]:
    """Cut one section: inner list elements first, then paragraphs/sentences."""
    if len(block) <= limit:
        return [block]

    inner = _unwrap(block)

    if inner is None:
        return _split_plain(block, limit)

    children = split_sections(inner)

    if len(children) > 1:
        out: list[str] = []

        for child in children:
            out.extend(_split_block(child, limit))

        return out

    return _split_plain(inner, limit)


def _split_message(body: str, limit: int) -> list[str]:
    if len(body) <= limit:
        return [body]

    sections = split_sections(body)

    if len(sections) > 1:
        out: list[str] = []

        for section in sections:
            out.extend(_split_block(section, limit))

        return out

    return _split_block(body, limit)


def _call_line(call: dict[str, Any]) -> str:
    name = str(call.get("name") or "")

    arguments = call.get("arguments")

    if arguments is None:
        function = call.get("function") or {}

        name = str(function.get("name") or name)
        arguments = function.get("arguments")

    if isinstance(arguments, str):
        rendered = arguments
    else:
        rendered = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)

    return f"[tool_calls] {name}({rendered})"


def _message_body(message: dict[str, Any], config: ChunkConfig) -> str:
    role = str(message.get("role") or "")

    if role not in ROLES:
        return ""

    content = message.get("content")

    if not isinstance(content, str):
        content = "" if content is None else str(content)

    lines = [content.strip()] if content.strip() else []

    if role == "assistant" and config.include_tool_calls:
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                lines.append(_call_line(call))

    return "\n".join(line for line in lines if line).strip()


def _pack(parts: list[Part], max_chars: int) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    current: list[Part] = []
    size = 0

    for part in parts:
        extra = len(part.text) + (2 if current else 0)

        if current and size + extra > max_chars:
            chunks.append(ChunkDraft(current))
            current = []
            size = 0
            extra = len(part.text)

        current.append(part)
        size += extra

    if current:
        chunks.append(ChunkDraft(current))

    return chunks


def build_chunks(
    record: dict[str, Any], config: ChunkConfig
) -> tuple[str, list[ChunkDraft]]:
    """Build the ordered chunk drafts for one archived turn."""
    ts = iso_timestamp(record.get("started_at"))
    messages = record.get("messages") or []

    if not isinstance(messages, list):
        return ts, []

    parts: list[Part] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "")

        body = _message_body(message, config)

        if not body:
            continue

        header = f"[{ts}] {role}: "
        limit = max(1, config.max_chars - len(header))

        for piece in _split_message(body, limit):
            piece = piece.strip()

            if piece:
                parts.append(Part(role=role, body=piece, header=header))

    return ts, _pack(parts, config.max_chars)
