"""The snapshot: five stable lists plus the current goal.

The model returns ids; the service owns them. An unchanged item keeps its
id and its exact text, a new item arrives with an empty id and gets the
next free one, and an unknown or duplicated id is treated as new rather
than trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

PREFIXES = {
    "unresolved_items": "u",
    "locked_decisions": "d",
    "forbidden_behaviors": "f",
    "entity_map": "e",
}

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", re.DOTALL)


class SnapshotError(ValueError):
    """The model's reply is not a usable snapshot."""


@dataclass
class Item:
    id: str
    text: str

    def to_json(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text}


@dataclass
class Entity:
    id: str
    name: str
    meaning: str

    def to_json(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "meaning": self.meaning}


@dataclass
class Snapshot:
    goal: str = ""
    unresolved_items: list[Item] = field(default_factory=list)
    locked_decisions: list[Item] = field(default_factory=list)
    forbidden_behaviors: list[Item] = field(default_factory=list)
    entity_map: list[Entity] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "unresolved_items": [item.to_json() for item in self.unresolved_items],
            "locked_decisions": [item.to_json() for item in self.locked_decisions],
            "forbidden_behaviors": [
                item.to_json() for item in self.forbidden_behaviors
            ],
            "entity_map": [entity.to_json() for entity in self.entity_map],
        }

    @classmethod
    def from_json(cls, raw: Any) -> Snapshot:
        if not isinstance(raw, dict):
            return cls()

        def items(key: str) -> list[Item]:
            out: list[Item] = []

            for entry in raw.get(key) or []:
                if not isinstance(entry, dict):
                    continue

                text = str(entry.get("text") or "").strip()
                item_id = str(entry.get("id") or "").strip()

                if text:
                    out.append(Item(id=item_id, text=text))

            return out

        entities: list[Entity] = []

        for entry in raw.get("entity_map") or []:
            if not isinstance(entry, dict):
                continue

            name = str(entry.get("name") or "").strip()

            if name:
                entities.append(
                    Entity(
                        id=str(entry.get("id") or "").strip(),
                        name=name,
                        meaning=str(entry.get("meaning") or "").strip(),
                    )
                )

        return cls(
            goal=str(raw.get("goal") or "").strip(),
            unresolved_items=items("unresolved_items"),
            locked_decisions=items("locked_decisions"),
            forbidden_behaviors=items("forbidden_behaviors"),
            entity_map=entities,
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.goal
            or self.unresolved_items
            or self.locked_decisions
            or self.forbidden_behaviors
            or self.entity_map
        )

    def render(self) -> str:
        """The block the model receives at an epoch boundary."""
        blocks: list[str] = []

        if self.goal:
            blocks.append(f"<goal>\n{self.goal}\n</goal>")

        if self.unresolved_items:
            blocks.append(_render_items("unresolved_items", self.unresolved_items))

        if self.locked_decisions:
            blocks.append(_render_items("locked_decisions", self.locked_decisions))

        if self.forbidden_behaviors:
            blocks.append(_render_items("forbidden_behaviors", self.forbidden_behaviors))

        if self.entity_map:
            lines = [
                f"- [{entity.id}] {entity.name}"
                + (f" — {entity.meaning}" if entity.meaning else "")
                for entity in self.entity_map
            ]

            blocks.append("<entity_map>\n" + "\n".join(lines) + "\n</entity_map>")

        if not blocks:
            return ""

        return "<memory>\n" + "\n".join(blocks) + "\n</memory>"


def _render_items(section: str, items: list[Item]) -> str:
    lines = [f"- [{item.id}] {item.text}" for item in items]

    return f"<{section}>\n" + "\n".join(lines) + f"\n</{section}>"


def _decode(raw: str) -> dict[str, Any]:
    text = _FENCE_RE.sub("", raw.strip())
    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise SnapshotError("no JSON object in the reply")

    try:
        value = json.loads(text[start : end + 1])

    except json.JSONDecodeError as exc:
        raise SnapshotError(f"invalid JSON: {exc.msg}") from exc

    if not isinstance(value, dict):
        raise SnapshotError("the reply is not a JSON object")

    return value


def _suffix(item_id: str) -> int:
    digits = item_id[1:]

    return int(digits) if digits.isdigit() else 0


def _text(raw: Any) -> tuple[str, str]:
    """(id, text) for one unresolved / decision / forbidden entry."""
    if isinstance(raw, str):
        return "", raw.strip()

    if not isinstance(raw, dict):
        return "", ""

    value = raw.get("text") or raw.get("item") or raw.get("value") or ""

    return str(raw.get("id") or "").strip(), str(value).strip()


def _entity(raw: Any) -> tuple[str, str, str]:
    """(id, name, meaning) for one entity-map entry."""
    if isinstance(raw, str):
        return "", raw.strip(), ""

    if not isinstance(raw, dict):
        return "", "", ""

    name = raw.get("name") or raw.get("term") or raw.get("text") or ""
    meaning = raw.get("meaning") or raw.get("value") or ""

    return str(raw.get("id") or "").strip(), str(name).strip(), str(meaning).strip()


def _assign_items(
    previous: list[Item], raw_items: Any, prefix: str, max_items: int
) -> list[Item]:
    previous_by_id = {entry.id: entry for entry in previous}
    previous_by_text = {entry.text: entry.id for entry in previous}
    counter = max((_suffix(entry.id) for entry in previous), default=0)
    used_ids: set[str] = set()
    used_texts: set[str] = set()
    out: list[Item] = []

    for raw in raw_items or []:
        item_id, text = _text(raw)

        if not text or text in used_texts:
            continue

        candidate = item_id if item_id in previous_by_id and item_id not in used_ids else ""

        if not candidate:
            candidate = previous_by_text.get(text, "")

        if not candidate or candidate in used_ids:
            counter += 1
            candidate = f"{prefix}{counter}"

        used_ids.add(candidate)
        used_texts.add(text)
        out.append(Item(id=candidate, text=text))

        if len(out) >= max_items:
            break

    return out


def _assign_entities(previous: list[Entity], raw_items: Any, max_items: int) -> list[Entity]:
    previous_by_id = {entry.id: entry for entry in previous}
    previous_by_name = {entry.name: entry.id for entry in previous}
    counter = max((_suffix(entry.id) for entry in previous), default=0)
    used_ids: set[str] = set()
    used_names: set[str] = set()
    out: list[Entity] = []

    for raw in raw_items or []:
        item_id, name, meaning = _entity(raw)

        if not name or name in used_names:
            continue

        candidate = item_id if item_id in previous_by_id and item_id not in used_ids else ""

        if not candidate:
            candidate = previous_by_name.get(name, "")

        if not candidate or candidate in used_ids:
            counter += 1
            candidate = f"{PREFIXES['entity_map']}{counter}"

        used_ids.add(candidate)
        used_names.add(name)
        out.append(Entity(id=candidate, name=name, meaning=meaning))

        if len(out) >= max_items:
            break

    return out


def _count(snapshot: Snapshot) -> int:
    return (
        len(snapshot.unresolved_items)
        + len(snapshot.locked_decisions)
        + len(snapshot.forbidden_behaviors)
        + len(snapshot.entity_map)
    )


def parse_snapshot(
    raw: str,
    previous: Snapshot,
    *,
    max_items: int = 50,
    min_retention: float = 0.5,
) -> Snapshot:
    """Validate one rewrite; raise SnapshotError when it cannot be trusted."""
    obj = _decode(raw)
    goal = str(obj.get("goal") or obj.get("current_goal") or "").strip()

    if not goal:
        raise SnapshotError("empty goal")

    snapshot = Snapshot(
        goal=goal,
        unresolved_items=_assign_items(
            previous.unresolved_items,
            obj.get("unresolved_items"),
            PREFIXES["unresolved_items"],
            max_items,
        ),
        locked_decisions=_assign_items(
            previous.locked_decisions,
            obj.get("locked_decisions"),
            PREFIXES["locked_decisions"],
            max_items,
        ),
        forbidden_behaviors=_assign_items(
            previous.forbidden_behaviors,
            obj.get("forbidden_behaviors"),
            PREFIXES["forbidden_behaviors"],
            max_items,
        ),
        entity_map=_assign_entities(
            previous.entity_map, obj.get("entity_map"), max_items
        ),
    )

    before = _count(previous)
    after = _count(snapshot)

    if before >= 4 and after < before * min_retention:
        raise SnapshotError(f"drop rate too high: kept {after} of {before} items")

    return snapshot
