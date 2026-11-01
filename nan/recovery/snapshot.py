"""
Persistence: one snapshot per agent -- its Turn list, and nothing
else.

A Turn is self-contained (identity, observation, reply, verb
results, exit record), so a list of Turns is the agent's whole
live state. The spawn tree is not stored either: every turn
carries its parent_id, so restore rebuilds the tree from the
files on disk.

The store is whole-state, never append: the list is serialized to
a temp file and atomically replaced over the target, so a crash
leaves either the old snapshot or the new one, never a torn one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nan.turn import Death, InboxMessage, Turn


@dataclass(frozen=True)
class AgentSnapshot:
    """
    One agent's turns, in order. The file is named by its id; the
    id, parent and depth come from the first turn.
    """

    turns: tuple[Turn, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("a snapshot holds at least one turn")

        first = self.turns[0]

        if any(turn.agent_id != first.agent_id for turn in self.turns):
            raise ValueError(f"snapshot mixes agents: expected {first.agent_id}")

    @property
    def agent_id(self) -> str:
        return self.turns[0].agent_id

    @property
    def parent_id(self) -> str | None:
        return self.turns[0].parent_id

    @property
    def depth(self) -> int:
        return self.turns[0].depth

    @property
    def last(self) -> Turn:
        return self.turns[-1]

    @property
    def death(self) -> Death | None:
        """The exit record, when this agent had already died."""
        return next((turn.death for turn in self.turns if turn.death is not None), None)

    def to_json(self) -> dict[str, Any]:
        return {"turns": [turn.to_json() for turn in self.turns]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AgentSnapshot:
        turns = data.get("turns")

        if not isinstance(turns, list):
            raise ValueError("snapshot requires a turns list")

        return cls(turns=tuple(Turn.from_json(item) for item in turns))


@dataclass(frozen=True)
class InFlight:
    """
    The snapshot's last open turn (ended_at is None), and what a
    restore must do with it:

    answered  the model produced a message: replay it verbatim
              (thinking included), pair each tool_call_id with
              the result that landed, and mark the rest
              interrupted
    requeue   nothing was produced: drop the turn and re-deliver
              its inbox to the next turn
    """

    turn: Turn

    @classmethod
    def read(cls, snapshot: AgentSnapshot) -> InFlight | None:
        last = snapshot.last

        return None if last.ended_at is not None else cls(turn=last)

    @property
    def answered(self) -> bool:
        return (
            self.turn.reply is not None
            or bool(self.turn.thinking)
            or bool(self.turn.calls)
            or self.turn.rendered is not None
        )

    @property
    def requeue(self) -> tuple[InboxMessage, ...]:
        return () if self.answered else self.turn.inbox


class SnapshotStore:
    """
    Atomic per-agent snapshots under one directory. `save` offloads
    the write to a thread; reads happen at boot and in tests.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    def path(self, agent_id: str) -> Path:
        return self._directory / f"{agent_id}.json"

    async def save(self, snapshot: AgentSnapshot, *, durable: bool = True) -> None:
        """
        Whole-state overwrite. `durable` adds the fsync: the
        response and committed writes need it; the intent and
        verb snapshots may lose their tail on a power cut (restore
        can requeue).
        """
        await asyncio.to_thread(self._write, snapshot, durable)

    def _write(self, snapshot: AgentSnapshot, durable: bool) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)

        target = self.path(snapshot.agent_id)

        fd, tmp_name = tempfile.mkstemp(
            dir=self._directory,
            prefix=f".{snapshot.agent_id}-",
            suffix=".tmp",
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(snapshot.to_json(), handle, ensure_ascii=False)

                if durable:
                    handle.flush()

                    os.fsync(handle.fileno())

            os.replace(tmp_name, target)

        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)

            raise

    def load_all(self) -> dict[str, AgentSnapshot]:
        """
        Every readable snapshot, keyed by agent id. An unreadable
        file is logged and skipped: its agent cannot come back.
        """
        snapshots: dict[str, AgentSnapshot] = {}

        if not self._directory.is_dir():
            return snapshots

        for path in sorted(self._directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))

                snapshot = AgentSnapshot.from_json(data)

                if snapshot.agent_id != path.stem:
                    raise ValueError(
                        f"snapshot id {snapshot.agent_id} does not match file {path.name}"
                    )

            except (OSError, ValueError) as exc:
                logger.error("snapshot {} unreadable; agent not restored: {}", path.name, exc)

                continue

            snapshots[snapshot.agent_id] = snapshot

        return snapshots

    def remove(self, agent_id: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path(agent_id).unlink()
