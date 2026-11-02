"""
Persistence: one snapshot per agent -- its identity, its Turn list,
plus the messages no turn has observed yet, and nothing else.

Identity belongs to the snapshot, not to the turns: an agent exists
as soon as its snapshot does, so a child that was spawned but never
observed a turn is still durable. The turns carry the same identity
so each Turn stays self-contained for observers.

The spawn tree is not stored: `parent_id` names the edge, so
restore rebuilds the tree from the files on disk.

The store is whole-state, never append: the state is serialized to
a temp file and atomically replaced over the target, so a crash
leaves either the old snapshot or the new one, never a torn one.
Every write is fsynced: the agent is durable at each phase
boundary.
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
    One agent: its identity, its turns in order, and the inbox
    messages that arrived but no turn has observed. The file is
    named by the agent's id. `turns` may be empty -- that is a
    spawned agent which has not opened its first turn yet.
    """

    agent_id: str
    parent_id: str | None
    depth: int
    turns: tuple[Turn, ...] = ()
    pending: tuple[InboxMessage, ...] = ()

    def __post_init__(self) -> None:
        if any(turn.agent_id != self.agent_id for turn in self.turns):
            raise ValueError(f"snapshot {self.agent_id} carries another agent's turn")

    @property
    def last(self) -> Turn:
        return self.turns[-1]

    @property
    def death(self) -> Death | None:
        """The exit record, when this agent had already died."""
        return next((turn.death for turn in self.turns if turn.death is not None), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "turns": [turn.to_json() for turn in self.turns],
            "pending": [message.to_json() for message in self.pending],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AgentSnapshot:
        try:
            agent_id = data["agent_id"]
            parent_id = data["parent_id"]
            depth = data["depth"]
            turns = data["turns"]
        except KeyError as exc:
            raise ValueError(f"snapshot missing {exc}") from exc

        if not isinstance(agent_id, str) or not isinstance(depth, int):
            raise ValueError("snapshot requires a string agent_id and an int depth")

        if not isinstance(turns, list):
            raise ValueError("snapshot requires a turns list")

        return cls(
            agent_id=agent_id,
            parent_id=parent_id if isinstance(parent_id, str) else None,
            depth=depth,
            turns=tuple(Turn.from_json(item) for item in turns),
            pending=tuple(InboxMessage.from_json(message) for message in data.get("pending", ())),
        )


@dataclass(frozen=True)
class InFlight:
    """
    The snapshot's last open turn (phase before `committed`), and
    what a restore must do with it:

    requeue   nothing was observed: drop the turn and re-deliver its
              inbox to the next turn
    resume    the turn already has inputs or progress: keep it, the
              loop continues it from its phase
    """

    turn: Turn

    @classmethod
    def read(cls, snapshot: AgentSnapshot) -> InFlight | None:
        if not snapshot.turns:
            return None

        last = snapshot.last

        return None if last.phase == "committed" else cls(turn=last)

    @property
    def unobserved(self) -> bool:
        """Nothing was observed: the turn cannot resume, only requeue."""
        return self.turn.phase == "observed"

    @property
    def requeue(self) -> tuple[InboxMessage, ...]:
        return self.turn.inbox if self.unobserved else ()


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

    async def save(self, snapshot: AgentSnapshot) -> None:
        """Whole-state, fsynced overwrite. The write runs in a thread."""
        await asyncio.to_thread(self._write, snapshot)

    def _write(self, snapshot: AgentSnapshot) -> None:
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

            except (OSError, ValueError, TypeError, AttributeError) as exc:
                logger.error("snapshot {} unreadable; agent not restored: {}", path.name, exc)

                continue

            snapshots[snapshot.agent_id] = snapshot

        return snapshots

    def remove(self, agent_id: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path(agent_id).unlink()
