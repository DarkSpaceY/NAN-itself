"""
Vocabulary of the modules package.

DataSpace / DataSpaceReader / Turn / Module / ModuleState /
ModuleRecord. Pure contracts; imports nothing from siblings.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, Mapping


MODULE_HEADER = "# @module"


# ============================================================================
# DataSpace
# ============================================================================


class DataSpace:
    """
    A Module-owned published state.

    The owner Module is the only writer.
    Readers receive detached deep copies.
    """

    def __init__(
        self,
        owner: str,
        initial: Mapping[str, Any] | None = None,
    ) -> None:
        self.owner = owner
        self._value: dict[str, Any] = deepcopy(
            dict(initial or {})
        )
        self.revision = 0

    def publish(
        self,
        value: Mapping[str, Any],
    ) -> None:
        """
        Atomically publish a new state.

        The input is deep-copied so the caller cannot retain
        references into DataSpace internals.
        """
        if not isinstance(value, Mapping):
            raise TypeError(
                "DataSpace value must be a Mapping"
            )

        value_copy = deepcopy(
            dict(value)
        )

        self._value = value_copy
        self.revision += 1

    def snapshot(self) -> dict[str, Any]:
        """
        Return a completely detached snapshot.
        """
        return deepcopy(self._value)

    def dump(self) -> dict[str, Any]:
        """
        Alias used by Facade persistence.
        """
        return self.snapshot()


class DataSpaceReader:
    """
    Read-only handle exposed to dependent Modules.

    It deliberately does not expose publish().
    """

    __slots__ = ("_space",)

    def __init__(
        self,
        space: DataSpace,
    ) -> None:
        self._space = space

    @property
    def owner(self) -> str:
        return self._space.owner

    @property
    def revision(self) -> int:
        return self._space.revision

    def snapshot(self) -> dict[str, Any]:
        return self._space.snapshot()


# ============================================================================
# Module
# ============================================================================


@dataclass(frozen=True)
class Turn:
    """
    One agent turn as seen by Modules, and the ONLY record of it.

    A turn is self-contained: everything needed to reconstruct
    what the agent saw and did lives on the Turn itself.

        identity   agent_hash / parent_hash / depth
        inputs     task / world (structured sources; the
                   observation message is their rendering)
        snapshots  persona / history (as of this turn's start)
                   -- there is no persistent history anywhere:
                   the next turn's history snapshot is derived
                   from the last turn's history + messages
        messages   this turn's message flow (observation +
                   assistant + tool results), i.e. what the
                   model produced; full model input for the
                   turn is history + the first message
        response   usage / finish_reason / model (duck-typed;
                   tool calls live on the assistant message)
        outcome    error / started_at / ended_at

    query() receives the in-flight turn: identity and world are
    already fixed, the snapshot and result fields are still None.

    on_turn() receives the same turn completed, filled via
    dataclasses.replace().

    The history snapshot is cleared (empty tuple) whenever the
    running character count exceeded the engine's limit: earlier
    turns stay on record, but the broken chain starts fresh.
    """

    agent_hash: str
    parent_hash: str | None
    depth: int

    task: str | None

    world: Mapping[str, Mapping[str, Any]]

    persona: str | None = None

    history: tuple[Any, ...] = ()

    messages: tuple[Any, ...] = ()

    usage: Any | None = None

    finish_reason: str | None = None

    model: str | None = None

    error: str | None = None

    started_at: float | None = None
    ended_at: float | None = None


class Module:
    """
    Base class for all Modules.

    A Module owns:
        self.data

    A Module can read:
        self.dependencies

    A Module may use:
        self.llm -- LLM handle provisioned by the Facade when the
        composition root supplies one

    A Module's start() represents its service lifetime.

    Performance rule (the Module's one hard obligation):

        1. query() must only read state that was already computed
           earlier; it must not run heavy work or call the LLM.
        2. Heavy work belongs in start()'s background loop and in
           on_turn(); each of those runs in its own task.
        3. A slow query() delays every agent's execution start.
    """

    id: ClassVar[str]

    llm: Any | None = None

    requires: ClassVar[
        tuple[str, ...]
    ] = ()

    data: DataSpace

    dependencies: Mapping[
        str,
        DataSpaceReader,
    ]

    async def start(self) -> None:
        """
        Enter the Module's running state.

        Normally this should be a long-running coroutine.

        A state-only Module may return immediately; Facade will treat
        it as exited and retry it later.
        """
        raise NotImplementedError

    async def stop(self) -> None:
        """
        Stop all work owned by this Module.

        Exceptions are recorded by Facade, but do not prevent
        cleanup of other Modules.
        """

    async def on_turn(
        self,
        record: Turn,
    ) -> None:
        """
        Observe a completed agent execution.

        Runs in its own task, isolated from other Modules and from
        the agents themselves. Heavy processing is allowed here;
        keep query() a cheap projection of the results.
        """

    async def query(
        self,
        turn: Turn,
    ) -> str | None:
        """
        Produce Agent-facing context/prompt for this turn.

        Query failure does not automatically bring the Module down.
        """
        return None

    def serialize_state(self) -> Any:
        """
        Return JSON-serializable private Module state.

        Facade stores the result under:
            ./data/modules/private/<module_id>.json
        """
        return None

    def restore_state(
        self,
        state: Any,
    ) -> None:
        """
        Restore private Module state.
        """
        pass


class ModuleState(Enum):
    NEW = auto()
    STARTING = auto()
    RUNNING = auto()
    DOWN = auto()
    STOPPING = auto()


@dataclass
class ModuleRecord:
    id: str
    cls: type[Module]
    instance: Module
    data: DataSpace

    source: str

    generation: int = 0

    task: asyncio.Task[None] | None = None

    state: ModuleState = ModuleState.NEW

    error: BaseException | None = None

    retry_at: float = 0.0

    source_fingerprint: tuple[int, int] | None = None

    imported_module_name: str | None = None


class DuplicateModuleError(RuntimeError):
    """
    Raised when two sources claim the same module id.

    The message always lists both sources.
    """
