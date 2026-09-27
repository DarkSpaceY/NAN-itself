"""
Vocabulary of the modules package.

DataSpace / DataSpaceReader / Turn / ChannelSpec / Module /
ModuleState / ModuleRecord. Pure contracts; imports nothing
from siblings.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from pydantic import BaseModel


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
    what the agent saw and did lives on the Turn itself -- as
    STRUCTURE. The message rendering lives exclusively in the
    engine's prompt assembly; no message is ever carried here.

        identity   agent_hash / parent_hash / depth
        inputs     task / world / ambient (Module context) /
                   reports (harvested child reports) -- the
                   structured sources of the observation
        snapshots  persona / history (as of this turn's start)
                   -- there is no persistent history anywhere:
                   the next turn's history snapshot is the last
                   turn's history plus its rendered messages
                   (rendered by the engine, frozen on append)
        flow       this turn's model exchange: reply (the
                   assistant's text, possibly empty) and calls
                   (tool calls) with results (one per call,
                   positional); all empty while in flight
        response   usage / finish_reason / model (duck-typed)
        outcome    error / started_at / ended_at

    ask() receives the in-flight turn: identity and world are
    already fixed, the observation inputs are already on it
    (task, reports) except the ambient context, which the
    Modules' own ask fills in; the flow and result fields are
    still empty.

    tell() receives the same turn completed, filled via
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

    ambient: tuple[str, ...] = ()

    reports: tuple[Any, ...] = ()

    reply: str | None = None

    calls: tuple[Any, ...] = ()

    results: tuple[str, ...] = ()

    usage: Any | None = None

    finish_reason: str | None = None

    model: str | None = None

    error: str | None = None

    started_at: float | None = None
    ended_at: float | None = None


class ChannelSpec:
    """
    Declaration of one downlink channel on a Module.

    A channel is a data endpoint the model feeds through the
    invoke_channels verb. The payload shape is declared with a
    pydantic model -- the same annotation-driven mechanism @tool
    uses -- so the JSON schema handed to the model and the
    validation applied at write time derive from one source.

        model   pydantic model classifying valid payloads;
                None means the channel accepts any JSON value
                unvalidated
    """

    __slots__ = ("description", "model")

    def __init__(
        self,
        model: type[BaseModel] | None = None,
        *,
        description: str = "",
    ) -> None:
        self.model = model
        self.description = description

    def json_schema(self) -> dict[str, Any] | None:
        """
        JSON schema for the model, or None when unvalidated.
        """
        if self.model is None:
            return None

        schema = self.model.model_json_schema()

        schema.pop("title", None)

        for property_schema in schema.get(
            "properties",
            {},
        ).values():
            property_schema.pop("title", None)

        return schema

    def validate(
        self,
        payload: Any,
    ) -> tuple[Any, str | None]:
        """
        Validate one payload against the declared model.

        Returns (normalized_payload, None) on success and
        (None, error_message) on rejection. Unvalidated channels
        pass the payload through unchanged.
        """
        if self.model is None:
            return payload, None

        try:
            validated = self.model.model_validate(
                payload
            )

        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

        return validated.model_dump(), None


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

        1. ask() must only read state that was already computed
           earlier; it must not run heavy work or call the LLM.
           A slow ask() delays every agent's execution start.
        2. tell() runs in its own task, so it may *await* long
           operations (thread results, async APIs). A synchronous
           CPU-heavy or blocking call still stalls the one shared
           event loop and freezes every module and agent; such
           work belongs on threads (start()'s daemon threads, or
           asyncio.to_thread for short known-blocking calls).
        3. DataSpace carries small JSON facts only: publish() and
           snapshot() deepcopy the full state on the event loop
           every round, so frame data or large documents must
           never go through it.
        4. Provisioning-length work (model weight loading, device
           probing) must not run synchronously inside a coroutine;
           do it on start()'s background threads before the loop.
    """

    id: ClassVar[str]

    llm: Any | None = None

    requires: ClassVar[
        tuple[str, ...]
    ] = ()

    # Downlink channels the model may feed through the
    # invoke_channels verb. Declared on the subclass; plain
    # Modules stay channel-free.
    channels: ClassVar[
        Mapping[str, ChannelSpec]
    ] = MappingProxyType({})

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

    async def tell(
        self,
        record: Turn,
    ) -> None:
        """
        The engine tells the module a turn completed.

        Runs in its own task, isolated from other Modules and
        from the agents themselves: long *awaited* work is fine
        (thread results, async APIs), but a synchronous CPU-heavy
        or blocking call would stall the one shared event loop
        and freeze every module and agent -- offload such work
        to a thread (asyncio.to_thread or start()'s daemon
        threads). Any failure crashes the module: DOWN +
        supervised restart; it never propagates to the agents.
        Keep ask() a cheap projection of the results.
        """

    async def ask(
        self,
        turn: Turn,
    ) -> str | None:
        """
        The engine asks for this turn's ambient context.

        Cheap projection only. Any failure crashes the module:
        DOWN + supervised restart. The asking agent is never
        affected -- it just sees no contribution this turn.
        """
        return None

    def feed(
        self,
        channel: str,
        payload: Any,
    ) -> str:
        """
        The model feeds a payload into a channel.

        Returns 'written' or 'rejected'; sync by design. The
        runtime validates the channel and its schema before this
        call, so the default accepts the handed-down payload;
        override feed only to store or act on it. An exception
        out of feed() crashes the module: DOWN + supervised
        restart; the model only sees a failure string.
        """
        return "written"

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
