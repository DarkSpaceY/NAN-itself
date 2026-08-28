from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Mapping


SubagentWorker = Callable[
    ["AgentContext"],
    Awaitable[Any],
]


class SubagentLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentContext:
    """
    Runtime context owned by one Agent execution.

    `world` is shared by all Agents within the same dispatch tree.
    `skill` identifies the currently active Skill (subagents may
    switch it via activate_skill).
    """

    agent_hash: str
    parent_hash: str | None

    depth: int

    task: str | None

    skill: Any | None

    world: Mapping[str, Any]

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
    )


class SubagentHandle:
    """
    Handle for an asynchronously running Subagent.

    Dispatching a Subagent never waits for completion.
    The parent Agent can explicitly wait on the handle later.
    """

    __slots__ = (
        "agent_hash",
        "parent_hash",
        "depth",
        "_task",
    )

    def __init__(
        self,
        *,
        agent_hash: str,
        parent_hash: str,
        depth: int,
        task: asyncio.Task[Any],
    ) -> None:
        self.agent_hash = agent_hash
        self.parent_hash = parent_hash
        self.depth = depth
        self._task = task

    @property
    def done(self) -> bool:
        return self._task.done()

    @property
    def cancelled(self) -> bool:
        return self._task.cancelled()

    def cancel(self) -> bool:
        return self._task.cancel()

    async def wait(self) -> Any:
        return await self._task

    async def result(self) -> Any:
        return await self._task


class AgentRuntime:
    """
    Minimal Core/Subagent execution runtime.

    This layer deliberately does not know about:
        - LLM providers
        - ModuleFacade
        - MCP
        - Skill filesystem
        - prompt assembly

    It only owns:
        - Agent identity
        - Agent context
        - shared world snapshot
        - Subagent dispatch
        - recursive dispatch
        - sleep/wait
        - depth limit
    """

    def __init__(
        self,
        *,
        max_subagent_depth: int = 3,
    ) -> None:
        if max_subagent_depth < 0:
            raise ValueError(
                "max_subagent_depth must be >= 0"
            )

        self.max_subagent_depth = max_subagent_depth

        self._agents: dict[str, AgentContext] = {}

        # Set by the composition root: new-input wake signal used
        # by sleep() so idle naps yield to pending messages.
        self.interrupt_event: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # Root Agent
    # ------------------------------------------------------------------

    def create_root(
        self,
        *,
        world: Mapping[str, Any] | None = None,
        skill: Any | None = None,
        task: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentContext:
        agent_hash = self._new_agent_hash()

        context = AgentContext(
            agent_hash=agent_hash,
            parent_hash=None,
            depth=0,
            task=task,
            skill=skill,
            world=self._freeze_world(world or {}),
            metadata=MappingProxyType(
                dict(metadata or {})
            ),
        )

        self._agents[agent_hash] = context

        return context

    # ------------------------------------------------------------------
    # Subagent
    # ------------------------------------------------------------------

    def dispatch(
        self,
        parent: AgentContext,
        *,
        task: str,
        worker: SubagentWorker,
        skill: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SubagentHandle:
        """
        Dispatch a Subagent and return immediately.

        The new Subagent:
            - gets a new identity
            - inherits the parent's world snapshot
            - starts with an empty Skill by default
            - increments depth
        """
        child_depth = parent.depth + 1

        if child_depth > self.max_subagent_depth:
            raise SubagentLimitError(
                "Maximum Subagent depth exceeded: "
                f"{child_depth} > "
                f"{self.max_subagent_depth}"
            )

        agent_hash = self._new_agent_hash()

        child = AgentContext(
            agent_hash=agent_hash,
            parent_hash=parent.agent_hash,
            depth=child_depth,
            task=task,
            skill=skill if skill is not None else parent.skill,
            world=parent.world,
            metadata=MappingProxyType(
                dict(metadata or {})
            ),
        )

        self._agents[agent_hash] = child

        task_handle = asyncio.create_task(
            self._run_worker(
                child,
                worker,
            ),
            name=(
                f"subagent:"
                f"{agent_hash}"
            ),
        )

        return SubagentHandle(
            agent_hash=agent_hash,
            parent_hash=parent.agent_hash,
            depth=child_depth,
            task=task_handle,
        )

    async def _run_worker(
        self,
        context: AgentContext,
        worker: SubagentWorker,
    ) -> Any:
        try:
            return await worker(context)
        finally:
            # Keep final context available for inspection after the
            # execution has completed.
            self._agents[context.agent_hash] = context

    # ------------------------------------------------------------------
    # Agent state
    # ------------------------------------------------------------------

    def get_agent(
        self,
        agent_hash: str,
    ) -> AgentContext | None:
        return self._agents.get(agent_hash)

    def agents(self) -> tuple[AgentContext, ...]:
        return tuple(
            self._agents.values()
        )

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    def set_interrupt_event(self, event: asyncio.Event | None) -> None:
        self.interrupt_event = event

    async def sleep(
        self,
        seconds: float,
    ) -> tuple[float, bool]:
        """
        Sleep for `seconds`, OR until the interrupt event fires
        (new input arrived). Returns (waited, interrupted).
        """
        if seconds < 0:
            raise ValueError(
                "seconds must be >= 0"
            )

        event = self.interrupt_event

        if event is None:
            await asyncio.sleep(seconds)
            return seconds, False

        started = time.time()

        sleeper = asyncio.create_task(
            asyncio.sleep(seconds),
            name="agent-sleep",
        )

        waiter = asyncio.create_task(
            event.wait(),
            name="agent-sleep-interrupt",
        )

        done, pending = await asyncio.wait(
            {sleeper, waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        return time.time() - started, waiter in done

    @staticmethod
    async def wait(
        handle: SubagentHandle,
    ) -> Any:
        return await handle.wait()

    @staticmethod
    async def wait_all(
        *handles: SubagentHandle,
    ) -> list[Any]:
        return await asyncio.gather(
            *(handle.wait() for handle in handles)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _new_agent_hash() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _freeze_world(
        world: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """
        Freeze only the outer mapping.

        The normal producer of `world` should already provide a detached
        snapshot. We deliberately do not recursively convert arbitrary
        values here, because the Module/DataSpace layer owns snapshot
        semantics.
        """
        return MappingProxyType(
            dict(world)
        )