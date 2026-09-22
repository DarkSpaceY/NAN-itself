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
    """

    agent_hash: str
    parent_hash: str | None

    depth: int

    task: str | None

    world: Mapping[str, Any]

    metadata: Mapping[str, Any] = field(
        default_factory=dict,
    )


class SubagentHandle:
    """
    Handle for an asynchronously running Subagent.

    Dispatching a Subagent never waits for completion.
    The parent Agent can explicitly wait on the handle later.

    The underlying Task is owned by AgentRuntime. The handle is only
    the caller-facing reference to that execution.
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
        - Subagent task ownership
        - sleep/wait
        - depth limit
        - runtime shutdown

    Lifecycle:

        dispatch
            ->
        runtime owns Task
            ->
        worker executes
            ->
        handle may await it
            ->
        Task is removed from ownership set

    Runtime shutdown:

        shutdown()
            ->
        reject new dispatches
            ->
        cancel all live Subagent Tasks
            ->
        await every Task
            ->
        no live Subagent Tasks remain
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

        # Agent identity / context registry.
        #
        # Contexts intentionally remain available after execution
        # completes so reports/debugging/inspection can still resolve
        # the execution identity.
        self._agents: dict[
            str,
            AgentContext,
        ] = {}

        # AgentRuntime is the owner of every live Subagent Task.
        #
        # This is intentionally separate from `_agents`: AgentContext
        # is persistent execution metadata, whereas this set represents
        # actual live async work.
        self._subagent_tasks: set[
            asyncio.Task[Any]
        ] = set()

        # Once shutdown starts, no new Subagent may be dispatched.
        self._stopping = False

    # ------------------------------------------------------------------
    # Root Agent
    # ------------------------------------------------------------------

    def create_root(
        self,
        *,
        world: Mapping[str, Any] | None = None,
        task: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentContext:
        """
        Create one root Agent execution context.

        Root creation itself does not create an asyncio Task.
        """
        if self._stopping:
            raise RuntimeError(
                "AgentRuntime is shutting down."
            )

        agent_hash = (
            self._new_agent_hash()
        )

        context = AgentContext(
            agent_hash=agent_hash,
            parent_hash=None,
            depth=0,
            task=task,
            world=self._freeze_world(
                world or {}
            ),
            metadata=MappingProxyType(
                dict(metadata or {})
            ),
        )

        self._agents[
            agent_hash
        ] = context

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
        metadata: Mapping[str, Any] | None = None,
    ) -> SubagentHandle:
        """
        Dispatch a Subagent and return immediately.

        The new Subagent:
            - gets a new identity
            - inherits the parent's world snapshot
            - increments depth

        The resulting asyncio Task is owned by this AgentRuntime.
        """
        if self._stopping:
            raise RuntimeError(
                "AgentRuntime is shutting down."
            )

        child_depth = (
            parent.depth + 1
        )

        if (
            child_depth
            > self.max_subagent_depth
        ):
            raise SubagentLimitError(
                "Maximum Subagent depth exceeded: "
                f"{child_depth} > "
                f"{self.max_subagent_depth}"
            )

        agent_hash = (
            self._new_agent_hash()
        )

        child = AgentContext(
            agent_hash=agent_hash,
            parent_hash=parent.agent_hash,
            depth=child_depth,
            task=task,
            world=parent.world,
            metadata=MappingProxyType(
                dict(metadata or {})
            ),
        )

        self._agents[
            agent_hash
        ] = child

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

        self._subagent_tasks.add(
            task_handle
        )

        task_handle.add_done_callback(
            self._on_subagent_done
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
            return await worker(
                context
            )

        finally:
            # Keep final context available for inspection after the
            # execution has completed, including cancellation.
            self._agents[
                context.agent_hash
            ] = context

    def _on_subagent_done(
        self,
        task: asyncio.Task[Any],
    ) -> None:
        """
        Remove a completed Task from runtime ownership.

        Calling task.exception() marks an exception as retrieved so
        fire-and-forget Subagents do not later produce:

            Task exception was never retrieved

        This does NOT consume the exception from callers awaiting
        the SubagentHandle. `await handle.wait()` still raises it.
        """
        self._subagent_tasks.discard(
            task
        )

        if task.cancelled():
            return

        try:
            task.exception()
        except (
            asyncio.CancelledError,
        ):
            pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Runtime lifecycle
    # ------------------------------------------------------------------

    @property
    def stopping(self) -> bool:
        return self._stopping

    @property
    def active_subagent_count(self) -> int:
        return sum(
            not task.done()
            for task in self._subagent_tasks
        )

    async def shutdown(self) -> None:
        """
        Stop every still-running Subagent owned by this runtime.

        Normal parent-turn behavior is intentionally different:
        a Subagent may outlive its parent's StepEngine execution.

        `shutdown()` is the application/runtime lifecycle boundary
        where all remaining Subagent work must be cancelled and
        awaited before dependent runtimes are closed.
        """
        if self._stopping:
            # The first shutdown call owns the actual cleanup.
            # Subsequent calls wait only for tasks that remain in the
            # ownership set.
            tasks = list(
                self._subagent_tasks
            )

            if tasks:
                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

            return

        self._stopping = True

        # Freeze the ownership boundary before the first await:
        # dispatch() can no longer create a new Subagent after this
        # point.
        tasks = list(
            self._subagent_tasks
        )

        if not tasks:
            return

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        # The done callbacks should already have discarded these,
        # but clearing defensively keeps the invariant explicit.
        self._subagent_tasks.clear()

    # ------------------------------------------------------------------
    # Agent state
    # ------------------------------------------------------------------

    def get_agent(
        self,
        agent_hash: str,
    ) -> AgentContext | None:
        return self._agents.get(
            agent_hash
        )

    def agents(
        self,
    ) -> tuple[AgentContext, ...]:
        return tuple(
            self._agents.values()
        )

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    async def sleep(
        self,
        seconds: float,
    ) -> None:
        """
        Sleep for `seconds`.
        """
        if seconds < 0:
            raise ValueError(
                "seconds must be >= 0"
            )

        await asyncio.sleep(
            seconds
        )

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
            *(
                handle.wait()
                for handle in handles
            )
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

        The normal producer of `world` should already provide a
        detached snapshot. We deliberately do not recursively convert
        arbitrary values here because the Module/DataSpace layer owns
        snapshot semantics.

        Therefore:

            DataSpace.snapshot()
                -> deep detached data

            AgentRuntime._freeze_world()
                -> immutable outer mapping
        """
        return MappingProxyType(
            dict(world)
        )