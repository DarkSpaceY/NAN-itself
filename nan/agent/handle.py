"""
Handle: a cross-agent channel and the control right over one
agent.

Semantics:

    - a Handle is bound to one target agent and one implicit
      holder (the agent that spawned the target; the root's
      handle is held by the composition root)
    - through a handle the holder may send() text into the
      target's inbox and kill() the target; the target replies
      upstream through its parent reference

A Handle is a thin value: the routing, cascade and lifecycle
live on the Agent and the Book.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import asyncio

    from nan.agent.core import Agent


class InboxTarget(Protocol):
    """
    What a Handle needs from its target: an inbox to deliver
    into, a death to request, and the loop task to await.
    """

    agent_id: str

    @property
    def task(self) -> asyncio.Task[None] | None: ...

    async def receive_up(self, text: str, source: str) -> None: ...

    def request_death(self, reason: str) -> None: ...


class Handle:
    """
    One agent's channel+control over another. Immutable.
    """

    __slots__ = ("holder_id", "_target")

    def __init__(self, *, holder_id: str, target: InboxTarget) -> None:
        self.holder_id = holder_id
        self._target = target

    @property
    def agent_id(self) -> str:
        """The short id of the agent this handle points at."""
        return self._target.agent_id

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The target's loop task (None before it starts)."""
        return self._target.task

    async def send(self, text: str) -> None:
        """
        Deliver text into the target's inbox (kind "send").
        """
        await self._target.receive_up(text, self.holder_id)

    def kill(self) -> None:
        """
        Kill the target; the death reason is fixed to this holder.
        Cascade is the target's own business: it tears down
        everything IT holds.
        """
        self._target.request_death(f"killed by holder {self.holder_id}")


class Book:
    """
    Process-wide registry of live agents.

    The Book is the authoritative id -> agent map used for
    routing, status snapshots and whole-tree shutdown. Agents
    register on spawn and unregister on exit.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        self._agents[agent.agent_id] = agent

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def all(self) -> list[Agent]:
        return list(self._agents.values())
