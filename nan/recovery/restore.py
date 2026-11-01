"""
Boot restore: rebuild the live tree from the snapshots on disk.

One snapshot per agent holds that agent's Turn list; every turn
carries agent_id / parent_id / depth, so the spawn tree is not
stored separately -- it is rebuilt from the files.

Restore is automatic and has no user input: the last open turn
either closes (the model already answered: its missing verb
results are filled as interrupted) or drops (nothing was
produced: its inbox rides into the next turn), then every agent
starts running again. Snapshots whose agent already carried an
exit record, and subtrees whose parent is gone, are discarded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from nan.events import sink
from nan.turn import freeze_turn

from .snapshot import AgentSnapshot, InFlight, SnapshotStore

if TYPE_CHECKING:
    from nan.agent.core import Agent
    from nan.agent.handle import Book
    from nan.agent.sandbox import Sandbox
    from nan.injection import InjectionHub
    from nan.llm import LLMProvider


async def restore(
    *,
    book: Book,
    llm: LLMProvider,
    injections: InjectionHub,
    sandbox: Sandbox,
    snapshots: SnapshotStore,
) -> Agent | None:
    """
    Rebuild every agent the snapshots describe and start it.

    Returns the root, or None when there is no root snapshot (the
    caller then boots a fresh one). Must run after the sink was
    reset for this boot, so restore events carry the boot id.
    """
    loaded = snapshots.load_all()

    if not loaded:
        return None

    # An exit record means the agent already died; disposal did not
    # finish (the process stopped between commit and cleanup).
    for agent_id, snapshot in list(loaded.items()):
        if snapshot.death is not None:
            logger.warning("[{}] snapshot carries a death; discarded", agent_id)

            snapshots.remove(agent_id)

            del loaded[agent_id]

    roots = [snapshot for snapshot in loaded.values() if snapshot.depth == 0]

    if not roots:
        logger.error("no root snapshot; starting fresh")

        for agent_id in loaded:
            snapshots.remove(agent_id)

        return None

    if len(roots) > 1:
        logger.error("{} root snapshots; keeping {}", len(roots), roots[0].agent_id)

        for snapshot in roots[1:]:
            loaded.pop(snapshot.agent_id, None)

    root_snapshot = roots[0]

    # Children grouped by the parent they name; only the subtree
    # hanging off the root is reachable.
    tree: dict[str, list[AgentSnapshot]] = {}

    for snapshot in loaded.values():
        if snapshot.parent_id is not None:
            tree.setdefault(snapshot.parent_id, []).append(snapshot)

    for group in tree.values():
        group.sort(key=lambda snapshot: snapshot.agent_id)

    reachable = {snapshot.agent_id: snapshot for snapshot in _subtree(root_snapshot, tree)}

    for agent_id in loaded.keys() - reachable.keys():
        logger.warning("[{}] snapshot has no live parent; discarded", agent_id)

        snapshots.remove(agent_id)

    # Local imports: nan.agent imports nan.recovery, so the cycle
    # must close after this module is fully loaded.
    from nan.agent.core import Agent
    from nan.agent.handle import Handle

    agents: dict[str, Agent] = {}

    for snapshot in reachable.values():
        parent = agents.get(snapshot.parent_id) if snapshot.parent_id is not None else None

        agent = Agent(
            book=book,
            llm=llm,
            injections=injections,
            sandbox=sandbox,
            parent=parent,
            agent_id=snapshot.agent_id,
            snapshots=snapshots,
        )

        agent.turns = list(snapshot.turns)

        if parent is None:
            if snapshot.depth != 0:
                logger.error(
                    "[{}] snapshot is a root but depth is {}", agent.agent_id, snapshot.depth
                )

        else:
            parent._children[agent.agent_id] = Handle(holder_id=parent.agent_id, target=agent)

            if snapshot.depth != agent.depth:
                logger.warning(
                    "[{}] snapshot depth {} != tree depth {}; using the tree",
                    agent.agent_id,
                    snapshot.depth,
                    agent.depth,
                )

        agents[agent.agent_id] = agent

    for agent in agents.values():
        await _recover(agent, snapshots)

    for agent in agents.values():
        agent.start()

    return agents[root_snapshot.agent_id]


def _subtree(root: AgentSnapshot, tree: dict[str, list[AgentSnapshot]]) -> list[AgentSnapshot]:
    """The root and its subtree, parents before children."""
    ordered: list[AgentSnapshot] = []
    seen: set[str] = set()
    pending = [root]

    while pending:
        snapshot = pending.pop(0)

        if snapshot.agent_id in seen:
            continue

        seen.add(snapshot.agent_id)

        ordered.append(snapshot)

        pending.extend(tree.get(snapshot.agent_id, ()))

    return ordered


async def _recover(agent: Agent, snapshots: SnapshotStore) -> None:
    """
    Apply the last open turn's disposition, persist the result and
    announce the restore. Completed turns are already durable.
    """
    snapshot = AgentSnapshot(turns=tuple(agent.turns)) if agent.turns else None

    outcome = None

    if snapshot is not None:
        inflight = InFlight.read(snapshot)

        if inflight is not None:
            if inflight.answered:
                # The model answered: close the turn without asking
                # again, filling the verbs that never reported.
                agent.turns[-1] = freeze_turn(inflight.turn.close_interrupted())

                outcome = "closed"

                sink.emit(
                    "turn_interrupted",
                    content={
                        "agent_id": agent.agent_id,
                        "turn_index": len(agent.turns),
                        "reason": "restored mid-execution",
                    },
                )

            else:
                # Nothing was produced: the turn is dropped and its
                # inbox rides into the next one.
                agent.turns.pop()

                outcome = "requeued"

                for message in inflight.requeue:
                    agent._deliver(message)

    agent._messages = None

    if agent.turns:
        await snapshots.save(AgentSnapshot(turns=tuple(agent.turns)))

    else:
        snapshots.remove(agent.agent_id)

    sink.emit(
        "snapshot_restored",
        content={
            "agent_id": agent.agent_id,
            "parent_id": agent.parent.agent_id if agent.parent is not None else None,
            "depth": agent.depth,
            "turns": len(agent.turns),
            "in_flight": outcome,
        },
    )

    logger.info(
        "[{}] restored {} turns from snapshot ({})",
        agent.agent_id,
        len(agent.turns),
        outcome or "clean",
    )
