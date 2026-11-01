"""
The Agent: one homogeneous loop, root or subagent.

    1 agent = 1 asyncio Task. A turn opens when the agent observes
    (drains its inbox, collects its deltas), and stays resident in
    `turns` until it commits: inject -> think -> act -> freeze.
    The Turn list IS the conversation; history is derived from the
    rendered turns after the last context epoch.

    Retry: a retryable failure re-runs the SAME turn's model call
    along the backoff ladder, counted in LLMCall.retries. When the
    ladder runs out, a subagent dies; the root parks the turn
    (paused) instead of dying.

    Persistence: the whole Turn list is snapshotted at four points
    per turn (intent, response, verb results, commit), so a stop
    restarts from the last complete fact rather than from nothing.

    Exit has two shapes. Shutdown (SIGTERM/SIGINT) suspends: the
    in-flight turn stays on disk, no death, no cascade, no notices.
    Death (finish/kill/unrecoverable) disposes: cascade to held
    children, notify the parent, delete the snapshot.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from nan.agent.engine import TurnEngine
from nan.agent.handle import Handle
from nan.agent.verbs import VERBS
from nan.config import settings
from nan.events import sink
from nan.llm import Message
from nan.recovery import (
    AgentSnapshot,
    CancelReason,
    FailureKind,
    SnapshotStore,
    Unrecoverable,
    classify,
)
from nan.turn import (
    Death,
    DeltaEvent,
    InboxKind,
    InboxMessage,
    Turn,
    TurnOrigin,
    freeze_turn,
)

if TYPE_CHECKING:
    from nan.agent.handle import Book
    from nan.agent.sandbox import Sandbox
    from nan.injection import InjectionHub
    from nan.llm import LLMProvider


class Agent:
    def __init__(
        self,
        *,
        book: Book,
        llm: LLMProvider,
        injections: InjectionHub,
        sandbox: Sandbox,
        parent: Agent | None = None,
        agent_id: str | None = None,
        snapshots: SnapshotStore | None = None,
    ) -> None:
        # --- collaborators (handed in) ---
        self.book = book
        self.llm = llm
        self.engine = TurnEngine(llm)
        self.injections = injections
        self.parent = parent  # upstream channel for send
        self.sandbox = sandbox  # execution-level sandbox for the shell verb
        self._snapshots = snapshots or SnapshotStore(settings.paths.snapshots)

        # --- identity ---
        self.agent_id = agent_id or uuid.uuid4().hex[:8]
        self.depth: int = 0 if parent is None else parent.depth + 1

        # --- persona: boot read ---
        # The file is CORE: an unreadable persona refuses startup.
        # It is re-read every turn (a mid-run failure fails that
        # turn) and a content change is a system-prompt
        # invalidation point.
        try:
            self._persona_last = settings.paths.persona.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"persona file unreadable at boot: {settings.paths.persona}"
            ) from exc

        # --- children: the handles I hold (cascade follows the spawn tree) ---
        self._children: dict[str, Handle] = {}

        # --- inbox: delivery + its wake source ---
        self._inbox: asyncio.Queue[InboxMessage] = asyncio.Queue()
        self._inbox_wake = asyncio.Event()

        # --- tasks: the loop, and the one cancellable in-flight turn ---
        self._task: asyncio.Task[None] | None = None
        self._turn_task: asyncio.Task[Turn] | None = None

        # --- pause state: the flag gates the loop, the event wakes it ---
        self._paused = False
        self._resume_wake = asyncio.Event()

        # --- turns: the only runtime record ---
        # `_messages` is a cache of the rendered history (turns after
        # the last context epoch); it is dropped whenever turns change.
        self.turns: list[Turn] = []
        self._messages: list[Message] | None = None

        # Deltas that arrived while no turn was open. They ride the
        # next turn's observation. Transient by design: a crash may
        # lose them, a turn list never does.
        self._pending_deltas: list[DeltaEvent] = []

        # --- lifecycle: the exit record, and the start flag ---
        self._death: Death | None = None  # first exit record wins (latch)
        self._started = False
        self._cancel_reason: CancelReason | None = None

    # ==================================================================
    # Identity / handles
    # ==================================================================

    def start(self) -> None:
        """Register and start the loop task."""
        self.book.register(self)

        self._task = asyncio.create_task(self.loop(), name=f"agent:{self.agent_id}")

    def child_handle(self, agent_id: Any) -> Handle | None:
        if not isinstance(agent_id, str):
            return None

        return self._children.get(agent_id)

    def child_handles(self) -> list[Handle]:
        return list(self._children.values())

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The loop task; None before start()."""
        return self._task

    @property
    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    @property
    def history(self) -> list[Message]:
        """
        The rendered conversation the next model call sees: every
        committed turn from the last context epoch on. The open
        turn is unrendered and therefore absent.
        """
        if self._messages is None:
            self._messages = self._rebuild_history()

        return self._messages

    def _rebuild_history(self) -> list[Message]:
        start = 0

        for index, turn in enumerate(self.turns):
            if turn.origin == "context_cleared":
                start = index

        return [
            message
            for turn in self.turns[start:]
            if turn.rendered is not None
            for message in turn.rendered
        ]

    # ==================================================================
    # Inbox
    # ==================================================================

    async def receive_up(self, text: str, source: str) -> None:
        """
        Handle.send lands here. The sender's role is resolved at
        delivery: only the user (via the root handle) and my
        parent can send into me directly; a held child reporting
        up is labelled subagent. Anything else is unverified and
        delivered as "unknown" with a warning.
        """
        if self.depth == 0 and source == "user":
            kind: InboxKind = "user"
        elif self.parent is not None and source == self.parent.agent_id:
            kind = "parentagent"
        elif source in self._children:
            kind = "subagent"
        else:
            kind = "unknown"

            logger.warning("[{}] message from unverified source '{}'", self.agent_id, source)

        self._deliver(InboxMessage(kind=kind, source=source, text=text))

    def _deliver(self, message: InboxMessage) -> None:
        """Enqueue one inbox message and wake the (possibly
        event-driven) loop. Safe from any task."""
        self._inbox.put_nowait(message)

        self._inbox_wake.set()

    # ==================================================================
    # Spawning
    # ==================================================================

    def spawn_child(self, task: str) -> tuple[Agent, str | None]:
        """
        Create and start a child; return (child, error). The
        caller (spawn verb) renders the error for the model.
        """
        if settings.agent.max_depth >= 0 and self.depth + 1 > settings.agent.max_depth:
            return self, f"Max depth exceeded: {self.depth + 1} > {settings.agent.max_depth}"

        if settings.agent.max_children >= 0 and len(self._children) >= settings.agent.max_children:
            return self, f"Max children reached: {settings.agent.max_children}"

        # A child shares the parent's collaborators and inherits
        # everything else from the process-wide settings.
        child = Agent(
            book=self.book,
            llm=self.llm,
            injections=self.injections,
            sandbox=self.sandbox,
            parent=self,
            snapshots=self._snapshots,
        )

        # The holder is me (the spawner); the target is the child.
        self._children[child.agent_id] = Handle(holder_id=self.agent_id, target=child)

        # "spawn == spawn then send": the task is delivered as the
        # child's first inbox message, tagged from its parent.
        child._deliver(InboxMessage(kind="parentagent", source=self.agent_id, text=task))

        self.note_delta("subagent", f"+{child.agent_id} spawned")

        child.start()

        return child, None

    # ==================================================================
    # Death / pause
    # ==================================================================

    def note_delta(self, section: str, line: str) -> None:
        """
        Record a system-state event line ("subagent", "-j died:
        ..."), drained into the next turn's <delta> section under
        the given tag.
        """
        self._pending_deltas.append(DeltaEvent(section=section, line=line))

    def _begin_death(self, death: Death) -> bool:
        """Record the exit; the first record wins (idempotent)."""
        if self._death is not None:
            return False

        self._death = death

        return True

    def request_death(self, reason: str) -> None:
        """
        Kill this agent now: record the exit and cancel the loop
        (the loop's disposal does the rest: in-flight turn dropped,
        cascade, notice, snapshot removed). A death recorded before
        the loop takes its first step is honored on entry instead.
        """
        if not self._begin_death(Death(reason)):
            return

        self._cancel_reason = CancelReason.KILL

        self._cancel_tasks()

    def finish(self, report: str) -> None:
        """
        Record the final report (FinishVerb). The loop ends at
        the next boundary; disposal delivers it upstream. The
        record latches: a second finish is a no-op (the first
        report wins).
        """
        self._begin_death(Death("finished", report))

    def suspend(self) -> None:
        """
        Stop for process shutdown: cancel the loop, keep the
        snapshot. No death, no cascade, no notices -- the next
        boot restores from disk.
        """
        if self._death is not None:
            return

        self._cancel_reason = CancelReason.SHUTDOWN

        self._cancel_tasks()

    def pause(self) -> None:
        """
        Park the in-flight turn where it is (its inputs and facts
        stay on the turn, snapshotted); the loop waits for resume.
        Idempotent.
        """
        if self._paused:
            return

        self._paused = True

        self._cancel_reason = CancelReason.PAUSE

        sink.emit("agent_paused", content={"agent_id": self.agent_id})

        if self._turn_task is not None:
            self._turn_task.cancel()

    def resume(self) -> None:
        """
        Lift the pause. A parked turn continues from its facts:
        an answered turn closes its unfinished calls as
        interrupted, an unanswered one re-runs its model call.
        Idempotent.
        """
        if not self._paused and self._paused_turn() is None:
            return

        self._paused = False

        sink.emit("agent_resumed", content={"agent_id": self.agent_id})

        self._resume_wake.set()

    @property
    def paused(self) -> bool:
        """The agent is parked: no later turn exists, and either
        the gate is closed or the last turn carries the bit."""
        return self._paused or self._paused_turn() is not None

    def _cancel_tasks(self) -> None:
        if self._turn_task is not None:
            self._turn_task.cancel()

        if self._started and self._task is not None:
            self._task.cancel()

    def _paused_turn(self) -> Turn | None:
        turn = self.last_turn

        if turn is None or turn.ended_at is not None or not turn.paused or turn.death is not None:
            return None

        return turn

    # ==================================================================
    # The loop
    # ==================================================================

    async def loop(self) -> None:
        self._started = True

        sink.emit(
            "agent_started",
            content={
                "agent_id": self.agent_id,
                "depth": self.depth,
                "parent_id": (self.parent.agent_id if self.parent else None),
                "pacing": (
                    "interval" if settings.agent.autonomous_interval > 0 else "event-driven"
                ),
            },
        )

        try:
            while self._death is None:
                if self._paused:
                    await self._wait_while_paused()

                    if self._death is not None:
                        break

                    continue

                if self._paused_turn() is not None:
                    if await self._resume_parked_turn():
                        continue

                    break

                # Event-driven mode (interval <= 0): no filler
                # turns -- block until the inbox has something,
                # unless the last committed turn called verbs: those
                # results are still owed to the model, so the next
                # step of the ReACT cycle runs without new input.
                if settings.agent.autonomous_interval <= 0 and not self._needs_step():
                    await self._wait_for_inbox()

                    if self._death is not None:
                        break

                    if self._paused:
                        continue

                turn_task = asyncio.create_task(self.run_turn(), name=f"turn:{self.agent_id}")

                self._turn_task = turn_task

                try:
                    await turn_task

                except asyncio.CancelledError:
                    if self._cancel_reason is CancelReason.PAUSE and self._death is None:
                        # Paused mid-turn: the turn stayed resident
                        # (and was snapshotted); wait for resume.
                        self._cancel_reason = None

                        self._turn_task = None

                        continue

                    raise

                self._turn_task = None

                if self._death is not None:
                    break

                if settings.agent.autonomous_interval > 0:
                    await asyncio.sleep(settings.agent.autonomous_interval)

                # Yield to the loop even at interval 0.
                await asyncio.sleep(0)

        finally:
            if self._death is not None:
                await self._dispose()
            else:
                await self._suspend()

    async def _wait_while_paused(self) -> None:
        while self._paused:
            self._resume_wake.clear()

            await self._resume_wake.wait()

    async def _wait_for_inbox(self) -> None:
        while self._inbox.empty():
            self._inbox_wake.clear()

            if self._inbox.empty():
                await self._inbox_wake.wait()

    def _needs_step(self) -> bool:
        """A committed turn that called verbs owes the model its
        results: the next step runs without new input."""
        turn = self.last_turn

        return bool(
            turn is not None and turn.ended_at is not None and turn.calls and turn.death is None
        )

    # ==================================================================
    # One turn
    # ==================================================================

    async def run_turn(self) -> Turn:
        turn = self._open_turn()

        try:
            persona, marker = self._persona_for_turn()

            if marker is not None:
                turn = replace(turn, persona=marker)

            self.turns[-1] = turn

            # Write 1 of 4: the turn's intent (identity, inbox,
            # deltas). Not fsynced: restore may requeue it.
            await self._save_snapshot(durable=False)

            sink.emit(
                "turn_started",
                content={
                    "agent_id": self.agent_id,
                    "turn_index": len(self.turns),
                    "inbox_count": len(turn.inbox),
                },
            )

            return await self._commit(await self._drive(turn, persona=persona))

        except asyncio.CancelledError:
            await self._on_cancel()

            raise

        except Exception as exc:
            await self._fail_turn(exc)

            return self.turns[-1]

    def _open_turn(self) -> Turn:
        """
        Observe and open the turn: drain inputs and pending deltas,
        decide the context epoch, append the resident turn.
        """
        origin: TurnOrigin = "normal"

        if not self.turns:
            origin = "first"

        elif self._history_chars() > settings.agent.history_char_limit:
            logger.info(
                "[{}] history over limit {}; clearing context",
                self.agent_id,
                settings.agent.history_char_limit,
            )

            self.turns.clear()

            self._messages = None

            origin = "context_cleared"

        turn = Turn(
            agent_id=self.agent_id,
            parent_id=(self.parent.agent_id if self.parent else None),
            depth=self.depth,
            inbox=tuple(self._drain_inbox()),
            delta=tuple(self._pending_deltas),
            origin=origin,
        )

        self._pending_deltas = []

        self.turns.append(turn)

        self._messages = None

        return turn

    async def _drive(self, turn: Turn, *, persona: str, retries: int = 0) -> Turn:
        """Phases inject -> think -> act; the caller commits."""
        if not turn.injections:
            turn = await self.engine.inject(turn, inject=self.injections.inject)

            self.turns[-1] = turn

        turn = await self._think_with_ladder(turn, persona=persona, retries=retries)

        self.turns[-1] = turn

        # Write 2 of 4: the model answered.
        await self._save_snapshot()

        turn = await self.engine.act(turn, execute=self._run_call)

        self.turns[-1] = turn

        # Write 3 of 4: the verbs that landed. Not fsynced: a
        # crash here closes the missing calls as interrupted.
        await self._save_snapshot(durable=False)

        return turn

    async def _think_with_ladder(self, turn: Turn, *, persona: str, retries: int) -> Turn:
        ladder = settings.agent.backoff

        while True:
            try:
                return await self.engine.think(
                    turn,
                    persona=persona,
                    history=self.history,
                    retries=retries,
                )

            except Exception as exc:
                if classify(exc) is not FailureKind.RETRYABLE:
                    raise

                if retries >= len(ladder):
                    raise Unrecoverable(
                        f"retries exhausted after {retries}: {type(exc).__name__}: {exc}"
                    ) from exc

                delay = ladder[retries]

                sink.emit(
                    "turn_failed",
                    content={
                        "agent_id": self.agent_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "backoff_s": delay,
                        "retry_index": retries,
                    },
                )

                logger.warning(
                    "[{}] retryable failure; retrying in {}s: {}",
                    self.agent_id,
                    delay,
                    exc,
                )

                await asyncio.sleep(delay)

                retries += 1

    async def _commit(self, turn: Turn) -> Turn:
        if self._death is not None:
            turn = replace(turn, death=self._death)

        turn = freeze_turn(turn)

        self.turns[-1] = turn

        self._messages = None

        # Write 4 of 4: the turn is complete and durable.
        await self._save_snapshot()

        sink.emit(
            "turn_done",
            content={
                "agent_id": self.agent_id,
                "turn_index": len(self.turns),
                "usage": (turn.llm.usage.model_dump() if turn.llm and turn.llm.usage else None),
                "latency_s": (turn.llm.ended_at - turn.llm.started_at if turn.llm else 0.0),
                "retries": (turn.llm.retries if turn.llm else 0),
                "calls": len(turn.calls),
                "history_chars": self._history_chars(),
            },
        )

        # The archive sees every turn that ended, in the order it
        # ended. Delivery failure is logged there and dropped.
        await self.injections.archive(turn)

        return turn

    async def _on_cancel(self) -> None:
        if self._cancel_reason is CancelReason.PAUSE and self._death is None:
            self.turns[-1] = _park(self.turns[-1], "paused")

            await self._save_snapshot()

    async def _fail_turn(self, exc: Exception) -> None:
        """
        The turn failed: park it. A subagent dies on the spot; the
        root parks and waits for a resume (never death).
        """
        if isinstance(exc, Unrecoverable):
            reason = exc.reason

            logger.warning("[{}] turn unrecoverable: {}", self.agent_id, reason)

        else:
            reason = f"{type(exc).__name__}: {exc}"

            logger.exception("[{}] turn failed: {}", self.agent_id, reason)

        self.turns[-1] = _park(self.turns[-1], reason)

        await self._save_snapshot()

        sink.emit(
            "turn_interrupted",
            content={
                "agent_id": self.agent_id,
                "turn_index": len(self.turns),
                "reason": reason,
            },
        )

        if self.depth == 0:
            self._paused = True

            sink.emit("agent_paused", content={"agent_id": self.agent_id})

            return

        self._begin_death(Death(f"unrecoverable: {reason}"))

    async def _resume_parked_turn(self) -> bool:
        """
        Continue the parked turn. Returns False when the agent is
        now dying (a failed resume on a subagent).
        """
        turn = self._paused_turn()

        assert turn is not None

        turn = replace(
            turn,
            paused=False,
            delta=tuple([*turn.delta, DeltaEvent(section="agent", line="resumed")]),
        )

        self.turns[-1] = turn

        self._messages = None

        await self._save_snapshot()

        # The same turn continues: re-announcing it lets the UI
        # overwrite the interrupted attempt (same turn_index).
        sink.emit(
            "turn_started",
            content={
                "agent_id": self.agent_id,
                "turn_index": len(self.turns),
                "inbox_count": len(turn.inbox),
            },
        )

        try:
            if turn.reply is None and not turn.calls and not turn.thinking:
                persona, _ = self._persona_for_turn()

                await self._commit(await self._drive(turn, persona=persona))

            else:
                # The model already answered: close the calls that
                # never reported, without re-executing them.
                await self._commit(turn.close_interrupted())

                sink.emit(
                    "turn_interrupted",
                    content={
                        "agent_id": self.agent_id,
                        "turn_index": len(self.turns),
                        "reason": "resumed mid-execution",
                    },
                )

        except asyncio.CancelledError:
            await self._on_cancel()

            raise

        except Exception as exc:
            await self._fail_turn(exc)

            return self._death is None

        return self._death is None

    # ==================================================================
    # Persona (re-read every turn)
    # ==================================================================

    def _persona_for_turn(self) -> tuple[str, str | None]:
        """
        Re-read the persona file every turn. A read failure raises
        (the turn fails). Returns the text to use plus, when it
        changed since the last turn, that text as the turn's
        persona marker.
        """
        try:
            text = settings.paths.persona.read_text(encoding="utf-8")

        except OSError:
            raise RuntimeError(
                f"[{self.agent_id}] persona file unreadable: {settings.paths.persona}"
            ) from None

        changed = text != self._persona_last

        if changed:
            logger.info("[{}] persona changed; new system prompt from this turn", self.agent_id)

            self._persona_last = text

        return text, (text if changed else None)

    # ==================================================================
    # Verbs
    # ==================================================================

    async def _run_call(self, call: Any) -> str:
        started = time.time()

        sink.emit(
            "verb_started",
            content={"agent_id": self.agent_id, "verb": call.name, "arguments": call.arguments},
        )

        verb = VERBS.get(call.name)

        if verb is None:
            result = f"Unknown action '{call.name}'. Available: {', '.join(VERBS)}."

            sink.emit(
                "verb_failed",
                content={
                    "agent_id": self.agent_id,
                    "verb": call.name,
                    "error": {"type": "unknown_verb", "message": result},
                },
            )

            return result

        try:
            outcome = await verb.execute(call=call, agent=self)

        except Exception as exc:
            sink.emit(
                "verb_failed",
                content={
                    "agent_id": self.agent_id,
                    "verb": call.name,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )

            raise

        content = {"agent_id": self.agent_id, "verb": call.name, "text": outcome.text}

        if outcome.detail:
            content["detail"] = outcome.detail

        sink.emit("verb_result", content=content)

        sink.emit(
            "verb_done",
            content={
                "agent_id": self.agent_id,
                "verb": call.name,
                "duration_s": time.time() - started,
            },
        )

        return outcome.text

    # ==================================================================
    # Persistence
    # ==================================================================

    async def _save_snapshot(self, *, durable: bool = True) -> None:
        if not self.turns:
            return

        await self._snapshots.save(
            AgentSnapshot(turns=tuple(self.turns)),
            durable=durable,
        )

    # ==================================================================
    # Exit: dispose (death) / suspend (shutdown)
    # ==================================================================

    async def _dispose(self) -> None:
        death = self._death

        if death is None:
            raise RuntimeError("dispose without a death record")

        # An in-flight turn (cancelled with the loop, or still
        # running) dies with the agent, never written to history.
        turn_task, self._turn_task = self._turn_task, None

        if turn_task is not None and not turn_task.done():
            turn_task.cancel()

            with contextlib.suppress(BaseException):
                await turn_task

        # Cascade: every child I hold dies with me; its loop task
        # is awaited.
        child_tasks = []

        for handle in self.child_handles():
            handle.kill()

            task = handle.task

            if task is not None:
                child_tasks.append(task)

        await asyncio.gather(*child_tasks, return_exceptions=True)

        # Notify upstream: a finish report rides the inbox (it
        # carries content); a bare death rides the holder's system
        # event ledger, surfacing in their next <delta> section.
        # Either way the child slot is released here.
        if self.parent is not None:
            if death.report is not None:
                self.parent._deliver(
                    InboxMessage(
                        kind="subagent",
                        source=self.agent_id,
                        text=death.report,
                        event="finish",
                    )
                )

            else:
                self.parent.note_delta("subagent", f"-{self.agent_id} died: {death.reason}")

            self.parent._children.pop(self.agent_id, None)

        self.injections.forget(self.agent_id)

        self.book.unregister(self.agent_id)

        self._snapshots.remove(self.agent_id)

        sink.emit(
            "agent_exited",
            content={"agent_id": self.agent_id, "reason": death.reason, "report": death.report},
        )

        logger.info(
            "[{}] stopped after {} turns ({})", self.agent_id, len(self.turns), death.reason
        )

    async def _suspend(self) -> None:
        """Process shutdown: keep the turn list, end quietly."""
        turn_task, self._turn_task = self._turn_task, None

        if turn_task is not None and not turn_task.done():
            turn_task.cancel()

            with contextlib.suppress(BaseException):
                await turn_task

        await self._save_snapshot()

        self.book.unregister(self.agent_id)

        logger.info("[{}] suspended after {} turns", self.agent_id, len(self.turns))

    # ==================================================================
    # Small helpers
    # ==================================================================

    def _drain_inbox(self) -> list[InboxMessage]:
        drained: list[InboxMessage] = []

        while True:
            try:
                drained.append(self._inbox.get_nowait())

            except asyncio.QueueEmpty:
                return drained

    def _history_chars(self) -> int:
        """History size, including replayed reasoning (visible and encrypted)."""
        return sum(
            len(message.content or "")
            + sum(len(block.text) + len(block.data or "") for block in message.thinking)
            for message in self.history
        )


# ----------------------------------------------------------------------
# Module-level pure helpers
# ----------------------------------------------------------------------


def _park(turn: Turn, reason: str) -> Turn:
    return replace(
        turn,
        paused=True,
        delta=tuple([*turn.delta, DeltaEvent(section="agent", line=reason)]),
    )
