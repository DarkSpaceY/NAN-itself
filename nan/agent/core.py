"""
The Agent: one homogeneous loop, root or subagent.

    1 agent = 1 asyncio Task. The loop is a fold: it derives the
    agent's Status, folds it through `control.transition`, and
    performs the action that comes back. Death, pause, shutdown,
    the ReACT continuation and the autonomous cadence are decided
    there, once; the loop holds no `if death` / `if paused`
    between awaits.

    A turn opens when the agent observes (drains its inbox,
    collects its deltas) and stays resident in `turns` until it
    commits. Its `phase` is the one progress marker -- inject ->
    think -> act -> commit -- so a paused, failed or restored turn
    continues from that phase. Nothing infers progress from the
    payload shape.

    The Turn list IS the conversation; history is the rendered
    turns of the current context epoch.

    Retry: a retryable failure re-runs the SAME turn's model call
    along the backoff ladder, counted in LLMCall.retries. When the
    ladder runs out, a subagent dies; the root parks the turn
    instead of dying.

    Persistence: a turn is durable at its intent, after the model
    answers, after EACH verb result, and at commit, so a stop
    restarts from the last complete fact rather than from nothing.

    Pause is one latch on every agent in the subtree: a pause
    cancels the in-flight turn where it stands and the whole tree
    waits; a resume lifts it and each turn continues from its
    phase. Pause is process-scoped -- a restart lifts it.

    Exit has two shapes, and both cascade over the subtree. Shutdown
    (SIGTERM/SIGINT) suspends: every agent persists its in-flight
    turn, nothing dies, no notices. Death (finish/kill/unrecoverable)
    disposes: every agent drops its snapshot, reports upstream and
    releases its children.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from nan.agent.control import Action, Status, transition
from nan.agent.engine import TurnEngine
from nan.agent.verbs import CHILD_VERBS, ROOT_VERBS, tool_definitions
from nan.config import settings
from nan.events import sink
from nan.llm import Message
from nan.recovery import (
    AgentSnapshot,
    FailureKind,
    SnapshotStore,
    Unrecoverable,
    classify,
)
from nan.turn import (
    Death,
    DeltaEvent,
    InboxMessage,
    Turn,
    TurnOrigin,
    freeze_turn,
)

if TYPE_CHECKING:
    from nan.agent.sandbox import Sandbox
    from nan.injection import InjectionHub
    from nan.llm import LLMProvider


class Agent:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        injections: InjectionHub,
        sandbox: Sandbox,
        parent: Agent | None = None,
        agent_id: str | None = None,
        snapshots: SnapshotStore | None = None,
    ) -> None:
        # --- collaborators (handed in) ---
        self.llm = llm
        self.engine = TurnEngine(llm)
        self.injections = injections
        self.parent = parent  # upstream channel for send
        self.sandbox = sandbox  # execution-level sandbox for the shell verb
        self._snapshots = snapshots or SnapshotStore(settings.paths.snapshots)

        # --- identity: root or child is ONE fact, `parent` ---
        # It decides the verb set (a root never exits, so it cannot
        # finish) and the terminal of a failed turn: the root parks
        # for a resume, a child dies.
        self.verbs = ROOT_VERBS if parent is None else CHILD_VERBS
        self.tools = tool_definitions(self.verbs)

        # --- identity ---
        self.agent_id = agent_id or uuid.uuid4().hex[:8]
        self.depth: int = 0 if parent is None else parent.depth + 1

        # --- persona: boot read ---
        # The file is CORE: an unreadable persona refuses startup.
        self._persona_last = self._read_persona()

        # --- children: the spawn tree edge, keyed by child id ---
        # `Turn.parent_id` is the durable fact; this map is the live
        # projection, rebuilt by restore.
        self.children: dict[str, Agent] = {}

        # --- inbox: delivery + the single wake source ---
        # Not-yet-observed messages, in arrival order. It rides
        # every snapshot, so a message delivered before the next
        # turn opens survives a stop.
        self._inbox: list[InboxMessage] = []
        self._wake = asyncio.Event()

        # --- tasks: the loop, and the one cancellable in-flight turn ---
        self._task: asyncio.Task[None] | None = None
        self._turn_task: asyncio.Task[Turn] | None = None

        # --- turns: the only runtime record ---
        # `_messages` is a cache of the rendered history (turns of
        # the current context epoch); it is dropped whenever turns
        # change.
        self.turns: list[Turn] = []
        self._messages: list[Message] | None = None

        # Deltas that arrived while no turn was open. They ride the
        # next turn's observation. Transient by design: a crash may
        # lose them, a turn list never does.
        self._pending_deltas: list[DeltaEvent] = []

        # --- control latches (read by `_status`, written by the
        # verbs, the gateway and the process) ---
        self._death: Death | None = None  # first exit record wins
        self._paused = False  # the whole agent is frozen
        self._stopping = False  # shutdown: persist, no death
        self._started = False

        # The autonomous cadence: True when a turn may open. It is
        # spent by opening a turn and earned back by one interval's
        # sleep (see control.transition).
        self._paced = True

        # Set when pause cancels the in-flight turn, read by the
        # cancellation handlers, cleared once consumed. A latch,
        # not `_paused`: a resume may lift the freeze before the
        # cancellation is delivered, yet the turn must still park.
        self._pause_cancel = False

    # ==================================================================
    # Identity
    # ==================================================================

    def start(self) -> None:
        """Start the loop task."""
        self._task = asyncio.create_task(self.loop(), name=f"agent:{self.agent_id}")

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
        committed turn of the current context epoch. The open turn
        is unrendered and therefore absent.
        """
        if self._messages is None:
            self._messages = self._rebuild_history()

        return self._messages

    def _rebuild_history(self) -> list[Message]:
        if not self.turns:
            return []

        epoch = self.turns[-1].epoch

        return [
            message
            for turn in self.turns
            if turn.epoch == epoch and turn.rendered is not None
            for message in turn.rendered
        ]

    # ==================================================================
    # Inbox
    # ==================================================================

    def deliver(self, message: InboxMessage) -> None:
        """
        Put one message in the inbox and wake the loop. The sender
        names the edge, so the kind is decided there (see
        `InboxMessage.from_user` / `from_parent` / `from_child`).
        Safe from any task.
        """
        self._inbox.append(message)

        self._wake.set()

    # ==================================================================
    # Spawning
    # ==================================================================

    async def spawn_child(self, task: str) -> tuple[Agent | None, str | None]:
        """
        Create, make durable and start a child; return (child,
        error). The caller (spawn verb) renders the error for the
        model. The task is delivered before the snapshot is written,
        so the child survives a stop from birth.
        """
        if settings.agent.max_depth >= 0 and self.depth + 1 > settings.agent.max_depth:
            return None, f"Max depth exceeded: {self.depth + 1} > {settings.agent.max_depth}"

        if settings.agent.max_children >= 0 and len(self.children) >= settings.agent.max_children:
            return None, f"Max children reached: {settings.agent.max_children}"

        # A child shares the parent's collaborators and inherits
        # everything else from the process-wide settings.
        child = Agent(
            llm=self.llm,
            injections=self.injections,
            sandbox=self.sandbox,
            parent=self,
            snapshots=self._snapshots,
        )

        self.children[child.agent_id] = child

        # "spawn == spawn then send": the task is delivered as the
        # child's first inbox message, tagged from its parent.
        child.deliver(InboxMessage.from_parent(task, source=self.agent_id))

        self.note_delta("subagent", f"+{child.agent_id} spawned")

        # Durable before it runs: a snapshot with no turns is a
        # spawned agent that has not observed anything yet.
        try:
            await child.persist()

        finally:
            # Started even when this spawn was cancelled mid-write
            # (pause or shutdown): the child is registered and holds
            # its task, so an unstarted one would be a zombie. A child
            # born into a frozen subtree is frozen too.
            if self._paused:
                child.pause()

            child.start()

        return child, None

    # ==================================================================
    # Death / pause / shutdown
    # ==================================================================

    def note_delta(self, section: str, line: str) -> None:
        """
        Record a system-state event line ("subagent", "-j died:
        ..."), drained into the next turn's <delta> section under
        the given tag. Passive: it never wakes the loop.
        """
        self._pending_deltas.append(DeltaEvent(section=section, line=line))

    def _note(self, line: str, *, section: str = "agent") -> None:
        """
        Attach one runtime fact to the turn it happened during: the
        open turn when there is one, otherwise the pending channel
        the next observation drains. A committed turn is frozen and
        never takes a new line.
        """
        turn = self.last_turn

        if turn is not None and turn.phase != "committed":
            self.turns[-1] = turn.noted(line, section=section)

        else:
            self.note_delta(section, line)

    @property
    def _exiting(self) -> bool:
        """The agent is leaving: death, or shutdown."""
        return self._death is not None or self._stopping

    def _begin_death(self, death: Death) -> bool:
        """Record the exit; the first record wins (idempotent)."""
        if self._death is not None:
            return False

        self._death = death

        self._wake.set()

        return True

    def request_death(self, reason: str) -> None:
        """
        Kill this agent now: record the exit and cancel the loop
        (the loop's disposal does the rest: in-flight turn dropped,
        cascade, notice, snapshot removed).
        """
        if not self._begin_death(Death(reason)):
            return

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
        if self._exiting:
            return

        self._stopping = True

        self._wake.set()

        self._cancel_tasks()

    def pause(self) -> None:
        """
        Freeze the agent and its whole subtree: every in-flight turn
        is cancelled where it stands (its inputs and facts stay on
        the turn, snapshotted) and every loop waits for resume.
        Idempotent.
        """
        if self._paused or self._exiting:
            return

        self._park()

        if self._turn_task is not None:
            self._pause_cancel = True

            self._turn_task.cancel()

        for child in list(self.children.values()):
            child.pause()

    def _park(self) -> None:
        """The one pause transition: freeze and announce it."""
        self._paused = True

        sink.emit("agent_paused", content={"agent_id": self.agent_id})

        self._wake.set()

    def resume(self) -> None:
        """
        Lift the freeze over the whole subtree. Each resident turn
        continues from its phase: an unanswered one re-runs its
        model call, an answered one closes its unfinished calls as
        interrupted. Idempotent.
        """
        if not self._paused:
            return

        self._paused = False

        sink.emit("agent_resumed", content={"agent_id": self.agent_id})

        self._wake.set()

        for child in list(self.children.values()):
            child.resume()

    @property
    def paused(self) -> bool:
        """The agent is frozen: no turn runs until resume."""
        return self._paused

    def _cancel_tasks(self) -> None:
        if self._turn_task is not None:
            self._turn_task.cancel()

        if self._started and self._task is not None:
            self._task.cancel()

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
            while True:
                action = transition(self._status(), interval=settings.agent.autonomous_interval)

                if action == "stop":
                    break

                await self._perform(action)

        finally:
            if self._death is not None:
                await self._dispose()
            else:
                await self._suspend()

    def _status(self) -> Status:
        """The latches plus the resident turn: everything the fold reads."""
        return Status(
            death=self._death,
            stopping=self._stopping,
            paused=self._paused,
            open_turn=self._resident_turn() is not None,
            owed_step=self._owes_a_step(),
            input_ready=bool(self._inbox),
            paced=self._paced,
        )

    def _resident_turn(self) -> Turn | None:
        """The open turn, if any: the last turn, before its commit."""
        turn = self.last_turn

        return None if turn is None or turn.phase == "committed" else turn

    def _owes_a_step(self) -> bool:
        """A committed turn that called verbs owes the model its
        results: the next step runs without new input."""
        turn = self.last_turn

        return bool(
            turn is not None and turn.phase == "committed" and turn.calls and turn.death is None
        )

    async def _perform(self, action: Action) -> None:
        """Carry out one action; the fold re-reads the agent after it."""
        if action == "wait":
            # The check in `_status` and this clear are one
            # synchronous step, so no delivery can slip between.
            self._wake.clear()

            await self._wake.wait()

            return

        if action == "pace":
            # The cadence: wait it out, then the fold opens the turn.
            await asyncio.sleep(settings.agent.autonomous_interval)

            self._paced = True

            return

        # A turn runs here, and the cadence is spent with it -- a
        # resumed turn counts as a turn too.
        self._paced = False

        turn_task = asyncio.create_task(
            self._resume_turn() if action == "resume" else self.run_turn(),
            name=f"turn:{self.agent_id}",
        )

        self._turn_task = turn_task

        try:
            await turn_task

        except asyncio.CancelledError:
            # Only a pause leaves the loop alive: the turn stays
            # resident and the next fold continues it. An exit
            # (death or shutdown) re-raises, and the loop unwinds.
            if self._exiting or not self._pause_cancel:
                raise

        finally:
            self._turn_task = None
            self._pause_cancel = False

    # ==================================================================
    # One turn
    # ==================================================================

    async def run_turn(self) -> Turn:
        """Observe, open, run and commit one turn."""
        return await self._advance(self._open_turn(), resume=False)

    async def _resume_turn(self) -> Turn:
        """Continue the resident turn from its phase."""
        turn = self._resident_turn()

        assert turn is not None

        # The same turn continues: the `resumed` line rides its
        # observation and re-announcing it lets the UI overwrite the
        # interrupted attempt (same turn index).
        turn = turn.noted("resumed")

        self.turns[-1] = turn

        return await self._advance(turn, resume=True)

    def _open_turn(self) -> Turn:
        """
        Observe and open the turn: drain inputs and pending deltas,
        decide the context epoch, append the resident turn.
        """
        previous = self.last_turn

        origin: TurnOrigin = "normal"
        epoch = previous.epoch if previous is not None else 0

        if previous is None:
            origin = "first"

        elif self._history_chars() > settings.agent.history_char_limit:
            logger.info(
                "[{}] history over limit {}; clearing context",
                self.agent_id,
                settings.agent.history_char_limit,
            )

            self.turns.clear()

            origin = "context_cleared"
            epoch += 1

        turn = Turn(
            agent_id=self.agent_id,
            parent_id=(self.parent.agent_id if self.parent else None),
            depth=self.depth,
            epoch=epoch,
            origin=origin,
            inbox=tuple(self._drain_inbox()),
            delta=tuple(self._pending_deltas),
        )

        self._pending_deltas = []

        self.turns.append(turn)

        self._messages = None

        return turn

    async def _advance(self, turn: Turn, *, resume: bool) -> Turn:
        """
        Run the remaining phases of `turn` and commit it. A
        cancelled turn is parked where it stands (pause); a failed
        one is parked (root) or dies (child).
        """
        try:
            persona, marker = self._persona_for_turn()

            if marker is not None:
                turn = replace(turn, persona=marker)

            self.turns[-1] = turn

            # Write 1 of 4: the turn's intent (identity, inbox,
            # deltas), so a stop can requeue it.
            await self.persist()

            sink.emit(
                "turn_started",
                content={
                    "agent_id": self.agent_id,
                    "turn_index": len(self.turns),
                    "inbox_count": len(turn.inbox),
                },
            )

            turn = await (
                self._continue(turn, persona=persona)
                if resume
                else self._drive(turn, persona=persona)
            )

            return await self._commit(turn)

        except asyncio.CancelledError:
            # A pause cancel parks the turn where it stands. Any
            # other cancel ends the loop, and the turn is dropped
            # with the agent instead.
            if not self._exiting and self._pause_cancel:
                await self._park_turn()

            raise

        except Exception as exc:
            await self._fail_turn(exc)

            return self.turns[-1]

    async def _drive(self, turn: Turn, *, persona: str) -> Turn:
        """Phases inject -> think -> act, persisting after each."""
        if turn.phase == "observed":
            turn = await self.engine.inject(turn, inject=self.injections.inject)

            self.turns[-1] = turn

        turn = await self._think(turn, persona=persona)

        self.turns[-1] = turn

        # Write 2 of 4: the model answered.
        await self.persist()

        # Each verb result is recorded as it lands (write 3 of 4),
        # so a crash mid-batch loses only the call in flight.
        turn = await self.engine.act(turn, execute=self._run_call, record=self._record_result)

        self.turns[-1] = turn

        await self.persist()

        return turn

    async def _record_result(self, turn: Turn) -> None:
        """The act phase's durability hook: this is the turn as it
        now stands, with one more verb result on it."""
        self.turns[-1] = turn

        await self.persist()

    async def _continue(self, turn: Turn, *, persona: str) -> Turn:
        """Resume a resident turn from its phase, never re-running verbs."""
        if turn.phase in ("observed", "injected"):
            # No verb ran yet: the model call is safe to run again.
            return await self._drive(turn, persona=persona)

        if turn.phase == "acted":
            # Every verb reported: only the commit is owed.
            return turn

        # "thought": the model answered, but its verbs may have run
        # partially. Re-executing them is not safe, so what never
        # reported is closed as interrupted (the model sees it).
        reported = {result.call_id for result in turn.results}

        if any(call.id not in reported for call in turn.calls):
            sink.emit(
                "turn_interrupted",
                content={
                    "agent_id": self.agent_id,
                    "turn_index": len(self.turns),
                    "reason": "resumed mid-execution",
                },
            )

        return turn.close_interrupted()

    async def _think(self, turn: Turn, *, persona: str) -> Turn:
        """
        One model call, retried along the backoff ladder. Only the
        call is retried (verbs are at-most-once); a ladder that
        runs out is `Unrecoverable`.
        """
        ladder = settings.agent.backoff
        retries = 0

        while True:
            try:
                return await self.engine.think(
                    turn,
                    persona=persona,
                    history=self.history,
                    tools=self.tools,
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
        """Freeze, persist and announce the turn; archive it."""
        if self._death is not None:
            turn = replace(turn, death=self._death)

        turn = replace(
            turn,
            phase="committed",
            ended_at=turn.ended_at if turn.ended_at is not None else time.time(),
        )

        turn = freeze_turn(turn)

        self.turns[-1] = turn

        self._messages = None

        # Write 4 of 4: the turn is complete and durable.
        await self.persist()

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

    async def _park_turn(self) -> None:
        """Pause cancelled the turn: keep it exactly where it stands."""
        self._note("paused")

        await self.persist()

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

        self._note(reason)

        await self.persist()

        sink.emit(
            "turn_interrupted",
            content={
                "agent_id": self.agent_id,
                "turn_index": len(self.turns),
                "reason": reason,
            },
        )

        # A root parks the turn and waits to be resumed; a child has
        # nobody to resume it, so it dies.
        if self.parent is not None:
            self._begin_death(Death(f"unrecoverable: {reason}"))

            return

        self._park()

    # ==================================================================
    # Persona (re-read every turn)
    # ==================================================================

    def _persona_for_turn(self) -> tuple[str, str | None]:
        """
        Re-read the persona; a content change is the one legal
        system-prompt invalidation and rides the turn as its marker.
        """
        text = self._read_persona()

        if text == self._persona_last:
            return text, None

        logger.info("[{}] persona changed; new system prompt from this turn", self.agent_id)

        self._persona_last = text

        return text, text

    def _read_persona(self) -> str:
        """The persona file, read at boot and re-read every turn."""
        try:
            return settings.paths.persona.read_text(encoding="utf-8")

        except OSError as exc:
            raise RuntimeError(f"persona file unreadable: {settings.paths.persona}") from exc

    # ==================================================================
    # Verbs
    # ==================================================================

    async def _run_call(self, call: Any) -> str:
        started = time.time()

        sink.emit(
            "verb_started",
            content={"agent_id": self.agent_id, "verb": call.name, "arguments": call.arguments},
        )

        verb = self.verbs.get(call.name)

        if verb is None:
            return self._verb_error(
                call,
                "unknown_verb",
                f"Unknown action '{call.name}'. Available: {', '.join(self.verbs)}.",
            )

        try:
            outcome = await verb.execute(call=call, agent=self)

        except Exception as exc:
            # A failed verb is the model's to correct: its error
            # comes back as that call's result, never as a turn
            # failure.
            logger.exception("[{}] verb {} failed", self.agent_id, call.name)

            return self._verb_error(call, type(exc).__name__, f"{type(exc).__name__}: {exc}")

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

    def _verb_error(self, call: Any, error_type: str, text: str) -> str:
        """Report a failed verb to observers; the text is its result."""
        sink.emit(
            "verb_failed",
            content={
                "agent_id": self.agent_id,
                "verb": call.name,
                "error": {"type": error_type, "message": text},
            },
        )

        return text

    # ==================================================================
    # Persistence
    # ==================================================================

    async def persist(self) -> None:
        """
        Write this agent's whole state -- identity, turns and the
        unobserved inbox -- in one atomic, fsynced file.

        One writer, best effort: a failed write is logged and the
        last good snapshot stands. Losing the process over a disk
        error would lose far more than the one phase.
        """
        snapshot = AgentSnapshot(
            agent_id=self.agent_id,
            parent_id=(self.parent.agent_id if self.parent is not None else None),
            depth=self.depth,
            turns=tuple(self.turns),
            pending=tuple(self._inbox),
        )

        try:
            await self._snapshots.save(snapshot)

        except Exception:
            logger.exception(
                "[{}] snapshot write failed; the last good snapshot stands", self.agent_id
            )

    # ==================================================================
    # Exit: dispose (death) / suspend (shutdown)
    # ==================================================================

    async def _halt(self) -> None:
        """Cancel the in-flight turn and await what it leaves."""
        turn_task, self._turn_task = self._turn_task, None

        if turn_task is not None and not turn_task.done():
            turn_task.cancel()

            with contextlib.suppress(BaseException):
                await turn_task

    async def _stop_children(self, stop: Callable[[Agent], None]) -> None:
        """The exit cascades over the subtree: stop every child, then
        await its loop (each child repeats this over its own)."""
        children = list(self.children.values())

        for child in children:
            stop(child)

        await asyncio.gather(
            *(child.task for child in children if child.task is not None),
            return_exceptions=True,
        )

    async def _dispose(self) -> None:
        death = self._death

        if death is None:
            raise RuntimeError("dispose without a death record")

        # An in-flight turn dies with the agent, never written to
        # history; the subtree dies with me.
        await self._halt()

        await self._stop_children(
            lambda child: child.request_death(f"killed by holder {self.agent_id}")
        )

        self.children.clear()

        # Notify upstream: a finish report rides the inbox (it
        # carries content); a bare death rides the parent's system
        # event ledger, surfacing in their next <delta> section.
        # Either way the child slot is released here.
        if self.parent is not None:
            if death.report is not None:
                self.parent.deliver(
                    InboxMessage.from_child(death.report, source=self.agent_id, event="finish")
                )

            else:
                self.parent.note_delta("subagent", f"-{self.agent_id} died: {death.reason}")

            self.parent.children.pop(self.agent_id, None)

        self.injections.forget(self.agent_id)

        self._snapshots.remove(self.agent_id)

        sink.emit(
            "agent_exited",
            content={"agent_id": self.agent_id, "reason": death.reason, "report": death.report},
        )

        logger.info(
            "[{}] stopped after {} turns ({})", self.agent_id, len(self.turns), death.reason
        )

    async def _suspend(self) -> None:
        """Process shutdown: persist the subtree and end quietly."""
        await self._halt()

        await self._stop_children(lambda child: child.suspend())

        await self.persist()

        logger.info("[{}] suspended after {} turns", self.agent_id, len(self.turns))

    # ==================================================================
    # Small helpers
    # ==================================================================

    def _drain_inbox(self) -> list[InboxMessage]:
        drained, self._inbox = self._inbox, []

        return drained

    def _history_chars(self) -> int:
        """History size, including replayed reasoning (visible and encrypted)."""
        return sum(
            len(message.content or "")
            + sum(len(block.text) + len(block.data or "") for block in message.thinking)
            for message in self.history
        )
