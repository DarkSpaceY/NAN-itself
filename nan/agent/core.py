"""
The Agent: one homogeneous loop, root or subagent.

    1 agent = 1 asyncio Task. The loop runs turns continuously,
    paced by autonomous_interval (incoming messages are drained
    at turn boundaries).

    A turn: drain inbox -> collect injections -> one model call
    -> run verbs positionally -> freeze rendered messages (the
    history is add-only until the char limit forces a full
    clear).

    Pause/resume: pausing cancels the in-flight turn task; the
    interrupted turn is discarded whole and drained inputs are
    re-queued, so resume opens a clean turn.

    Death has one path. finish() marks the report and lets the
    loop end normally; kill/cascade/shutdown cancel the task.
    Either way _teardown runs exactly once: children are killed
    (cascade along held handles) and awaited, and the parent's
    inbox receives a finish report or a death notice.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger

from nan.agent.engine import TurnEngine, next_backoff
from nan.agent.handle import Handle
from nan.agent.verbs import VERBS
from nan.config import settings
from nan.events import sink
from nan.llm import Message
from nan.turn import (
    Death,
    DeltaEvent,
    InboxKind,
    InboxMessage,
    Turn,
    TurnOrigin,
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
    ) -> None:
        # --- collaborators (handed in) ---
        self.book = book
        self.llm = llm
        self.engine = TurnEngine(llm)
        self.injections = injections
        self.parent = parent  # upstream channel for send
        self.sandbox = sandbox  # execution-level sandbox for the shell verb

        # --- identity ---
        self.agent_id = uuid.uuid4().hex[:8]
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

        # --- per-turn accumulation (drained at each turn boundary) ---
        # Delta events (subagent lifecycle) render into the next
        # turn's <delta> sections; history is add-only and cleared
        # wholesale once over the char limit.
        self._delta_events: list[DeltaEvent] = []
        self._messages: list[Message] = []
        self._cleared = False  # context was cleared before this turn
        self.last_turn: Turn | None = None
        self.cycles = 0

        # --- lifecycle: the exit record, and the start flag ---
        self._death: Death | None = None  # first exit record wins (latch)
        self._started = False

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
        self._delta_events.append(DeltaEvent(section=section, line=line))

    def _begin_death(self, death: Death) -> bool:
        """Record the exit; the first record wins (idempotent)."""
        if self._death is not None:
            return False

        self._death = death

        return True

    def request_death(self, reason: str) -> None:
        """
        Kill this agent now: record the exit and cancel the loop
        (the loop's teardown does the rest: in-flight turn
        discarded, cascade, notice). A death recorded before the
        loop takes its first step is honored on entry instead.
        """
        if not self._begin_death(Death(reason)):
            return

        if self._started and self._task is not None:
            self._task.cancel()

    def finish(self, report: str) -> None:
        """
        Record the final report (FinishVerb). The loop ends at
        the next boundary; teardown delivers it upstream. The
        record latches: a second finish is a no-op (the first
        report wins).
        """
        self._begin_death(Death("finished", report))

    def pause(self) -> None:
        """
        Pause the loop: the in-flight turn task is cancelled and
        discarded whole; its drained inputs ride back into the
        inbox, so resume opens a clean turn. Idempotent.
        """
        if self._paused:
            return

        self._paused = True

        sink.emit("agent_paused", content={"agent_id": self.agent_id})

        if self._turn_task is not None:
            self._turn_task.cancel()

    def resume(self) -> None:
        """Lift the pause; the loop opens a clean turn. Idempotent."""
        if not self._paused:
            return

        self._paused = False

        sink.emit("agent_resumed", content={"agent_id": self.agent_id})

        self._resume_wake.set()

    @property
    def paused(self) -> bool:
        """Current pause flag, for the gateway's hello snapshot."""
        return self._paused

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

        backoff_index = 0

        try:
            while self._death is None:
                if self._paused:
                    await self._wait_while_paused()

                    if self._death is not None:
                        break

                # Event-driven mode (interval <= 0): no filler
                # turns -- block until the inbox has something,
                # unless the last committed turn called verbs: those
                # results are still owed to the model, so the next
                # step of the ReACT cycle runs without new input.
                if settings.agent.autonomous_interval <= 0 and not (
                    self.last_turn is not None and self.last_turn.calls
                ):
                    await self._wait_for_inbox()

                    if self._death is not None:
                        break

                    if self._paused:
                        continue

                turn_task = asyncio.create_task(
                    self.run_turn(retries=backoff_index), name=f"turn:{self.agent_id}"
                )

                self._turn_task = turn_task

                try:
                    await turn_task

                except asyncio.CancelledError:
                    if turn_task.cancelled() and _self_cancel_pending() == 0:
                        # Paused mid-turn: the turn was discarded
                        # whole (run_turn re-queued its inputs);
                        # wait for the resume, then go clean.
                        self._turn_task = None

                        continue

                    # Death: teardown awaits the unwinding turn
                    # through this handle.
                    raise

                except Exception as exc:
                    self._turn_task = None

                    delay, backoff_index = next_backoff(settings.agent.backoff, backoff_index)

                    sink.emit(
                        "turn_failed",
                        content={
                            "agent_id": self.agent_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "backoff_s": delay,
                            "retry_index": backoff_index - 1,
                        },
                    )

                    logger.exception("[{}] Turn failed; backing off {}s", self.agent_id, delay)

                    await asyncio.sleep(delay)

                    continue

                self._turn_task = None

                backoff_index = 0
                self.cycles += 1

                if self._death is not None:
                    break

                if settings.agent.autonomous_interval > 0:
                    await asyncio.sleep(settings.agent.autonomous_interval)

                # Yield to the loop even at interval 0.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            # Cancellation that bypassed request_death (event loop
            # shutdown) is still an exit: record it like any other.
            if self._death is None:
                self._death = Death("cancelled")

            raise

        finally:
            await self._teardown()

    async def _wait_while_paused(self) -> None:
        while self._paused:
            await self._resume_wake.wait()

            self._resume_wake.clear()

    async def _wait_for_inbox(self) -> None:
        while self._inbox.empty():
            self._inbox_wake.clear()

            if self._inbox.empty():
                await self._inbox_wake.wait()

    async def run_turn(self, *, retries: int = 0) -> Turn:
        # -- observe: drain inputs, build the arrival skeleton ------
        drained: list[InboxMessage] = list(self._drain_inbox())

        pending_deltas: list[DeltaEvent] = list(self._delta_events)

        self._delta_events = []

        if self.cycles == 0:
            origin: TurnOrigin = "first"
        elif self._cleared:
            origin = "context_cleared"
        else:
            origin = "normal"

        # The flag is consumed by this turn; a rollback puts it
        # back, so the next turn still reports the context epoch.
        was_cleared, self._cleared = self._cleared, False

        persona, persona_marker = self._persona_for_turn()

        skeleton = Turn(
            agent_id=self.agent_id,
            parent_id=(self.parent.agent_id if self.parent else None),
            depth=self.depth,
            inbox=tuple(drained),
            delta=tuple(pending_deltas),
            origin=origin,
            persona=persona_marker,
        )

        sink.emit(
            "turn_started",
            content={
                "agent_id": self.agent_id,
                "turn_index": self.cycles + 1,
                "inbox_count": len(drained),
            },
        )

        # -- pipeline: collect -> think -> act -> freeze ------------
        try:
            turn = await self.engine.run_turn(
                skeleton,
                persona=persona,
                history=self._messages,
                inject=self.injections.inject,
                execute=self._run_call,
                retries=retries,
            )

        except BaseException as exc:
            # The turn dies whole (pause, agent death or a failure
            # inside the pipeline): nothing rendered, nothing in
            # history; drained inputs go back for the next (clean)
            # turn, and the state consumed above is restored.
            if isinstance(exc, asyncio.CancelledError):
                sink.emit(
                    "turn_cancelled",
                    content={"agent_id": self.agent_id, "requeued_count": len(drained)},
                )

            self._cleared = was_cleared

            for message in drained:
                self._deliver(message)

            self._delta_events = [*pending_deltas, *self._delta_events]

            raise

        # -- apply: the engine returned whole, commit the state -----
        if self._death is not None:
            turn = replace(turn, death=self._death)

        self.last_turn = turn

        assert turn.rendered is not None

        self._messages.extend(turn.rendered)

        self._clear_history_over_limit()

        sink.emit(
            "turn_done",
            content={
                "agent_id": self.agent_id,
                "turn_index": self.cycles + 1,
                "usage": (turn.llm.usage.model_dump() if turn.llm and turn.llm.usage else None),
                "latency_s": (turn.llm.ended_at - turn.llm.started_at if turn.llm else 0.0),
                "retries": (turn.llm.retries if turn.llm else 0),
                "calls": len(turn.calls),
                "history_chars": sum(len(m.content or "") for m in self._messages),
            },
        )

        # The archive sees every turn that ended, in the order it
        # ended. Delivery failure is logged there and dropped.
        await self.injections.archive(turn)

        return turn

    # ==================================================================
    # Persona (re-read every turn)
    # ==================================================================

    def _persona_for_turn(self) -> tuple[str, str | None]:
        """
        Re-read the persona file every turn. A read failure raises
        (the turn fails and backs off). Returns the text to use
        plus, when it changed since the last turn, that text as
        the turn's persona marker.
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
    # Teardown (the single death path)
    # ==================================================================

    async def _teardown(self) -> None:
        death = self._death

        if death is None:
            raise RuntimeError("teardown without a death record")

        # An in-flight turn (cancelled with the loop, or still
        # running) is discarded, never written to history.
        turn_task, self._turn_task = self._turn_task, None

        if turn_task is not None:
            turn_task.cancel()

            with contextlib.suppress(asyncio.CancelledError, Exception):
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

        sink.emit(
            "agent_exited",
            content={"agent_id": self.agent_id, "reason": death.reason, "report": death.report},
        )

        logger.info("[{}] stopped after {} cycles ({})", self.agent_id, self.cycles, death.reason)

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

    def _clear_history_over_limit(self) -> None:
        chars = sum(len(m.content or "") for m in self._messages)

        if chars > settings.agent.history_char_limit:
            logger.info(
                "[{}] history {} chars over limit {}; clearing",
                self.agent_id,
                chars,
                settings.agent.history_char_limit,
            )

            self._messages.clear()

            # The next turn opens a new context epoch.
            self._cleared = True


# ----------------------------------------------------------------------
# Module-level pure helpers
# ----------------------------------------------------------------------


def _self_cancel_pending() -> int:
    task = asyncio.current_task()

    return task.cancelling() if task is not None else 0
