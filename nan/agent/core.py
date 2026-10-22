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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nan.agent.engine import DEFAULT_BACKOFF, TurnEngine, next_backoff
from nan.agent.verbs import VERBS, tool_definitions
from nan.events import sink
from nan.llm import LLMRequest, LLMResponse, Message
from nan.turn import (
    DeltaEvent,
    InboxKind,
    InboxMessage,
    LLMCall,
    Turn,
    freeze_turn,
    render_turn,
)

if TYPE_CHECKING:
    from nan.agent.handle import Book, Handle
    from nan.agent.sandbox import Sandbox
    from nan.injection import InjectionHub
    from nan.llm import LLMProvider

HISTORY_CHAR_LIMIT = 100_000


@dataclass
class AgentConfig:
    max_depth: int = 3  # -1: unlimited
    max_children: int = 5  # -1: unlimited
    autonomous_interval: float = 0.0
    shell_timeout: float = 30.0  # shell verb execution timeout
    shell_output_limit: int = 20_000  # shell verb head+tail truncation size
    backoff: tuple[float, ...] = DEFAULT_BACKOFF
    history_char_limit: int = HISTORY_CHAR_LIMIT


class Agent:
    def __init__(
        self,
        *,
        book: Book,
        llm: LLMProvider,
        injections: InjectionHub,
        workspace: Path,
        persona_path: Path,
        sandbox: Sandbox,
        parent: Agent | None = None,
        config: AgentConfig,
    ) -> None:
        self.book = book
        self.llm = llm
        self.engine = TurnEngine(llm)
        self.injections = injections
        self.workspace = workspace

        # Global persona: the file is a CORE file, config-grade.
        # Read at boot (an unreadable persona refuses startup)
        # and re-read every turn; a mid-run read failure fails
        # the turn (backoff + retry). A content change is a
        # system-prompt invalidation point (new prompt from that
        # turn on; unchanged content is a cache no-op).
        self.persona_path = persona_path

        try:
            self._persona_last = persona_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"persona file unreadable at boot: {persona_path}") from exc

        self.sandbox = sandbox  # execution-level sandbox for the shell verb

        self.parent = parent  # upstream channel for send

        self.config = config

        self.agent_id = uuid.uuid4().hex[:8]
        self.depth: int = 0 if parent is None else parent.depth + 1

        # Handles I hold for my children. A handle stays with
        # its spawner, so the cascade follows the spawn tree.
        self._children: dict[str, Handle] = {}

        self._inbox: asyncio.Queue[InboxMessage] = asyncio.Queue()

        # Wake source for event-driven mode: set by every delivery.
        self._inbox_wake = asyncio.Event()

        # Pause/resume state: the flag gates the loop, the event
        # wakes the waiter, and _turn_task is the cancellable
        # in-flight turn.
        self._paused = False
        self._resumed = asyncio.Event()
        self._turn_task: asyncio.Task[Turn] | None = None

        # Delta events (subagent lifecycle for now) drained into the
        # next turn's <delta><subagent> section.
        self._delta_events: list[DeltaEvent] = []

        # Add-only rendered history: cleared wholesale once over
        # the char limit.
        self._messages: list[Message] = []

        self.last_turn: Turn | None = None
        self.cycles = 0

        self.report: str | None = None
        self.done: bool = False
        self._dying: bool = False  # death initiated (latch); reason is separate storage
        self._death_reason: str | None = None
        self._started: bool = False

        self._task: asyncio.Task[None] | None = None

    # ==================================================================
    # Identity / handles
    # ==================================================================

    def start(self) -> None:
        """Register and start the loop task."""
        self.book.register(self)

        self._task = asyncio.create_task(self.loop(), name=f"agent:{self.agent_id}")

    def handle_to_self(self, holder_id: str) -> Handle:
        from nan.agent.handle import Handle

        return Handle(holder_id=holder_id, target=self)

    def child_handle(self, agent_id: Any) -> Handle | None:
        if not isinstance(agent_id, str):
            return None

        return self._children.get(agent_id)

    def child_handles(self) -> list[Handle]:
        return list(self._children.values())

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
            # Not reachable through any legitimate channel today;
            # delivered rather than dropped, labelled and logged.
            kind = "unknown"

            logger.warning(
                "[{}] message from unverified source '{}'", self.agent_id, source
            )

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
        if self.config.max_depth >= 0 and self.depth + 1 > self.config.max_depth:
            return self, f"Max depth exceeded: {self.depth + 1} > {self.config.max_depth}"

        if self.config.max_children >= 0 and len(self._children) >= self.config.max_children:
            return self, f"Max children reached: {self.config.max_children}"

        child = Agent(
            book=self.book,
            llm=self.llm,
            injections=self.injections,
            workspace=self.workspace,
            persona_path=self.persona_path,  # one global persona, inherited
            sandbox=self.sandbox,
            parent=self,
            config=self.config,
        )

        self._children[child.agent_id] = child.handle_to_self(holder_id=self.agent_id)

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

    def request_death(self, reason: str) -> None:
        """
        Kill this agent now: flag + task cancellation. Idempotent
        (the latch is _dying; _death_reason is storage only and
        may stay None, e.g. finish exits via `done`). The loop's
        teardown does the rest (cascade, notice).
        """
        if not self._dying:
            self._dying = True

            self._death_reason = reason

            task = self._task

            if task is not None and not task.done():
                task.cancel()

            if task is not None and not self._started and not task.done():
                # The loop task was cancelled before its first
                # step: its try/finally never ran. Run teardown
                # here so cascade and upstream notice still happen.
                asyncio.get_running_loop().create_task(self._teardown(cancelled=False))

    def finish(self, report: str) -> None:
        """
        Record the final report (FinishVerb). The loop ends at
        the next check; teardown delivers it upstream. Latches
        _dying: a kill racing in after finish cannot flip the
        death notice's semantics.
        """
        self._dying = True

        self.report = report

        self.done = True

    def pause(self) -> None:
        """
        Pause the loop: the in-flight turn task is cancelled and
        discarded whole; its drained inputs ride back into the
        inbox, so resume opens a clean turn. Idempotent.
        """
        self._paused = True

        sink.emit("agent_paused", content={"agent_id": self.agent_id})

        task = self._turn_task

        if task is not None and not task.done():
            task.cancel()

    def resume(self) -> None:
        """Lift the pause; the loop opens a clean turn."""
        self._paused = False

        sink.emit("agent_resumed", content={"agent_id": self.agent_id})

        self._resumed.set()

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
                    "interval" if self.config.autonomous_interval > 0 else "event-driven"
                ),
            },
        )

        cancelled = False

        backoff_index = 0

        try:
            while not self.done and not self._dying:
                if self._paused:
                    await self._wait_while_paused()

                    if self._dying:
                        break

                # Event-driven mode (interval <= 0): no filler
                # turns -- block until the inbox has something.
                if self.config.autonomous_interval <= 0:
                    await self._wait_for_inbox()

                    if self._dying:
                        break

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
                        continue

                    raise

                except Exception as exc:
                    delay, backoff_index = next_backoff(self.config.backoff, backoff_index)

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

                backoff_index = 0
                self.cycles += 1

                if self.done or self._dying:
                    break

                if self.config.autonomous_interval > 0:
                    await asyncio.sleep(self.config.autonomous_interval)

                # Yield to the loop even at interval 0: a turn
                # with no real awaits (stubbed LLM, instant verbs)
                # would otherwise hog the event loop.
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            cancelled = True

            raise

        finally:
            await self._teardown(cancelled=cancelled)

    async def _wait_while_paused(self) -> None:
        self._resumed.clear()

        while self._paused:
            if self._resumed.is_set():
                self._resumed.clear()

                continue

            await self._resumed.wait()

    async def _wait_for_inbox(self) -> None:
        while self._inbox.empty():
            self._inbox_wake.clear()

            if self._inbox.empty():
                await self._inbox_wake.wait()

    async def run_turn(self, *, retries: int = 0) -> Turn:
        drained: list[InboxMessage] = []

        delta_events: list[DeltaEvent] = []

        try:
            return await self._run_turn_inner(drained, delta_events, retries=retries)

        except asyncio.CancelledError:
            # The turn dies whole (pause or agent death): nothing
            # rendered, nothing in history; drained inputs go
            # back for the next (clean) turn.
            sink.emit(
                "turn_cancelled",
                content={"agent_id": self.agent_id, "requeued_count": len(drained)},
            )

            for message in drained:
                self._deliver(message)

            self._delta_events = [*delta_events, *self._delta_events]

            raise

    async def _run_turn_inner(
        self,
        drained: list[InboxMessage],
        delta_events: list[DeltaEvent],
        *,
        retries: int,
    ) -> Turn:
        # -- observe ------------------------------------------------------
        drained.extend(self._drain_inbox())

        delta_events.extend(self._delta_events)

        self._delta_events = []

        skeleton = Turn(
            agent_id=self.agent_id,
            parent_id=(self.parent.agent_id if self.parent else None),
            depth=self.depth,
            inbox=tuple(drained),
            delta=tuple(delta_events),
        )

        sink.emit(
            "turn_started",
            content={
                "agent_id": self.agent_id,
                "turn_index": self.cycles + 1,
                "inbox_count": len(drained),
            },
        )

        contributions, service_events = await self.injections.collect(skeleton)

        sink.emit(
            "injections_collected",
            content={
                "agent_id": self.agent_id,
                "services": [
                    {
                        "service_id": i.service_id,
                        "ok": i.ok,
                        "duration_s": i.duration_s,
                        "text_len": len(i.text),
                    }
                    for i in contributions
                ],
            },
        )

        turn = replace(
            skeleton,
            injections=contributions,
            delta=tuple([*delta_events, *service_events]),
        )

        # -- think --------------------------------------------------------
        messages = [
            Message(role="system", content=self._persona_for_turn()),
            *self._messages,
            *render_turn(turn),
        ]

        try:
            generate_started = time.time()

            response = await self.engine.generate(
                agent_id=self.agent_id,
                depth=self.depth,
                request=LLMRequest(messages=messages, tools=tool_definitions(depth=self.depth)),
            )

            latency_s = time.time() - generate_started

        except Exception as exc:
            turn = _end_turn(turn, error=f"{type(exc).__name__}: {exc}")

            raise

        turn = _with_response(turn, response, latency_s=latency_s, retries=retries)

        # -- act ----------------------------------------------------------
        results: list[str] = []

        for call in turn.calls:
            logger.info("[{}] verb {} {}", self.agent_id, call.name, call.arguments)

            results.append(await self._run_call(call))

        turn = _with_results(turn, results)

        # -- freeze -------------------------------------------------------
        turn = freeze_turn(turn)

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
                "latency_s": (turn.llm.latency_s if turn.llm else 0.0),
                "retries": (turn.llm.retries if turn.llm else 0),
                "calls": len(turn.calls),
                "history_chars": sum(len(m.content or "") for m in self._messages),
            },
        )

        return turn

    # ==================================================================
    # Persona (re-read every turn)
    # ==================================================================

    def _persona_for_turn(self) -> str:
        """
        Re-read the persona file every turn. A read failure raises
        (the turn fails and backs off) -- persona is config-grade,
        we do not silently continue on a stale copy. The last known
        text is kept only for the changed-or-not log line.
        """
        try:
            text = self.persona_path.read_text(encoding="utf-8")

        except OSError:
            raise RuntimeError(
                f"[{self.agent_id}] persona file unreadable: {self.persona_path}"
            ) from None

        if text != self._persona_last:
            logger.info("[{}] persona changed; new system prompt from this turn", self.agent_id)

            self._persona_last = text

        return text

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

    async def _teardown(self, *, cancelled: bool) -> None:
        # An in-flight turn (cancelled with the loop, or still
        # running) is discarded, never written to history.
        turn_task = self._turn_task

        if turn_task is not None and not turn_task.done():
            turn_task.cancel()

        if turn_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn_task

        # Cascade first: every child I hold dies with me.
        children = self.child_handles()

        for handle in children:
            handle.kill()

        await asyncio.gather(
            *(
                t
                for h in children
                if (t := getattr(h._target, "_task", None)) is not None
            ),
            return_exceptions=True,
        )

        # Notify upstream: a finish report rides the inbox (it
        # carries content); a bare death rides the holder's system
        # event ledger, surfacing in their next <delta> section.
        if self.parent is not None:
            if self.report is not None:
                self.parent._deliver(
                    InboxMessage(
                        kind="subagent",
                        source=self.agent_id,
                        text=self.report,
                        event="finish",
                    )
                )

            else:
                reason = self._death_reason or ("cancelled" if cancelled else "exited")

                self.parent.note_delta("subagent", f"-{self.agent_id} died: {reason}")

        self.book.unregister(self.agent_id)

        reason = (
            "finished"
            if self.report is not None
            else (self._death_reason or ("cancelled" if cancelled else "exited"))
        )

        sink.emit(
            "agent_exited",
            content={"agent_id": self.agent_id, "reason": reason, "report": self.report},
        )

        logger.info(
            "[{}] stopped after {} cycles ({})",
            self.agent_id,
            self.cycles,
            self.report is not None and "finished" or (self._death_reason or "loop end"),
        )

    # ==================================================================
    # Small helpers
    # ==================================================================

    def _drain_inbox(self) -> list[InboxMessage]:
        items: list[InboxMessage] = []

        while True:
            try:
                items.append(self._inbox.get_nowait())

            except asyncio.QueueEmpty:
                return items

    def _clear_history_over_limit(self) -> None:
        chars = sum(len(m.content or "") for m in self._messages)

        if chars > self.config.history_char_limit:
            logger.info(
                "[{}] history {} chars over limit {}; clearing",
                self.agent_id,
                chars,
                self.config.history_char_limit,
            )

            self._messages.clear()


# ----------------------------------------------------------------------
# Module-level pure helpers (keep the Turn a value)
# ----------------------------------------------------------------------


def _self_cancel_pending() -> int:
    task = asyncio.current_task()

    return task.cancelling() if task is not None else 0


def _with_response(
    turn: Turn, response: LLMResponse, *, latency_s: float, retries: int
) -> Turn:
    return replace(
        turn,
        reply=response.content or "",
        calls=tuple(response.tool_calls),
        llm=LLMCall(
            model=response.model,
            usage=response.usage,
            finish_reason=response.finish_reason,
            latency_s=latency_s,
            retries=retries,
        ),
        ended_at=time.time(),
    )


def _with_results(turn: Turn, results: list[str]) -> Turn:
    return replace(turn, results=tuple(results))


def _end_turn(turn: Turn, error: str) -> Turn:
    return replace(turn, error=error, ended_at=time.time())
