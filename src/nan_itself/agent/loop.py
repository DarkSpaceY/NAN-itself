from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from loguru import logger


DEFAULT_TURN_GRACE = 5.0

DEFAULT_BACKOFF = (
    1.0,
    2.0,
    4.0,
    8.0,
    15.0,
    30.0,
    60.0,
)


class Inbox:
    """
    Bounded drop-oldest mailbox feeding the agent loop.

    Everything addressed to the agent goes here:

        - user messages
        - module notes
        - external events
        - anything else that should become agent input

    The world channel (DataSpace observations) does NOT pass
    through the Inbox. CoreAgent captures a fresh world snapshot
    at the beginning of every turn.

    Important semantics:

        Inbox is a wakeup / interrupt source.

        It does NOT determine whether the agent is allowed to run.

    A message arriving while the agent is executing a turn is kept
    in the Inbox and consumed at the next turn boundary.
    """

    def __init__(
        self,
        maxsize: int = 256,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be greater than zero")

        self._items: deque[str] = deque()
        self._maxsize = maxsize

        # Set whenever at least one item is available.
        #
        # This event is intentionally public through wake_event().
        # It allows external waits to be interrupted by new input.
        self._event = asyncio.Event()

    def put(
        self,
        item: str,
    ) -> None:
        """
        Add one item.

        When full, the oldest item is dropped.
        """
        if not item:
            return

        while len(self._items) >= self._maxsize:
            dropped = self._items.popleft()

            logger.warning(
                "Inbox overflow, dropped oldest item: {!r}",
                dropped[:80],
            )

        self._items.append(item)

        # Wake every waiter.
        self._event.set()

    def drain(self) -> list[str]:
        """
        Remove and return all currently queued items.

        This is the primary operation used at an agent turn boundary.
        """
        if not self._items:
            self._event.clear()
            return []

        items = list(self._items)

        self._items.clear()
        self._event.clear()

        return items

    def empty(self) -> bool:
        return not self._items

    def wake_event(self) -> asyncio.Event:
        """
        Return the event used to wake interruptible waits.

        The event is:

            set   -> at least one Inbox item exists
            clear -> Inbox is empty
        """
        return self._event

    async def wait_not_empty(self) -> None:
        """
        Wait until at least one item is present.

        This method is retained as a convenience for callers that
        explicitly want Inbox-only waiting.

        AgentLoop itself does NOT use this as its primary lifecycle
        mechanism, because the AgentLoop must remain autonomous.
        """
        while not self._items:
            await self._event.wait()

    async def get(self) -> str:
        """
        Wait for and return one item.
        """
        while True:
            if self._items:
                item = self._items.popleft()

                if not self._items:
                    self._event.clear()

                return item

            await self._event.wait()


class AgentLoop:
    """
    Permanent loop around a CoreAgent.

    The AgentLoop is a continuously running process.

    The important lifecycle rule is:

        Inbox != permission to run.

    An empty Inbox is a valid state. A turn may therefore receive
    an empty string and still execute autonomously.

    The Inbox is instead an interrupt / wakeup source:

        1. If the agent is waiting between turns, Inbox wakes it.
        2. If the agent is currently executing a turn, Inbox does
           NOT cancel that turn. The message is consumed by the
           following turn.
        3. Stop requests are different: they interrupt waits and
           give the current turn a grace window before cancellation.

    Failed turns are retried with escalating backoff.

    A successful turn resets the backoff and immediately continues
    the autonomous loop.

    NOTE:

    This implementation intentionally does NOT introduce an
    autonomous timer. If CoreAgent.run("") returns immediately,
    the loop will immediately execute another empty turn.

    If autonomous turns need pacing, use autonomous_interval below.
    """

    def __init__(
        self,
        agent: Any,
        inbox: Inbox,
        *,
        turn_grace: float = DEFAULT_TURN_GRACE,
        backoff: tuple[float, ...] = DEFAULT_BACKOFF,
        autonomous_interval: float = 0.0,
    ) -> None:
        self.agent = agent
        self.inbox = inbox

        self.turn_grace = max(0.0, turn_grace)

        self.backoff = backoff or (0.0,)

        # Optional pacing between successful autonomous turns.
        #
        # 0.0 means:
        #     immediately start the next turn.
        #
        # >0 means:
        #     wait this amount before the next turn, but wake
        #     immediately when Inbox receives input.
        self.autonomous_interval = max(
            0.0,
            autonomous_interval,
        )

        self.cycles = 0

        self._stopping = False
        self._stop_event = asyncio.Event()

        self._current_turn: asyncio.Task | None = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def request_stop(self) -> None:
        """
        Request graceful shutdown.

        This method is intentionally synchronous so the composition
        root can call it from signal handlers / shutdown code without
        awaiting anything.

        The current turn is NOT cancelled immediately.
        """
        if self._stopping:
            return

        logger.info("Agent loop stop requested")

        self._stopping = True
        self._stop_event.set()

    @property
    def stopping(self) -> bool:
        return self._stopping

    async def run_forever(self) -> None:
        """
        Run the Agent permanently until request_stop() is called.

        The lifecycle is:

            collect input
                ↓
            execute turn
                ↓
            success → next turn
                ↓
            failure → backoff → retry
                ↓
            repeat forever

        Empty input is valid and means autonomous execution.
        """
        backoff_index = 0

        # Input belonging to a failed turn must survive the retry.
        pending_retry_input = ""

        while not self._stopping:
            # ----------------------------------------------------------
            # Build input for this turn.
            #
            # This NEVER blocks waiting for Inbox.
            # ----------------------------------------------------------

            if pending_retry_input:
                inputs = self._merge_inputs(
                    pending_retry_input,
                    self._collect_inputs(),
                )
            else:
                inputs = self._collect_inputs()

            pending_retry_input = ""

            # ----------------------------------------------------------
            # Execute one turn.
            # ----------------------------------------------------------

            turn = asyncio.create_task(
                self.agent.run(inputs),
                name=f"agent-turn:{self.cycles}",
            )

            self._current_turn = turn

            stop_wait = asyncio.create_task(
                self._stop_event.wait(),
                name="agent-loop-stop",
            )

            try:
                done, _ = await asyncio.wait(
                    {turn, stop_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # ======================================================
                # CASE 1:
                # Turn completed normally.
                # ======================================================

                if turn in done:
                    stop_wait.cancel()
                    await self._cancel_and_suppress(stop_wait)

                    try:
                        result = turn.result()

                    except asyncio.CancelledError:
                        raise

                    except Exception:
                        delay = self.backoff[backoff_index]

                        backoff_index = min(
                            backoff_index + 1,
                            len(self.backoff) - 1,
                        )

                        logger.exception(
                            "Turn failed; backing off {}s",
                            delay,
                        )

                        # Preserve the exact input that failed.
                        pending_retry_input = inputs

                        self._current_turn = None

                        if await self._interruptible_wait(
                            delay,
                        ):
                            break

                        continue

                    # --------------------------------------------------
                    # Successful turn.
                    # --------------------------------------------------

                    backoff_index = 0
                    self.cycles += 1

                    reply = getattr(
                        result,
                        "content",
                        "",
                    )

                    logger.info(
                        "Cycle {} finished | reply: {}",
                        self.cycles,
                        (reply or "")[:200],
                    )

                    self._current_turn = None

                    # --------------------------------------------------
                    # IMPORTANT:
                    #
                    # We do NOT wait for Inbox here.
                    #
                    # The next loop iteration will execute another
                    # turn, even if inputs == "".
                    # --------------------------------------------------

                    if self.autonomous_interval > 0:
                        if await self._interruptible_wait(
                            self.autonomous_interval,
                        ):
                            break

                    continue

                # ======================================================
                # CASE 2:
                # Stop requested while turn is executing.
                # ======================================================

                stop_wait.cancel()
                await self._cancel_and_suppress(stop_wait)

                logger.info(
                    "Stopping; grace {}s for current turn",
                    self.turn_grace,
                )

                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(turn),
                        timeout=self.turn_grace,
                    )

                except asyncio.TimeoutError:
                    logger.warning(
                        "Current turn exceeded shutdown grace "
                        "period; cancelling",
                    )

                except asyncio.CancelledError:
                    # Preserve task cancellation semantics.
                    raise

                except Exception:
                    logger.exception(
                        "Current turn failed during shutdown grace",
                    )

                else:
                    # The turn completed successfully during the
                    # grace window, so count it exactly once.
                    self.cycles += 1

                    reply = getattr(
                        result,
                        "content",
                        "",
                    )

                    logger.info(
                        "Cycle {} finished during grace | reply: {}",
                        self.cycles,
                        (reply or "")[:200],
                    )

                # ------------------------------------------------------
                # The turn either:
                #
                #   - already completed
                #   - timed out
                #   - raised
                #
                # Calling cancel() on an already completed task is safe.
                # ------------------------------------------------------

                if not turn.done():
                    turn.cancel()

                await self._cancel_and_suppress(turn)

                self._current_turn = None

                return

            finally:
                # If the turn completed normally, this is already None.
                #
                # If the loop is externally cancelled while waiting,
                # make sure we don't leave an orphaned stop waiter.
                if not stop_wait.done():
                    stop_wait.cancel()

                await self._cancel_and_suppress(stop_wait)

        logger.info(
            "Agent loop stopped after {} cycles",
            self.cycles,
        )

    # ==================================================================
    # Waiting / interruption
    # ==================================================================

    async def _interruptible_wait(
        self,
        delay: float,
    ) -> bool:
        """
        Wait for `delay` seconds.

        The wait is interrupted by either:

            - stop request
            - new Inbox input

        Returns:

            True
                stop was requested

            False
                timer expired OR Inbox woke the loop
        """
        if self._stopping:
            return True

        if delay <= 0:
            return self._stopping

        sleeper = asyncio.create_task(
            asyncio.sleep(delay),
            name="agent-loop-timer",
        )

        stopper = asyncio.create_task(
            self._stop_event.wait(),
            name="agent-loop-stop",
        )

        inbox_wait = asyncio.create_task(
            self.inbox.wake_event().wait(),
            name="agent-loop-inbox",
        )

        try:
            done, _ = await asyncio.wait(
                {
                    sleeper,
                    stopper,
                    inbox_wait,
                },
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stopper in done or self._stopping:
                return True

            # Inbox is an interrupt source.
            #
            # We intentionally return immediately. The next turn
            # will drain the Inbox and receive the new input.
            return False

        finally:
            for task in (
                sleeper,
                stopper,
                inbox_wait,
            ):
                if not task.done():
                    task.cancel()

            for task in (
                sleeper,
                stopper,
                inbox_wait,
            ):
                await self._cancel_and_suppress(task)

    # ==================================================================
    # Input assembly
    # ==================================================================

    def _collect_inputs(self) -> str:
        """
        Collect everything currently available for the next turn.

        This method NEVER waits.

        Empty input is valid.
        """
        parts: list[str] = []

        # --------------------------------------------------------------
        # External addressed messages.
        # --------------------------------------------------------------

        batch = self.inbox.drain()

        if batch:
            parts.append(
                "\n\n".join(batch)
            )

        # --------------------------------------------------------------
        # Late subagent reports.
        #
        # These are not converted into Inbox messages because they
        # have their own delivery semantics inside CoreAgent.
        #
        # has_pending_reports() is intentionally checked here only
        # so an empty-input autonomous turn is still triggered when
        # a late report exists.
        # --------------------------------------------------------------

        if self.agent.has_pending_reports():
            # The actual reports are drained by CoreAgent.run().
            #
            # We do not need to manufacture input here.
            pass

        return "\n\n".join(parts)

    @staticmethod
    def _merge_inputs(
        original: str,
        extra: str,
    ) -> str:
        """
        Merge retry input with messages that arrived during backoff.
        """
        if original and extra:
            return f"{original}\n\n{extra}"

        return original or extra

    @staticmethod
    async def _cancel_and_suppress(
        task: asyncio.Task | None,
    ) -> None:
        """
        Cancel a task and consume its CancelledError / exception.

        Used only for helper tasks whose result is intentionally
        discarded.
        """
        if task is None:
            return

        if not task.done():
            task.cancel()

        try:
            await task

        except asyncio.CancelledError:
            pass

        except Exception:
            # Helper tasks are best-effort cleanup tasks.
            pass


class _StopRequested(Exception):
    """
    Retained for compatibility with older callers/tests.

    AgentLoop no longer needs this exception internally because
    lifecycle waiting is handled by events rather than Inbox-only
    blocking.
    """

    pass
