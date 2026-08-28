from __future__ import annotations

import asyncio
from collections import deque

from loguru import logger


DEFAULT_TURN_GRACE = 5.0

DEFAULT_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0)


class Inbox:
    """
    Bounded drop-oldest mailbox feeding the agent loop.

    Everything addressed to the agent goes here: user lines,
    module notes, anything else. The world channel (DataSpace
    observations) does NOT pass through; it is re-injected fresh
    on every turn by the CoreAgent itself.
    """

    def __init__(
        self,
        maxsize: int = 256,
    ) -> None:
        self._items: deque[str] = deque()
        self._maxsize = maxsize
        self._event = asyncio.Event()

    def put(
        self,
        item: str,
    ) -> None:
        if not item:
            return

        while len(self._items) >= self._maxsize:
            dropped = self._items.popleft()

            logger.warning(
                "Inbox overflow, dropped oldest item: {!r}",
                dropped[:80],
            )

        self._items.append(item)

        self._event.set()

    def drain(self) -> list[str]:
        items = list(self._items)

        self._items.clear()

        if not self._items:
            self._event.clear()

        return items

    def empty(self) -> bool:
        return not self._items

    def wake_event(self) -> asyncio.Event:
        """
        Public handle for interruptible waits: set on every put,
        cleared when the inbox drains empty. Sleep-like waits can
        race against this so new input interrupts them.
        """
        return self._event

    async def wait_not_empty(self) -> None:
        """
        Wait until at least one item is present, without
        consuming anything.
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
    The permanent loop around a CoreAgent.

    The agent is a persistent process: the loop never ends on its
    own. It blocks only when there is genuinely nothing to do;
    late subagent reports nudge it awake with an empty input so
    their seeded delivery happens promptly.

    Containment contract:
        - a failing turn never kills the loop (catch-all + backoff)
        - stop requests let the current turn finish within a grace
          window before it is cancelled
        - facades are stopped by the composition root afterwards
    """

    def __init__(
        self,
        agent: Any,
        inbox: Inbox,
        *,
        turn_grace: float = DEFAULT_TURN_GRACE,
        backoff: tuple[float, ...] = DEFAULT_BACKOFF,
    ) -> None:
        self.agent = agent
        self.inbox = inbox

        self.turn_grace = turn_grace
        self.backoff = backoff or (0.0,)

        self.cycles = 0

        self._stopping = False
        self._stop_event = asyncio.Event()
        self._current_turn: asyncio.Task | None = None

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def request_stop(self) -> None:
        self._stopping = True

        self._stop_event.set()

    @property
    def stopping(self) -> bool:
        return self._stopping

    async def run_forever(self) -> None:
        """
        Run cycles until requested to stop.

        A failed turn retries its own input batch with
        escalating backoff; success resets the escalation.
        """
        backoff_index = 0

        while not self._stopping:
            try:
                inputs = await self._next_inputs()

            except _StopRequested:
                break

            # Retry the same batch until success or shutdown.
            while True:
                turn = asyncio.create_task(
                    self.agent.run(inputs),
                    name=f"agent-turn:{self.cycles}",
                )

                self._current_turn = turn

                stop_wait = asyncio.create_task(
                    self._stop_event.wait(),
                    name="agent-loop-stop",
                )

                done, _ = await asyncio.wait(
                    {turn, stop_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if turn in done:
                    stop_wait.cancel()

                    try:
                        await turn

                    except asyncio.CancelledError:
                        raise

                    except Exception:
                        delay = self.backoff[
                            backoff_index
                        ]

                        backoff_index = min(
                            backoff_index + 1,
                            len(self.backoff) - 1,
                        )

                        logger.exception(
                            "Turn failed; backing "
                            "off {}s",
                            delay,
                        )

                        self._current_turn = None

                        if await self._interruptible_sleep(
                            delay,
                        ):
                            break

                        # Retry the same batch.
                        continue

                    backoff_index = 0

                    self.cycles += 1

                    # Duck-typed access: the loop accepts any
                    # agent whose run() returns a reply-like
                    # object, not only CoreAgent.AgentResult.
                    reply = getattr(
                        turn.result(),
                        "content",
                        "",
                    )

                    logger.info(
                        "Cycle {} finished | reply: {}",
                        self.cycles,
                        (reply or "")[:200],
                    )

                    self._current_turn = None

                    break

                # Stop requested mid-turn: give it a grace
                # window, then cancel.
                stop_wait.cancel()

                logger.info(
                    "Stopping; grace {}s for current turn",
                    self.turn_grace,
                )

                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(turn),
                        timeout=self.turn_grace,
                    )

                    # The turn finished within the grace
                    # window: log it exactly like a normal
                    # cycle instead of dropping the result.
                    self.cycles += 1

                    reply = getattr(
                        result,
                        "content",
                        "",
                    )

                    logger.info(
                        "Cycle {} finished during "
                        "grace | reply: {}",
                        self.cycles,
                        (reply or "")[:200],
                    )

                except (
                    asyncio.TimeoutError,
                    asyncio.CancelledError,
                    Exception,
                ):
                    pass

                turn.cancel()

                try:
                    await turn

                except (
                    asyncio.CancelledError,
                    Exception,
                ):
                    pass

                self._current_turn = None

                return

        logger.info(
            "Agent loop stopped after {} cycles",
            self.cycles,
        )

    async def _interruptible_sleep(
        self,
        delay: float,
    ) -> bool:
        """
        Sleep for the backoff delay.

        Returns True when a stop request arrived meanwhile.
        """
        sleeper = asyncio.create_task(
            asyncio.sleep(delay)
        )

        stopper = asyncio.create_task(
            self._stop_event.wait()
        )

        done, _ = await asyncio.wait(
            {sleeper, stopper},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in (sleeper, stopper):
            if task not in done:
                task.cancel()

                try:
                    await task
                except (
                    asyncio.CancelledError,
                    Exception,
                ):
                    pass

        return self._stopping

    # ==================================================================
    # Input assembly
    # ==================================================================

    async def _next_inputs(self) -> str:
        """
        Block until there is something worth a turn.

        Addressed messages win; a pending late report alone
        triggers an empty-input cycle whose seeded reports do
        the talking.
        """
        while True:
            batch = self.inbox.drain()

            if batch:
                return "\n\n".join(batch)

            if self.agent.has_pending_reports():
                return ""

            getter = asyncio.create_task(
                self.inbox.wait_not_empty()
            )

            stopper = asyncio.create_task(
                self._stop_event.wait()
            )

            done, _ = await asyncio.wait(
                {getter, stopper},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stopper in done:
                getter.cancel()

                try:
                    await getter
                except (
                    asyncio.CancelledError,
                    Exception,
                ):
                    pass

                raise _StopRequested()

            stopper.cancel()

            try:
                await stopper
            except asyncio.CancelledError:
                pass


class _StopRequested(Exception):
    pass
