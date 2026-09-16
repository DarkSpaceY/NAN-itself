from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from .engine import StepEngine
from .model import AgentResult
from .runtime import AgentRuntime
from ..events import EventBus, StreamSink
from ..utils.backoff import next_backoff
from ..utils.llm import Message


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


class CoreAgent:
    """
    Main Agent orchestration around the step engine.

    A turn is ONE observation -> model call -> result cycle:

        - skills are refreshed at every turn boundary
        - persona is re-read at every turn boundary
        - exactly one world snapshot is captured per turn
        - user messages arrive through the inbox Module and are
          part of the observation
        - a turn that ends with tool calls returns content=None;
          run_forever() immediately starts the next turn, whose
          observation is rebuilt fresh (modules, inbox included)
        - conversation history keeps every turn's messages; when
          its character count exceeds history_char_limit the
          whole history is cleared and the next turn starts fresh

    run_forever() owns the autonomous loop: failed turns retry
    with escalating backoff, stop requests get a grace window,
    and successful turns may be paced by autonomous_interval.
    """

    def __init__(
        self,
        *,
        llm,
        modules,
        tools,
        skills,
        persona_source: Callable[[], str],
        max_subagent_depth: int = 3,
        history_char_limit: int = 100_000,
        bus: EventBus | None = None,
        turn_grace: float = DEFAULT_TURN_GRACE,
        backoff: tuple[float, ...] = DEFAULT_BACKOFF,
        autonomous_interval: float = 0.0,
    ) -> None:
        self.llm = llm
        self.modules = modules
        self.tools = tools
        self.skills = skills
        self.persona_source = persona_source

        self.max_subagent_depth = max_subagent_depth

        self.history_char_limit = max(
            0,
            history_char_limit,
        )

        self.turn_grace = max(
            0.0,
            turn_grace,
        )

        self.backoff = backoff or (0.0,)

        self.autonomous_interval = max(
            0.0,
            autonomous_interval,
        )

        # Cross-turn conversation history (excludes the system
        # message, which is rebuilt from the persona every turn).
        self.history: list[Message] = []

        self.bus = bus

        self.agent_runtime = AgentRuntime(
            max_subagent_depth=max_subagent_depth,
        )

        self.engine = StepEngine(
            llm=llm,
            modules=modules,
            tools=tools,
            skills=skills,
            agent_runtime=self.agent_runtime,
        )

        # Loop lifecycle (run_forever).
        self.cycles = 0
        self._stopping = False
        self._stop_event = asyncio.Event()
        self._current_turn: asyncio.Task | None = None

    # ==================================================================
    # Main Agent
    # ==================================================================

    async def run(
        self,
    ) -> AgentResult:
        """
        Run one Main Agent turn: observe once, call once.
        """

        # --------------------------------------------------------------
        # Skills hot reload.
        # --------------------------------------------------------------

        self.skills.refresh()

        # --------------------------------------------------------------
        # Persona hot reload.
        # --------------------------------------------------------------

        persona = self.persona_source()

        # --------------------------------------------------------------
        # Exactly one fresh world snapshot per turn.
        # --------------------------------------------------------------

        world = self.modules.snapshot()

        # --------------------------------------------------------------
        # Create a fresh root context.
        # --------------------------------------------------------------

        root = self.agent_runtime.create_root(
            world=world,
        )

        # --------------------------------------------------------------
        # History: full retention, clear-all over the limit.
        # --------------------------------------------------------------

        if (
            self._history_chars()
            > self.history_char_limit
        ):
            logger.info(
                "History exceeded {} characters; "
                "clearing conversation history",
                self.history_char_limit,
            )

            self.history.clear()

        sink = (
            StreamSink(self.bus)
            if self.bus is not None
            else None
        )

        if sink is not None:
            sink.status_working()

        try:
            result = await self.engine.execute(
                context=root,
                persona=persona,
                history=self.history,
                report_sink=self._park_reports,
                sink=sink,
            )

        finally:
            if sink is not None:
                sink.status_idle()

        self.history.extend(
            result.messages
        )

        return result

    # ==================================================================
    # Autonomous loop
    # ==================================================================

    def request_stop(self) -> None:
        """
        Request graceful shutdown; safe from signal handlers.

        The current turn is NOT cancelled immediately: it gets a
        grace window (turn_grace) before cancellation.
        """
        if self._stopping:
            return

        logger.info("Agent stop requested")

        self._stopping = True
        self._stop_event.set()

    @property
    def stopping(self) -> bool:
        return self._stopping

    async def run_forever(self) -> None:
        """
        Run turns forever until request_stop().

        A turn that ends with tool calls is followed immediately
        by another turn: its observation is rebuilt fresh, so the
        model sees the tool results in history plus whatever the
        modules and the inbox now hold.
        """
        backoff_index = 0

        while not self._stopping:
            turn = asyncio.create_task(
                self.run(),
                name=f"agent-turn:{self.cycles}",
            )

            self._current_turn = turn

            stop_wait = asyncio.create_task(
                self._stop_event.wait(),
                name="agent-stop",
            )

            try:
                done, _ = await asyncio.wait(
                    {turn, stop_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # --------------------------------------------------
                # Turn completed normally.
                # --------------------------------------------------

                if turn in done:
                    stop_wait.cancel()
                    await self._cancel_and_suppress(
                        stop_wait
                    )

                    try:
                        result = turn.result()

                    except asyncio.CancelledError:
                        raise

                    except Exception:
                        delay, backoff_index = (
                            next_backoff(
                                self.backoff,
                                backoff_index,
                            )
                        )

                        logger.exception(
                            "Turn failed; backing off {}s",
                            delay,
                        )

                        self._current_turn = None

                        if await self._interruptible_wait(
                            delay,
                        ):
                            break

                        continue

                    backoff_index = 0
                    self.cycles += 1

                    logger.info(
                        "Cycle {} finished | reply: {}",
                        self.cycles,
                        (result.content or "")[:200],
                    )

                    self._current_turn = None

                    if self.autonomous_interval > 0:
                        if await self._interruptible_wait(
                            self.autonomous_interval,
                        ):
                            break

                    continue

                # --------------------------------------------------
                # Stop requested while the turn is executing.
                # --------------------------------------------------

                stop_wait.cancel()
                await self._cancel_and_suppress(
                    stop_wait
                )

                logger.info(
                    "Stopping; grace {}s for current turn",
                    self.turn_grace,
                )

                try:
                    await asyncio.wait_for(
                        asyncio.shield(turn),
                        timeout=self.turn_grace,
                    )

                    # The in-flight turn completed inside the
                    # grace window: it counts as a cycle.
                    self.cycles += 1

                except asyncio.TimeoutError:
                    logger.warning(
                        "Current turn exceeded shutdown grace "
                        "period; cancelling",
                    )

                except asyncio.CancelledError:
                    raise

                except Exception:
                    logger.exception(
                        "Current turn failed during shutdown grace",
                    )

                if not turn.done():
                    turn.cancel()

                await self._cancel_and_suppress(turn)

                self._current_turn = None

                return

            finally:
                if not stop_wait.done():
                    stop_wait.cancel()

                await self._cancel_and_suppress(
                    stop_wait
                )

        logger.info(
            "Agent stopped after {} cycles",
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
        Wait for `delay` seconds; returns True only when a stop
        was requested.
        """
        if self._stopping:
            return True

        if delay <= 0:
            return self._stopping

        try:
            await asyncio.sleep(delay)

        except asyncio.CancelledError:
            raise

        return self._stopping

    @staticmethod
    async def _cancel_and_suppress(
        task: asyncio.Task | None,
    ) -> None:
        """
        Cancel a task and consume its CancelledError / exception.
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
            pass

    # ==================================================================
    # Subagent reports
    # ==================================================================

    def _park_reports(
        self,
        reports: list[str],
    ) -> None:
        """
        Park finished subagent reports into the inbox Module.
        """
        inbox = self.modules.get(
            "inbox"
        )

        if inbox is None:
            logger.warning(
                "Inbox module not running; "
                "dropping {} subagent report(s)",
                len(reports),
            )

            return

        for report in reports:
            inbox.put(report)

    # ==================================================================
    # History
    # ==================================================================

    def _history_chars(
        self,
    ) -> int:
        return sum(
            len(message.content or "")
            for message in self.history
        )
