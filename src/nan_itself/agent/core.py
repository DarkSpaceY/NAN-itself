from __future__ import annotations

from typing import Callable

from .engine import StepEngine
from .model import AgentResult
from .reports import LateReportBuffer
from .runtime import AgentRuntime
from ..events import EventBus, StreamSink


class CoreAgent:
    """
    Main Agent orchestration around the step engine.

    Per-turn guarantees:

        - skills are refreshed at every turn boundary
        - persona is re-read at every turn boundary
        - exactly one world snapshot is captured per turn
        - late subagent reports are injected into the current turn
        - cross-turn conversation history is intentionally absent

    An empty user_input is valid.

    Empty input means that the AgentLoop is requesting an
    autonomous turn rather than a user-addressed turn.
    """

    def __init__(
        self,
        *,
        llm,
        modules,
        providers,
        skills,
        persona_source: Callable[[], str],
        max_subagent_depth: int = 3,
        bus: EventBus | None = None,
    ) -> None:
        self.llm = llm
        self.modules = modules
        self.providers = providers
        self.skills = skills
        self.persona_source = persona_source

        self.max_subagent_depth = max_subagent_depth

        self.bus = bus

        self.agent_runtime = AgentRuntime(
            max_subagent_depth=max_subagent_depth,
        )

        self.engine = StepEngine(
            llm=llm,
            modules=modules,
            providers=providers,
            skills=skills,
            agent_runtime=self.agent_runtime,
        )

        # Reports from child agents that finish after the parent turn
        # has already ended.
        #
        # They are delivered at the beginning of the next run().
        self._pending_reports = LateReportBuffer()

    # ==================================================================
    # Main Agent
    # ==================================================================

    async def run(
        self,
        user_input: str,
    ) -> AgentResult:
        """
        Run one Main Agent turn.

        A new world snapshot is captured exactly once.

        user_input may be empty. This is how AgentLoop requests an
        autonomous turn.
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
            skill=None,
            task=user_input,
        )

        # --------------------------------------------------------------
        # Deliver late subagent reports.
        # --------------------------------------------------------------

        seed_reports = self._pending_reports.drain()

        sink = (
            StreamSink(self.bus)
            if self.bus is not None
            else None
        )

        if sink is not None:
            sink.status_working()

        try:
            return await self.engine.execute(
                context=root,
                user_input=user_input,
                persona=persona,
                seed_reports=seed_reports,
                report_sink=self._pending_reports.park,
                sink=sink,
            )

        finally:
            if sink is not None:
                sink.status_idle()

    # ==================================================================
    # Pending late reports
    # ==================================================================

    def has_pending_reports(self) -> bool:
        return self._pending_reports.has_pending()

    def drain_pending_reports(self) -> list[str]:
        return self._pending_reports.drain()