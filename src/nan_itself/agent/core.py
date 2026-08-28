"""
CoreAgent: how the main agent plugs into the world.

The step engine knows nothing about a "main agent"; this module
is the thin orchestration around it:

    - Skills hot-reload at every turn boundary
    - the L1 persona is re-read every turn
    - one world snapshot per turn
    - late subagent reports are seeded into the next turn

Cross-turn conversation memory is intentionally absent: durable
state lives in Module observations and (later) the memory module,
so every main turn starts from a clean slate. A turn has no step
cap; it ends when the model produces text, raises, or stops.
"""

from __future__ import annotations

from typing import Callable

from .engine import (
    StepEngine,
)
from .model import (
    AgentResult,
)
from .reports import (
    LateReportBuffer,
)
from .runtime import (
    AgentRuntime,
)
from ..events import (
    EventBus,
    StreamSink,
)


class CoreAgent:
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

        # Reports from children that finished after their parent's
        # turn already ended. Delivered at the start of the next run().
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
        Late subagent reports are seeded into this turn.
        """
        # Skills hot-reload at the turn boundary: anything dropped
        # into the workspace becomes visible right when the next
        # consumer (catalog injection / activate_skill) runs.
        self.skills.refresh()

        # L1 persona is re-read every turn so persona edits
        # hot-reload without a restart.
        persona = self.persona_source()

        world = self.modules.snapshot()

        root = self.agent_runtime.create_root(
            world=world,
            skill=None,
            task=user_input,
        )

        seed_reports = self._pending_reports.drain()

        sink = StreamSink(self.bus) if self.bus is not None else None

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
    # Pending late reports (consumed by AgentLoop / tests)
    # ==================================================================

    def has_pending_reports(self) -> bool:
        return self._pending_reports.has_pending()

    def drain_pending_reports(self) -> list[str]:
        return self._pending_reports.drain()
