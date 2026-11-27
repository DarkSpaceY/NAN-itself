"""
Plan service: the agent's goal tree as prompt text.

An injection service (see docs/for-service-authors): registers with
the hub and answers every turn request with the whole tree for a new
epoch, or only what changed since the previous delivery inside it.

The CLI owns the tree (services/plan/plan_cli.py): this process never
writes it, and re-reads it on every turn, so a plan edited while the
service runs shows up on the next turn.

Run:  uv run main.py   (from this directory, after `uv sync`)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plan as tree
import uvicorn
from fastapi import FastAPI
from loguru import logger
from nan_service_kit import HubLink, ServiceConfig, health_route, read_config, setup_logging

HERE = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = HERE / "config.yaml"

# Shown once per epoch, above the tree: the tree itself is state, this
# line is the only instruction about how to keep it and when to open one.
HINT = (
    "- goal tree, edited with `plan add|done|drop`; each goal states an outcome, not a step."
    " Open one before multi-step or long-running work, and keep it current as results land."
)


@dataclass
class _State:
    """What one agent has already been shown in this epoch."""

    epoch: int
    seen: dict[str, str]
    turn_started_at: float
    last_text: str


class PlanService:
    def __init__(
        self,
        config: ServiceConfig,
        *,
        link: HubLink | None = None,
    ) -> None:
        self.config = config

        self.link = link or HubLink(config)

        self.path = config.home / "data" / "plan.json"

        self._states: dict[str, _State] = {}

    def contribution(self, turn: dict[str, Any] | None = None) -> str:
        turn = turn or {}

        agent_id = str(turn.get("agent_id") or "root")

        started = float(turn.get("started_at") or 0.0)

        epoch = int(turn.get("epoch") or 0)

        origin = str(turn.get("origin") or "normal")

        goals = tree.load(self.path)

        if not goals:
            self._states.pop(agent_id, None)

            return ""

        state = self._states.get(agent_id)

        # The hub re-injects a parked turn with the same started_at;
        # replay the exact text and do not advance the baseline.
        if state is not None and state.turn_started_at == started:
            return state.last_text

        full = state is None or state.epoch != epoch or origin in ("first", "context_cleared")

        text = (
            HINT + "\n" + tree.render_full(goals) if full else tree.render_delta(state.seen, goals)
        )

        self._states[agent_id] = _State(
            epoch=epoch,
            seen=tree.snapshot(goals),
            turn_started_at=started,
            last_text=text,
        )

        return text


def build_app(service: PlanService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with service.link:
            logger.info(
                "plan service up on {}:{} (plan {})",
                service.config.host,
                service.config.port,
                service.path,
            )

            yield

    app = FastAPI(lifespan=lifespan)

    health_route(app, service.config)

    @app.post("/turn")
    async def turn(body: dict[str, Any]) -> dict[str, str]:
        return {"text": service.contribution(body)}

    return app


def main() -> None:
    config = read_config(ServiceConfig, DEFAULT_CONFIG_PATH)

    setup_logging(config)

    uvicorn.run(
        build_app(PlanService(config)),
        host=config.host,
        port=config.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
