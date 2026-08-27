# @builtin

"""
System: the agent feeling its own body.

Facts only -- every line in query() is a directly verifiable
measurement or a deterministic event; no heuristics, no
conclusions. The agent judges, this module measures.

All heavy lifting lives in utils/system.py; this module is the
async shell: park, sample on a cadence, publish, render.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from loguru import logger

from src.nan_itself.modules.model import (
    Module,
    ModuleTurn,
)

from src.nan_itself.utils.system import (
    HAS_PSUTIL,
    SystemProbe,
)


WEEKDAY_ZH = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


def _fmt_span(seconds: float | None) -> str | None:
    if seconds is None:
        return None

    if seconds < 90:
        return f"{int(seconds)}s"

    minutes = seconds / 60

    if minutes < 90:
        return f"{int(minutes)}m"

    return f"{hours:.1f}h" if (hours := minutes / 60) else None


class SystemModule(Module):
    id = "system"

    poll_interval: float = 5.0

    render_preview_cap: int = 80

    events_render_limit: int = 3

    def __init__(self) -> None:
        self.probe = SystemProbe()

        self._latest: dict | None = None

    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not HAS_PSUTIL:
            logger.warning("psutil missing; system module degraded")

        logger.info("system module sampling every {}s", self.poll_interval)

        while True:
            try:
                facts = await asyncio.to_thread(self.probe.sample)

                self._latest = facts

                self.data.publish(
                    {
                        key: value
                        for key, value in facts.items()
                        if key != "events"
                    }
                    | {"events": facts.get("events", [])[-3:]}
                )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception("system sample failed")

            await asyncio.sleep(self.poll_interval)

    async def on_turn(self, record) -> None:
        pass

    # ------------------------------------------------------------------

    async def query(self, turn: ModuleTurn) -> str | None:
        facts = self._latest

        if facts is None:
            return None

        lines = ["[System]"]

        local = datetime.now().astimezone()

        offset = local.strftime("%z")

        lines.append(
            f"- now: {local.strftime('%H:%M')} "
            f"{WEEKDAY_ZH[local.weekday()]} "
            f"(UTC{offset[:3]})"
        )

        focus = facts.get("focus")

        if focus:
            app, title = focus

            title_part = (
                f" — {title[: self.render_preview_cap]}"
                if title
                else ""
            )

            lines.append(f"- focus: {app}{title_part}")

        idle = _fmt_span(facts.get("idle"))

        work = _fmt_span(facts.get("work_seconds"))

        awake = _fmt_span(facts.get("uptime"))

        lines.append(
            f"- idle: {idle or 'unknown'}"
            f" | work: {work or '0s'}"
            f" | awake: {awake or 'unknown'}"
        )

        lines.append(f"- {self._resource_line(facts)}")

        events = facts.get("events") or []

        if events:
            recent = events[-self.events_render_limit:]

            rendered = " | ".join(
                f"{text}"
                for _, text in recent
            )

            lines.append(f"- events: {rendered}")

        self_cpu = facts.get("self_cpu")

        self_mem = facts.get("self_mem_mb")

        if self_cpu is not None and self_mem is not None:
            lines.append(
                f"- self: nan cpu {self_cpu:.1f}%"
                f" mem {self_mem}MB"
            )

        return "\n".join(lines)

    def _resource_line(self, facts: dict) -> str:
        cpu = facts.get("cpu")

        mem = facts.get("mem")

        disk = facts.get("disk")

        parts = [
            f"cpu {cpu:.0f}%" if cpu is not None else None,
            f"mem {mem:.0f}%" if mem is not None else None,
            f"disk {disk:.0f}%" if disk is not None else None,
        ]

        line = " | ".join(p for p in parts if p)

        battery = facts.get("battery")

        if battery is not None:
            state = (
                "charging"
                if facts.get("charging")
                else "on battery"
            )

            line += f" | battery {battery}% {state}"

        return line
