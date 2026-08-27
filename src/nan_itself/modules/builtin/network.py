# @builtin

"""
Network: the agent feeling its long-range nerves.

Facts only -- rates, connections, reachability, public identity.
No conclusions ("untrusted network" belongs to the agent).
"""

from __future__ import annotations

import asyncio

from loguru import logger

from src.nan_itself.modules.model import (
    Module,
    ModuleTurn,
)

from src.nan_itself.utils.network import (
    NetworkProbe,
    _fmt_bps,
)


class NetworkModule(Module):
    id = "network"

    poll_interval: float = 5.0

    events_render_limit: int = 3

    def __init__(self) -> None:
        self.probe = NetworkProbe()

        self._latest: dict | None = None

    async def start(self) -> None:
        logger.info("network module sampling every {}s", self.poll_interval)

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
                logger.exception("network sample failed")

            await asyncio.sleep(self.poll_interval)

    async def on_turn(self, record) -> None:
        pass

    # ------------------------------------------------------------------

    async def query(self, turn: ModuleTurn) -> str | None:
        facts = self._latest

        if facts is None:
            return None

        lines = ["[Network]"]

        lines.append(self._link_line(facts))

        public = facts.get("public_ip")

        tunnel = facts.get("tunnel")

        if public or tunnel:
            parts = []

            if public:
                parts.append(f"public: {public}")

            if tunnel:
                parts.append(f"tunnel: {tunnel}")

            lines.append("- " + " | ".join(parts))

        lines.append(self._traffic_line(facts))

        lines.append(self._conn_line(facts))

        events = facts.get("events") or []

        if events:
            rendered = " | ".join(
                text for _, text in events[-self.events_render_limit:]
            )

            lines.append(f"- events: {rendered}")

        return "\n".join(lines)

    def _link_line(self, facts: dict) -> str:
        if not facts.get("online"):
            span = facts.get("offline_since")

            return (
                f"- link: offline"
                f"{_since(span)}"
            )

        parts = []

        if facts.get("ssid"):
            parts.append(facts["ssid"])

        if facts.get("local_ip"):
            parts.append(facts["local_ip"])

        head = facts.get("primary_if") or "link"

        line = f"- link: {head} " + " ".join(parts)

        if facts.get("gw"):
            line += f" -> gw {facts['gw']}"

        latency = facts.get("latency")

        if latency is not None:
            line += f" | online {latency:.0f}ms"

        else:
            line += " | online"

        return line

    def _traffic_line(self, facts: dict) -> str:
        dl = facts.get("dl")

        ul = facts.get("ul")

        if dl is None and ul is None:
            return "- traffic: sampling"

        return (
            f"- traffic: ↓{_fmt_bps(dl or 0)} "
            f"↑{_fmt_bps(ul or 0)}"
        )

    def _conn_line(self, facts: dict) -> str:
        est = facts.get("est")

        public = facts.get("public_conns")

        listen = facts.get("listen")

        if est is None:
            return "- conn: sampling"

        return (
            f"- conn: {est} est ({public or 0} public)"
            f" | listen {listen or 0}"
        )


def _since(ts: float | None) -> str:
    import time

    if ts is None:
        return ""

    return f" {int(time.time() - ts)}s"
