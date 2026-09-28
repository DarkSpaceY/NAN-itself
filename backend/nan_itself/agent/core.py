"""
The Agent: one class, one homogeneous loop, root or subagent.

A turn is ONE observation -> model call -> result cycle:

    - skills are refreshed at every turn boundary
    - persona is re-read from its file at every turn boundary
    - exactly one world snapshot is captured per agent per turn,
      never shared across agents
    - a subagent's task, the ambient Module context and finished
      children's reports ride on the turn as STRUCTURED inputs,
      harvested/collected at the turn boundary the same way for
      root and subagents alike (no inbox parking); rendering
      them into model messages is the engine's business
    - a turn that ends with tool calls runs the verbs and
      writes their results back positionally; the loop
      immediately starts the next turn, whose observation
      inputs are collected fresh
    - there is no persistent history: each Turn carries the
      history snapshot as of its start, and the next turn
      derives its snapshot from the last Turn's history plus
      its rendered messages (clear-all over the agent's
      history_char_limit)

One agent is ONE asyncio Task. The root is started by app.py, a
subagent by its parent's spawn(); either way the task runs loop()
and the Agent instance is the handle.

loop() owns the lifecycle for every agent: failed turns retry
with escalating backoff, a stop request is cooperative (it is
read at the next turn boundary), and successful turns may be
paced by autonomous_interval. Mid-turn abort is always the
whole task's cancellation, never an in-flight turn's. Exit
cleanup is a single path: descendants are stopped/cancelled and
a subagent's final report is recorded as a structured Report.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import replace
from typing import Any

from loguru import logger

from .model import (
    Report,
    SubagentLimitError,
)
from .verbs import (
    FINISH_TOOL_NAME,
    INVOKE_CHANNELS_TOOL_NAME,
    INVOKE_SKILL_TOOL_NAME,
    INVOKE_TOOL_TOOL_NAME,
    LIST_CHANNELS_TOOL_NAME,
    LIST_SKILLS_TOOL_NAME,
    LIST_TOOLS_TOOL_NAME,
    SHOW_CHANNELS_TOOL_NAME,
    SHOW_SKILL_TOOL_NAME,
    SHOW_TOOL_TOOL_NAME,
    SLEEP_TOOL_NAME,
    SPAWN_TOOL_NAME,
    VERBS,
)
from ..events import sink
from ..modules.model import Turn
from ..utils import paths
from ..utils.backoff import next_backoff


DEFAULT_BACKOFF = (
    1.0,
    2.0,
    4.0,
    8.0,
    15.0,
    30.0,
    60.0,
)


class Agent:
    def __init__(
        self,
        *,
        engine,
        modules,
        tools,
        skills,
        task=None,
        parent=None,
        max_subagent_depth: int = 3,
        history_char_limit: int = 100_000,
        backoff: tuple[float, ...] = DEFAULT_BACKOFF,
        autonomous_interval: float = 0.0,
    ) -> None:
        # Capabilities are shared references, not copies: a
        # spawned child inherits exactly these objects.
        self.engine = engine
        self.modules = modules
        self.tools = tools
        self.skills = skills

        # A subagent's instruction. The root has none: user input
        # reaches it through the inbox Module.
        self.task = task

        self.parent = parent

        self.max_subagent_depth = max_subagent_depth

        # Shared history retention policy (root and subagents
        # alike): full retention, clear-all over the character
        # limit. Applied at derivation time.
        self.history_char_limit = max(
            0,
            history_char_limit,
        )

        self.backoff = backoff or (0.0,)

        self.autonomous_interval = max(
            0.0,
            autonomous_interval,
        )

        # Stable within the process for the agent's whole life
        # (the root no longer rotates per turn).
        self.agent_hash = uuid.uuid4().hex

        self.parent_hash = (
            parent.agent_hash
            if parent is not None
            else None
        )

        self.depth = (
            0
            if parent is None
            else parent.depth + 1
        )

        # Children still owing a report; harvested at the next
        # turn boundary.
        self.children: list[Agent] = []

        # Set on exit; after that the report is final.
        self.done: bool = False

        # The structured report (subagents only).
        self.report: Report | None = None

        # This agent's last completed Turn; the next turn
        # derives its history snapshot from it.
        self.last_turn: Turn | None = None

        self.cycles = 0

        # Loop lifecycle. `_task` is the loop task of a spawned
        # subagent (None for a root driven by app.py).
        self._stopping = False
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def stopping(self) -> bool:
        return self._stopping

    # ==================================================================
    # Turn boundary
    # ==================================================================

    async def run(self) -> Turn:
        """
        Run one turn: observe once, call once, act once.

        The completed Turn is returned and delivered (to the
        Modules) exactly once, whether the turn succeeded, failed
        or was cancelled.
        """
        # ----------------------------------------------------------
        # Skills hot reload.
        # ----------------------------------------------------------

        self.skills.refresh()

        # ----------------------------------------------------------
        # Persona hot reload.
        # ----------------------------------------------------------

        persona = paths.persona_path().read_text(encoding="utf-8")

        # ----------------------------------------------------------
        # Exactly one fresh world snapshot per agent per turn,
        # never shared.
        # ----------------------------------------------------------

        world = self.modules.snapshot()

        # ----------------------------------------------------------
        # Fold finished children's reports into this turn's
        # observation (root and subagents alike).
        # ----------------------------------------------------------

        pending_reports = self._harvest_children()

        # ----------------------------------------------------------
        # History snapshot: the last turn's history plus its
        # rendered messages (rendering is the engine's business);
        # clear-all over the limit. No persistent history exists
        # anywhere.
        # ----------------------------------------------------------

        prior: tuple = ()

        if self.last_turn is not None:
            prior = (
                self.last_turn.history
                + self.engine.render_turn(
                    self.last_turn
                )
            )

            if (
                self._history_chars(prior)
                > self.history_char_limit
            ):
                logger.info(
                    "History exceeded {} characters; "
                    "clearing conversation history",
                    self.history_char_limit,
                )

                prior = ()

        # ----------------------------------------------------------
        # Skeleton turn: identity, world, persona, the derived
        # history snapshot, and the already-harvested child
        # reports. A subagent's task is shown to the model (the
        # root has none: user input reaches it through the inbox
        # Module). The flow is still empty, so the Modules'
        # query() sees the in-flight turn.
        # ----------------------------------------------------------

        turn = Turn(
            agent_hash=self.agent_hash,
            parent_hash=self.parent_hash,
            depth=self.depth,
            task=(
                self.task
                if self.depth > 0
                else None
            ),
            world=world,
            persona=persona,
            history=prior,
            reports=tuple(pending_reports),
            started_at=time.time(),
        )

        turn = await self._observe(turn)

        completed = turn

        try:
            completed = await self.engine.step(
                turn
            )

            completed = await self._run_calls(
                completed
            )

            self.last_turn = completed

            return completed

        except asyncio.CancelledError:
            completed = replace(
                completed,
                error="cancelled",
                ended_at=time.time(),
            )
            raise

        except Exception as exc:
            completed = replace(
                completed,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
                ended_at=time.time(),
            )
            raise

        finally:
            self.modules.deliver_turn(
                completed
            )

    async def _observe(
        self,
        turn: Turn,
    ) -> Turn:
        """
        Query the ambient Module context for this turn.

        Every running Module is queried against the in-flight
        turn (identity/world already fixed), each query mirrored
        as a module record on the main stream. The turn's other
        observation inputs (task, harvested child reports) are
        already on the Turn; rendering any of it into model
        messages is the engine's business (prompts).
        """
        pending_query_records: dict[str, str] = {}

        def on_module_start(
            module_id: str,
        ) -> None:
            pending_query_records[module_id] = (
                sink.emit(
                    "record_started",
                    content={
                        "category": "module_query",
                        "payload": {},
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )
            )

        def on_module_result(
            module_id: str,
            result: str | None,
            duration: float,
            failed: bool,
        ) -> None:
            record_id = (
                pending_query_records.pop(
                    module_id,
                    None,
                )
            )

            if not record_id:
                return

            if failed:
                sink.emit(
                    "record_failed",
                    id=record_id,
                    content={
                        "error": {
                            "type": "module_query_failed",
                            "message": "Module query failed.",
                        },
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )
                return

            if not result:
                sink.emit(
                    "record_void",
                    id=record_id,
                    content={
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )
                return

            entries = _result_entries(result)

            if entries:
                sink.emit(
                    "record_detail",
                    id=record_id,
                    content={
                        "entries": entries,
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

            sink.emit(
                "record_done",
                id=record_id,
                content={
                    "duration_s": duration,
                    "agent_hash": self.agent_hash,
                    "parent_hash": self.parent_hash,
                    "depth": self.depth,
                },
            )

        ambient_context = (
            await self.modules.query_snapshot(
                turn,
                on_start=on_module_start,
                on_result=on_module_result,
            )
        )

        return replace(
            turn,
            ambient=tuple(ambient_context),
        )

    def _harvest_children(
        self,
    ) -> list[Report]:
        """
        Collect the structured reports of finished children.

        Every harvested report is mirrored as an agent record on
        the main stream, straight from its fields. Children still
        running stay listed.
        """
        reports: list[Report] = []

        remaining: list[Agent] = []

        for child in list(self.children):
            if not child.done:
                remaining.append(
                    child,
                )

                continue

            if child.report is None:
                continue

            reports.append(
                child.report,
            )

            report = child.report

            record_id = (
                sink.emit(
                    "record_started",
                    content={
                        "category": "subagent_report",
                        "payload": {
                            "agent_id": report.agent_id,
                            "task": report.task or "",
                            "status": report.status,
                            "body": report.body,
                        },
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )
            )

            if record_id:
                entries: list[dict[str, Any]] = [
                    {
                        "kind": "field",
                        "label": "id",
                        "value": report.agent_id,
                    },
                    {
                        "kind": "field",
                        "label": "task",
                        "value": report.task or "",
                    },
                    {
                        "kind": "field",
                        "label": "status",
                        "value": report.status,
                    },
                ]

                entries.extend(
                    {
                        "kind": "text",
                        "text": line,
                    }
                    for line in (
                        report.body or ""
                    ).splitlines()
                )

                sink.emit(
                    "record_detail",
                    id=record_id,
                    content={
                        "entries": entries,
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

                sink.emit(
                    "record_done",
                    id=record_id,
                    content={
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

        self.children = remaining

        return reports

    # ==================================================================
    # Call routing
    # ==================================================================

    async def _run_calls(
        self,
        turn: Turn,
    ) -> Turn:
        """
        Run the turn's tool calls, writing each result back
        positionally onto the turn. A turn with no calls is
        returned unchanged; a call that raises leaves the
        results unwritten (the turn carries the error).
        """
        if not turn.calls:
            return turn

        results: list[str] = []

        for call in turn.calls:
            logger.info(
                "[turn:{}] tool {} {}",
                turn.agent_hash[:8],
                call.name,
                json.dumps(
                    call.arguments
                ),
            )

            results.append(
                await self._run_call(call)
            )

        return replace(
            turn,
            results=tuple(results),
        )

    async def _run_call(
        self,
        call,
    ) -> str:
        # The spawn verb owns its own record (the subagent_spawn
        # payload carries the child id/depth, which only the verb
        # knows after it spawns); every other verb is mirrored
        # here from the call itself.
        record_id = ""

        if call.name != SPAWN_TOOL_NAME:
            category, payload = _record_spec(call)

            record_id = sink.emit(
                "record_started",
                content={
                    "category": category,
                    "payload": payload,
                    "agent_hash": self.agent_hash,
                    "parent_hash": self.parent_hash,
                    "depth": self.depth,
                },
            )

        started = time.time()

        try:
            verb = VERBS.get(
                call.name
            )

            if verb is None:
                result_text = (
                    f"Unknown action '{call.name}'. "
                    f"Available actions: "
                    f"{', '.join(VERBS)}."
                )

            else:
                result_text = (
                    await verb.execute(
                        call=call,
                        agent=self,
                    )
                )

        except Exception as exc:
            if record_id:
                sink.emit(
                    "record_failed",
                    id=record_id,
                    content={
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

            raise

        if record_id:
            entries = _result_entries(
                result_text
            )

            if entries:
                sink.emit(
                    "record_detail",
                    id=record_id,
                    content={
                        "entries": entries,
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

            sink.emit(
                "record_done",
                id=record_id,
                content={
                    "duration_s": (
                        time.time() - started
                    ),
                    "agent_hash": self.agent_hash,
                    "parent_hash": self.parent_hash,
                    "depth": self.depth,
                },
            )

        return result_text

    # ==================================================================
    # Spawning
    # ==================================================================

    def spawn(
        self,
        task: str,
    ) -> "Agent":
        """
        Spawn a child Agent running the same homogeneous loop.

        The child shares this agent's engine / modules / tools /
        skills references and starts immediately (its own
        asyncio Task); its report is harvested at a later turn
        boundary. The instance itself is the handle.
        """
        child_depth = self.depth + 1

        if (
            child_depth
            > self.max_subagent_depth
        ):
            raise SubagentLimitError(
                "Maximum Subagent depth exceeded: "
                f"{child_depth} > "
                f"{self.max_subagent_depth}"
            )

        child = Agent(
            engine=self.engine,
            modules=self.modules,
            tools=self.tools,
            skills=self.skills,
            task=task,
            parent=self,
            max_subagent_depth=(
                self.max_subagent_depth
            ),
            history_char_limit=(
                self.history_char_limit
            ),
            backoff=self.backoff,
            autonomous_interval=(
                self.autonomous_interval
            ),
        )

        self.children.append(
            child,
        )

        child._task = asyncio.create_task(
            child.loop(),
            name=(
                f"subagent:"
                f"{child.agent_hash}"
            ),
        )

        return child

    # ==================================================================
    # Stop / teardown
    # ==================================================================

    def finish(
        self,
        report: str,
    ) -> None:
        """
        Submit the final report (FinishVerb).

        The structured Report is produced HERE, before `done` is
        flipped: a parent that harvests on `child.done` therefore
        always finds a final report -- there is no window between
        "finished" and "report ready". Rendering it into prompt
        text (<subagent_report>) is the engine's business.
        """
        self.report = Report(
            agent_id=self.agent_hash[:8],
            task=self.task,
            status="completed",
            body=report,
        )

        self.done = True

    def request_stop(self) -> None:
        """
        Request shutdown; safe from signal handlers.

        Idempotent and cooperative: it only sets flags, which the
        loop reads at the next turn boundary. Any mid-turn abort
        is the task's own cancellation (cascade teardown or
        external shutdown), never a per-turn grace window.
        """
        if self._stopping:
            return

        logger.info("Agent stop requested")

        self._stopping = True
        self._stop_event.set()

        for child in list(self.children):
            child.request_stop()

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
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=delay,
            )

        except asyncio.TimeoutError:
            return self._stopping

        return True

    async def _teardown(
        self,
        cancelled: bool,
    ) -> None:
        """
        Exit cleanup: the single path.

        Descendants are stopped and cancelled first. A subagent
        whose turn already finished has its report in place
        (finish() formatted it) and it is never overwritten; one
        cancelled before finishing reports its cancellation. In
        both cases the report precedes `done`, so a parent that
        harvests on `child.done` always finds the report ready.
        """
        lost = [
            child
            for child in self.children
            if child.done
        ]

        if lost:
            logger.warning(
                "Agent {} exiting with {} "
                "unharvested child report(s); dropped",
                self.agent_hash[:8],
                len(lost),
            )

        for child in list(self.children):
            child.request_stop()

        await self._shutdown_descendants()

        self.children.clear()

        # Only a subagent cancelled before it could finish owes a
        # report here, and never over an already-final one.
        if self.report is None and cancelled:
            self.report = Report(
                agent_id=self.agent_hash[:8],
                task=self.task,
                status="failed",
                body="error: cancelled",
            )

        self.done = True

        sink.emit(
            "status",
            content={
                "state": "idle",
                "agent_hash": self.agent_hash,
                "parent_hash": self.parent_hash,
                "depth": self.depth,
            },
        )

    async def _shutdown_descendants(self) -> None:
        """
        Cancel and await every descendant task, immediately.
        """
        tasks = self._descendant_tasks()

        for task in tasks:
            if not task.done():
                task.cancel()

        for task in tasks:
            await self._cancel_and_suppress(
                task,
            )

    def _descendant_tasks(
        self,
    ) -> list[asyncio.Task]:
        tasks: list[asyncio.Task] = []

        for child in self.children:
            if child._task is not None:
                tasks.append(
                    child._task,
                )

            tasks.extend(
                child._descendant_tasks()
            )

        return tasks

    # ==================================================================
    # The loop (root and subagent alike)
    # ==================================================================

    async def loop(self) -> None:
        """
        Run turns until stop or finish.

        1 agent = 1 Task: `await self.run()` inline, no per-turn
        task and no wait()/event race.
        """
        # create_task copies the parent context; every event this
        # agent emits carries its own identity explicitly.
        sink.emit(
            "status",
            content={
                "state": "working",
                "agent_hash": self.agent_hash,
                "parent_hash": self.parent_hash,
                "depth": self.depth,
            },
        )

        backoff_index = 0
        cancelled = False

        try:
            while (
                not self.done
                and not self._stopping
            ):
                try:
                    turn = await self.run()

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
                    self._reply_text(turn)[:200],
                )

                # Checked before any pacing wait: a stop or a
                # finish needs no extra delay.
                if self.done or self._stopping:
                    break

                if self.autonomous_interval > 0:
                    if await self._interruptible_wait(
                        self.autonomous_interval,
                    ):
                        break

            logger.info(
                "Agent {} stopped after {} cycles",
                self.agent_hash[:8],
                self.cycles,
            )

        except asyncio.CancelledError:
            cancelled = True
            raise

        finally:
            # Exit cleanup: the single path.
            await self._teardown(cancelled)

    # ==================================================================
    # Small helpers
    # ==================================================================

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
            logger.exception(
                "Task crashed while being reaped"
            )

    @staticmethod
    def _history_chars(
        messages,
    ) -> int:
        return sum(
            len(message.content or "")
            for message in messages
        )

    @staticmethod
    def _reply_text(
        turn: Turn,
    ) -> str:
        """
        The turn's reply text (empty when the turn never
        produced one -- in flight, failed or cancelled).
        """
        return turn.reply or ""


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


# Record category per verb: tools, skills, channels, sleeping and
# finishing each map to their own semantic category; anything else
# falls back to `unknown` carrying the original verb name. The spawn
# verb builds its own `subagent_spawn` record in agent/verbs.py.
_RECORD_CATEGORY = {
    INVOKE_TOOL_TOOL_NAME: "tool_call",
    LIST_TOOLS_TOOL_NAME: "tool_call",
    SHOW_TOOL_TOOL_NAME: "tool_call",
    INVOKE_SKILL_TOOL_NAME: "skill_invoke",
    LIST_SKILLS_TOOL_NAME: "skill_invoke",
    SHOW_SKILL_TOOL_NAME: "skill_invoke",
    INVOKE_CHANNELS_TOOL_NAME: "channel_write",
    LIST_CHANNELS_TOOL_NAME: "channel_write",
    SHOW_CHANNELS_TOOL_NAME: "channel_write",
    SLEEP_TOOL_NAME: "sleep",
    FINISH_TOOL_NAME: "finish",
}


def _as_str(
    value: Any,
) -> str:
    """
    The value as a string, or "" when it is not one.
    """
    return value if isinstance(value, str) else ""


def _split_tool_name(
    name: Any,
) -> tuple[str, str]:
    """
    Split a 'provider/tool' composite; a bare name is the tool.
    """
    if not isinstance(name, str):
        return "", ""

    provider, separator, tool = name.partition("/")

    if not separator:
        return "", provider

    return provider, tool


def _record_spec(
    call,
) -> tuple[str, dict[str, Any]]:
    """
    The typed (category, payload) pair for one verb call, built from
    the call's raw arguments. The payload mirrors the verb's own
    parameter shape; the fallback is `unknown` with the verb name.
    """
    category = _RECORD_CATEGORY.get(
        call.name,
    )

    arguments = (
        call.arguments
        if isinstance(call.arguments, dict)
        else {}
    )

    if category == "tool_call":
        provider, tool = _split_tool_name(
            arguments.get("name")
        )

        inner = arguments.get("arguments")

        return category, {
            "provider": provider,
            "tool": tool,
            "arguments": (
                inner if isinstance(inner, dict) else {}
            ),
        }

    if category == "skill_invoke":
        payload: dict[str, Any] = {
            "skill": _as_str(
                arguments.get("name")
            ),
        }

        resource = _as_str(
            arguments.get("path")
        )

        if resource:
            payload["resource"] = resource

        return category, payload

    if category == "channel_write":
        channel_payload: dict[str, Any] = {
            "module": _as_str(
                arguments.get("module")
            ),
            "channel": _as_str(
                arguments.get("channel")
            ),
        }

        if "payload" in arguments:
            channel_payload["payload"] = (
                arguments["payload"]
            )

        return category, channel_payload

    if category == "sleep":
        seconds = arguments.get("seconds")

        if not isinstance(
            seconds,
            (int, float),
        ) or isinstance(seconds, bool):
            seconds = 0.0

        return category, {"seconds": seconds}

    if category == "finish":
        return category, {}

    return "unknown", {"verb": call.name}


def _result_entries(
    text: str,
) -> list[dict[str, Any]]:
    """
    The typed detail entries for a verb/module result. A JSON body
    travels as one `code` block (the typed form of the old
    pretty-printed dump); any other text stays plain `text` lines.
    """
    if not text:
        return []

    stripped = text.strip()

    if stripped[:1] in ("{", "["):
        try:
            json.loads(stripped)

        except Exception:
            pass

        else:
            return [
                {
                    "kind": "code",
                    "text": text,
                }
            ]

    return [
        {
            "kind": "text",
            "text": line,
        }
        for line in text.splitlines()
    ]
