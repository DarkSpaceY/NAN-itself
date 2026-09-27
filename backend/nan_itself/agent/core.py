"""
The Agent: one class, one homogeneous loop, root or subagent.

A turn is ONE observation -> model call -> result cycle:

    - skills are refreshed at every turn boundary
    - persona is re-read from its file at every turn boundary
    - exactly one world snapshot is captured per agent per turn,
      never shared across agents
    - a subagent's task rides in its observation; finished
      children's reports are harvested at the next turn boundary
      and folded into the observation the same way for root and
      subagents alike (no inbox parking)
    - a turn that ends with tool calls runs the verbs, appends
      their results as tool messages and returns; the loop
      immediately starts the next turn, whose observation is
      rebuilt fresh
    - there is no persistent history: each Turn carries the
      history snapshot as of its start, and the next turn
      derives its snapshot from the last Turn (clear-all over
      the agent's history_char_limit)

One agent is ONE asyncio Task. The root is started by app.py, a
subagent by its parent's spawn(); either way the task runs loop()
and the Agent instance is the handle.

loop() owns the lifecycle for every agent: failed turns retry
with escalating backoff, a stop request is cooperative (it is
read at the next turn boundary), and successful turns may be
paced by autonomous_interval. Mid-turn abort is always the
whole task's cancellation, never an in-flight turn's. Exit
cleanup is a single path: descendants are stopped/cancelled and
a subagent's final report is formatted as <subagent_report>.
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
    SubagentLimitError,
)
from .prompts import (
    build_observation,
)
from .reports import (
    format_child_report,
    report_record_name,
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
from ..utils.llm import Message


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

        # The formatted <subagent_report> (subagents only).
        self.report: str | None = None

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
        # History snapshot: derived from the last turn; clear-all
        # over the limit. No persistent history exists anywhere.
        # ----------------------------------------------------------

        prior: tuple = ()

        if self.last_turn is not None:
            prior = (
                self.last_turn.history
                + self.last_turn.messages
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
        # Skeleton turn: identity, world, persona and the derived
        # history snapshot. Its messages are still empty, so the
        # Modules' query() sees the in-flight turn with nothing
        # rendered yet.
        # ----------------------------------------------------------

        turn = Turn(
            agent_hash=self.agent_hash,
            parent_hash=self.parent_hash,
            depth=self.depth,
            task=self.task,
            world=world,
            persona=persona,
            history=prior,
            started_at=time.time(),
        )

        turn = await self._observe(
            turn,
            pending_reports,
        )

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
        reports: list[str],
    ) -> Turn:
        """
        Assemble this turn's observation message.

        One ambient context per turn: every running Module is
        queried against the in-flight turn (identity/world already
        fixed), each query mirrored as a module record on the main
        stream. This agent's task and its harvested child reports
        are folded in the same way for root and subagents alike.
        """
        pending_query_records: dict[str, str] = {}

        def on_module_start(
            module_id: str,
        ) -> None:
            pending_query_records[module_id] = (
                sink.emit(
                    "record_started",
                    content={
                        "kind": "module",
                        "name": module_id,
                        "summary": "",
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
                        "summary": "query failed",
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

            for line in result.splitlines():
                sink.emit(
                    "record_detail",
                    id=record_id,
                    content={
                        "line": line,
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

            sink.emit(
                "record_done",
                id=record_id,
                content={
                    "summary": "",
                    "note": f"{duration:.1f}s",
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
            messages=(
                build_observation(
                    ambient_context=ambient_context,
                    task=(
                        self.task
                        if self.depth > 0
                        else None
                    ),
                    reports=reports,
                ),
            ),
        )

    def _harvest_children(
        self,
    ) -> list[str]:
        """
        Collect the formatted reports of finished children.

        Every harvested report is mirrored as an agent record on
        the main stream. Children still running stay listed.
        """
        reports: list[str] = []

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

            record_id = (
                sink.emit(
                    "record_started",
                    content={
                        "kind": "agent",
                        "name": report_record_name(
                            child.report,
                        ),
                        "summary": "",
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )
            )

            if record_id:
                for line in child.report.splitlines():
                    sink.emit(
                        "record_detail",
                        id=record_id,
                        content={
                            "line": line,
                            "agent_hash": self.agent_hash,
                            "parent_hash": self.parent_hash,
                            "depth": self.depth,
                        },
                    )

                sink.emit(
                    "record_done",
                    id=record_id,
                    content={
                        "summary": "report",
                        "note": "",
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
        Run the tool calls of the turn's last message, appending
        each result as a tool message. A turn whose last message
        carries no tool calls is returned unchanged.
        """
        messages = list(turn.messages)

        if not messages:
            return turn

        calls = getattr(
            messages[-1],
            "tool_calls",
            None,
        )

        if not calls:
            return turn

        for call in calls:
            logger.info(
                "[turn:{}] tool {} {}",
                turn.agent_hash[:8],
                call.name,
                json.dumps(
                    call.arguments
                ),
            )

            result_text = (
                await self._run_call(call)
            )

            messages.append(
                _tool_message(
                    call.id,
                    result_text,
                )
            )

        return replace(
            turn,
            messages=tuple(messages),
        )

    async def _run_call(
        self,
        call,
    ) -> str:
        record_id = sink.emit(
            "record_started",
            content={
                "kind": _RECORD_KINDS.get(
                    call.name,
                    "verb",
                ),
                "name": call.name,
                "summary": "",
                "agent_hash": self.agent_hash,
                "parent_hash": self.parent_hash,
                "depth": self.depth,
            },
        )

        if record_id:
            for line in _pretty_args(
                call.arguments
            ):
                sink.emit(
                    "record_detail",
                    id=record_id,
                    content={
                        "line": line,
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
                        "summary": (
                            f"{type(exc).__name__}"
                        ),
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

            raise

        if record_id:
            for line in _result_lines(
                result_text
            ):
                sink.emit(
                    "record_detail",
                    id=record_id,
                    content={
                        "line": line,
                        "agent_hash": self.agent_hash,
                        "parent_hash": self.parent_hash,
                        "depth": self.depth,
                    },
                )

            sink.emit(
                "record_done",
                id=record_id,
                content={
                    "summary": _compact_result(
                        result_text
                    ),
                    "note": (
                        f"{time.time() - started:.1f}s"
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

        The formatted <subagent_report> is produced HERE, before
        `done` is flipped: a parent that harvests on `child.done`
        therefore always finds a final report -- there is no
        window between "finished" and "report ready".
        """
        self.report = format_child_report(
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
            self.report = format_child_report(
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
            pass

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
        The turn's reply: the text of its last assistant message.
        The turn may end on a tool result, so scan backwards.
        """
        for message in reversed(turn.messages):
            if (
                getattr(
                    message,
                    "role",
                    None,
                )
                == "assistant"
            ):
                return message.content or ""

        return ""


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


# UI record kind per verb: tools, skills, channels (targets),
# spawning, sleeping and finishing each get their own kind;
# anything else stays a generic verb.
_RECORD_KINDS = {
    INVOKE_TOOL_TOOL_NAME: "tool",
    LIST_TOOLS_TOOL_NAME: "tool",
    SHOW_TOOL_TOOL_NAME: "tool",
    INVOKE_SKILL_TOOL_NAME: "skill",
    LIST_SKILLS_TOOL_NAME: "skill",
    SHOW_SKILL_TOOL_NAME: "skill",
    INVOKE_CHANNELS_TOOL_NAME: "target",
    LIST_CHANNELS_TOOL_NAME: "target",
    SHOW_CHANNELS_TOOL_NAME: "target",
    SPAWN_TOOL_NAME: "spawn",
    SLEEP_TOOL_NAME: "sleep",
    FINISH_TOOL_NAME: "finish",
}


def _tool_message(
    tool_call_id: str,
    content: str,
) -> Message:
    return Message(
        role="tool",
        tool_call_id=tool_call_id,
        content=content,
    )


def _pretty_args(
    arguments: Any,
) -> list[str]:
    if not arguments:
        return []

    try:
        return json.dumps(
            arguments,
            ensure_ascii=False,
            indent=2,
        ).splitlines()

    except Exception:
        return [
            str(arguments)[:200]
        ]


def _result_lines(
    text: str,
) -> list[str]:
    return (
        (text or "").splitlines()
    )


def _compact_result(
    text: str,
) -> str:
    flat = (
        (text or "")
        .replace("\n", " ")
        .strip()
    )

    # MCP CallToolResult JSON: surface the human text,
    # not the envelope.
    if flat.startswith("{"):
        try:
            payload = json.loads(flat)

            if isinstance(
                payload,
                dict,
            ):
                parts = [
                    str(
                        block.get(
                            "text",
                            "",
                        )
                    )
                    for block in (
                        payload.get(
                            "content",
                            [],
                        )
                    )
                    if (
                        isinstance(
                            block,
                            dict,
                        )
                        and block.get(
                            "type"
                        )
                        == "text"
                    )
                ]

                if parts:
                    flat = (
                        " ".join(
                            parts
                        ).strip()
                    )

        except Exception:
            pass

    return flat
