"""Shared fakes for the nan test suite (ported from
paper_search/nan_core/tests/smoke_agents.py)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nan.agent import Agent
from nan.agent.handle import Book
from nan.agent.sandbox import Sandbox
from nan.config import settings
from nan.injection import InjectionHub
from nan.llm import ThinkingBlock, ToolCall, Usage


@dataclass
class FakeStreamEvent:
    kind: str
    text: str | None = None
    thinking_text: str | None = None
    thinking: list[ThinkingBlock] | None = None
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    finish_reason: str | None = None


@dataclass
class FakeLLM:
    """
    Scripted LLM: a `script` maps a marker (matched against the
    whole prompt blob) to a queue of steps:

        ("text",)                        -> plain text reply
        ("tool", name, arguments)        -> one tool call

    Every request is recorded in `.requests`.
    """

    model = "fake"

    def __init__(self) -> None:
        self.scripts: dict[str, list[tuple[Any, ...]]] = {}
        self.requests: list[Any] = []
        self.thinking: str | None = None

    def script(self, marker: str, calls: list[tuple[Any, ...]]) -> None:
        self.scripts[marker] = list(calls)

    async def _respond(self, request: Any) -> tuple[str | None, list[ToolCall]]:
        blob = " ".join((m.content or "") for m in request.messages)

        for marker, queue in self.scripts.items():
            if marker in blob and queue:
                step = queue.pop(0)

                if step[0] == "text":
                    return "ok", []

                call = ToolCall(id=f"t{len(self.requests)}", name=step[1], arguments=step[2])

                return None, [call]

        return "ok", []

    async def generate(self, request: Any) -> AsyncIterator[FakeStreamEvent]:
        self.requests.append(request)

        content, calls = await self._respond(request)

        if self.thinking:
            yield FakeStreamEvent("thinking", thinking_text=self.thinking)

        if content:
            yield FakeStreamEvent("text", text=content)

        for c in calls:
            yield FakeStreamEvent("tool_call", tool_call=c)

        yield FakeStreamEvent(
            "done",
            usage=Usage(),
            finish_reason="stop",
            thinking=[ThinkingBlock(text=self.thinking)] if self.thinking else None,
        )


class SlowFakeLLM(FakeLLM):
    """
    FakeLLM whose FIRST model call blocks until released (later
    calls run normally). Test seam for pause-mid-turn: the hook
    lives in `_respond`, shared by both call paths.
    """

    def __init__(self) -> None:
        super().__init__()

        self.first_call_started = asyncio.Event()

        self.release_first = asyncio.Event()

        self._first_consumed = False

    async def _respond(self, request: Any) -> tuple[str | None, list[ToolCall]]:
        if not self._first_consumed and not self.release_first.is_set():
            self._first_consumed = True

            self.first_call_started.set()

            await self.release_first.wait()

        return await super()._respond(request)


def make_agent(
    *,
    tmp_path: Path,
    book: Book | None = None,
    llm: FakeLLM | None = None,
    persona: str = "ROOTPERSONA",
    agent: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Agent:
    workspace = tmp_path / "workspace"

    workspace.mkdir(parents=True, exist_ok=True)

    # The persona file is the only source. Tests that pass their
    # own persona path get it untouched; the rest get a tmp file
    # with the default text.
    if "persona_path" in kwargs:
        persona_path = kwargs.pop("persona_path")
    else:
        persona_path = tmp_path / "persona.md"

        persona_path.write_text(persona, encoding="utf-8")

    # Components read the live global settings; stamp this test's
    # layout and policy in (the autouse fixture restores them).
    settings.paths.workspace = workspace

    settings.paths.persona = persona_path

    for key, value in (agent or {}).items():
        setattr(settings.agent, key, value)

    # The sandbox is a required Agent dependency; tests that don't
    # care about isolation get the process-wide one.
    sandbox = kwargs.pop("sandbox", Sandbox())

    return Agent(
        book=book or Book(),
        llm=llm or FakeLLM(),  # type: ignore[arg-type]
        injections=InjectionHub(),
        sandbox=sandbox,
        **kwargs,
    )


async def await_until(predicate: Any, *, timeout: float = 3.0) -> None:
    """Poll a predicate on the event loop until true or timeout."""
    deadline = asyncio.get_running_loop().time() + timeout

    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition not met in time")

        await asyncio.sleep(0.02)


@pytest.fixture
def no_sink() -> Any:
    """Ensure no event bus is attached (default state)."""
    from nan.events import sink

    sink.reset()

    yield sink

    sink.reset()
