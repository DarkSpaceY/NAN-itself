"""
Structural contracts of the refactored agent package.

Locks the public surface, the RolePolicy visibility matrix and
the small protocol details that earlier lived untested inside
CoreAgent.
"""

import asyncio

import pytest

from src.nan_itself import agent as agent_pkg
from src.nan_itself.utils.llm import (
    ToolCall,
)
from src.nan_itself.agent import (
    VERBS,
    CoreAgent,
    Inbox,
    LateReportBuffer,
    RolePolicy,
)
from src.nan_itself.tools import (
    ProviderRuntime,
)
from src.nan_itself.agent.verbs import (
    ACTIVATE_SKILL_TOOL_NAME,
)


# ============================================================================
# Public surface
# ============================================================================


def test_package_exports_are_complete():
    required = {
        "DEFAULT_BACKOFF",
        "DEFAULT_TURN_GRACE",
        "AgentContext",
        "AgentLoop",
        "AgentResult",
        "AgentRuntime",
        "AgentTurn",
        "ChildSubagent",
        "CoreAgent",
        "Inbox",
        "LateReportBuffer",
        "RolePolicy",
        "StepEngine",
        "SubagentHandle",
        "SubagentLimitError",
        "VERBS",
    }

    assert required <= set(agent_pkg.__all__)


# ============================================================================
# RolePolicy visibility matrix
# ============================================================================


class _DepthStub:
    def __init__(self, depth: int) -> None:
        self.depth = depth


@pytest.mark.parametrize(
    "verb_name,main_visible,sub_visible",
    [
        ("sleep", True, True),
        ("dispatch_subagent", True, True),
        (ACTIVATE_SKILL_TOOL_NAME, False, True),
    ],
)
def test_role_visibility_matrix(
    verb_name,
    main_visible,
    sub_visible,
):
    verb = VERBS[verb_name]

    assert (
        verb.visible(0, RolePolicy)
        is main_visible
    )
    assert (
        verb.visible(1, RolePolicy)
        is sub_visible
    )


def test_hidden_verb_reply_names_the_verb():
    reply = RolePolicy.hidden_verb_reply(
        ACTIVATE_SKILL_TOOL_NAME
    )

    assert ACTIVATE_SKILL_TOOL_NAME in reply


def test_orphan_archive_policy():
    # Main-tree orphans are parked; deep-subtree orphans dropped.
    assert RolePolicy.archives_orphan_reports(0)
    assert not RolePolicy.archives_orphan_reports(1)
    assert not RolePolicy.archives_orphan_reports(2)


# ============================================================================
# Empty-reply corrective protocol
# ============================================================================


class EmptyReplyLLM:
    def __init__(self) -> None:
        self.requests = []

    async def generate_complete(self, request):
        self.requests.append(request)

        class R:
            content = ""
            tool_calls = []
            finish_reason = "stop"

        await asyncio.sleep(0)

        return R()


class _FakeModules:
    def __init__(self):
        self.turn_records: list = []

    def snapshot(self):
        return {}

    def deliver_turn(self, record):
        self.turn_records.append(record)

    async def query_snapshot(self, turn, world):
        return []


class _FakeSkills:
    def refresh(self):
        pass

    def catalog(self):
        return ()

    def names(self):
        return ()

    def activate(self, name):
        raise KeyError(name)


@pytest.mark.asyncio
async def test_empty_reply_ends_turn_immediately():
    """
    The old Ollama-specific corrective injection is gone: an
    empty reply simply ends the turn (logged, not corrected).
    """
    llm = EmptyReplyLLM()

    core_agent = CoreAgent(
        llm=llm,
        modules=_FakeModules(),
        providers=ProviderRuntime(),
        skills=_FakeSkills(),
        persona_source=lambda: "CORE",
    )

    result = await core_agent.run("hello")

    assert len(llm.requests) == 1
    assert result.content == ""


# ============================================================================
# Inbox overflow policy
# ============================================================================


def test_inbox_overflow_drops_oldest():
    inbox = Inbox(maxsize=2)

    inbox.put("a")
    inbox.put("b")
    inbox.put("c")

    assert inbox.drain() == ["b", "c"]



# ============================================================================
# Late report buffer unit
# ============================================================================


def test_late_report_buffer_round_trip():
    buffer = LateReportBuffer()

    assert not buffer.has_pending()

    buffer.park(["r1", "r2"])

    assert buffer.has_pending()

    assert buffer.drain() == ["r1", "r2"]
    assert not buffer.has_pending()
    assert buffer.drain() == []


# ============================================================================
# Depth limit surfaces through dispatch, not an exception
# ============================================================================


class DispatchThenTextLLM:
    def __init__(self) -> None:
        self.requests = []

    async def generate_complete(self, request):
        self.requests.append(request)

        if len(self.requests) == 1:
            class R:
                content = ""
                tool_calls = [
                    ToolCall(
                        id="d1",
                        name="dispatch_subagent",
                        arguments={"task": "too deep"},
                    )
                ]
                finish_reason = "tool_calls"

            return R()

        class R2:
            content = "gave up"
            tool_calls = []
            finish_reason = "stop"

        return R2()


@pytest.mark.asyncio
async def test_dispatch_at_depth_limit_returns_limit_text():
    llm = DispatchThenTextLLM()

    core_agent = CoreAgent(
        llm=llm,
        modules=_FakeModules(),
        providers=ProviderRuntime(),
        skills=_FakeSkills(),
        persona_source=lambda: "CORE",
        max_subagent_depth=0,
    )

    result = await core_agent.run("go")

    assert result.content == "gave up"

    tool_messages = [
        message
        for message in llm.requests[1].messages
        if message.role == "tool"
    ]

    assert tool_messages, "dispatch result missing"

    assert "Maximum Subagent depth exceeded" in (
        tool_messages[-1].content or ""
    )
