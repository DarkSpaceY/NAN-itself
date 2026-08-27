from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.nan_itself.tools import (
    AgentToolView,
    LocalToolProvider,
    ProviderRuntime,
    ProviderSpec,
    Provider,
    tool,
)
from src.nan_itself.tools import local as local_backend


def make_local_runtime(
    tmp_path: Path,
    *,
    builtin_tools: tuple | None = None,
) -> ProviderRuntime:
    return ProviderRuntime(
        builtin_config_path=(
            tmp_path / "does-not-exist.yaml"
        ),
        workspace_mcp_dir=(
            tmp_path / "workspace" / "tools" / "mcps"
        ),
        workspace_local_dir=(
            tmp_path / "workspace" / "tools" / "local"
        ),
        builtin_tools=builtin_tools,
    )


class CalculatorTools(LocalToolProvider):
    id = "calculator"

    @tool(description="Add two integers.")
    def add(self, a: int, b: int) -> int:
        return a + b

    @tool()
    def greet(
        self,
        who: str,
        punctuation: str = "!",
    ) -> str:
        """Greet someone."""
        return f"hello {who}{punctuation}"

    @tool(name="delayed_sum")
    async def slow_add(self, a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b


# ============================================================================
# Declaration and schema generation
# ============================================================================


def test_tool_decorator_collects_methods_in_order():
    instance = CalculatorTools()

    assert instance.tool_names() == (
        "add",
        "greet",
        "delayed_sum",
    )


def test_explicit_description_wins_over_docstring():
    instance = CalculatorTools()

    add = instance.get_tool_method("add")
    greet = instance.get_tool_method("greet")

    assert add.description == "Add two integers."
    assert greet.description == "Greet someone."


def test_schema_derives_from_type_annotations():
    tools = CalculatorTools().build_tools()

    add = tools["add"]

    assert add.inputSchema == {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }


def test_defaults_are_optional_in_schema():
    tools = CalculatorTools().build_tools()

    greet = tools["greet"]

    assert greet.inputSchema["required"] == [
        "who",
    ]

    assert (
        greet.inputSchema["properties"][
            "punctuation"
        ]["default"]
        == "!"
    )


def test_tool_rename_is_honored():
    tools = CalculatorTools().build_tools()

    assert "slow_add" not in tools
    assert "delayed_sum" in tools


def test_unannotated_parameter_is_rejected():
    class Broken(LocalToolProvider):
        id = "broken"

        @tool
        def do(self, value) -> str:
            return str(value)

    with pytest.raises(ValueError):
        Broken()


# ============================================================================
# Invocation semantics
# ============================================================================


@pytest.mark.asyncio
async def test_sync_and_async_tools_invoke_through_provider():
    spec = ProviderSpec(
        name="calculator",
        kind="local",
        source="<test>",
        origin="builtin",
    )

    provider = Provider(
        spec=spec,
        tools=CalculatorTools().build_tools(),
        instance=CalculatorTools(),
    )

    added = await provider.call_tool(
        "add",
        {"a": 2, "b": 3},
    )

    assert added.isError is False
    assert added.content[0].text == "5"

    greeted = await provider.call_tool(
        "greet",
        {"who": "nan"},
    )

    assert greeted.content[0].text == (
        "hello nan!"
    )

    delayed = await provider.call_tool(
        "delayed_sum",
        {"a": 1, "b": 1},
    )

    assert delayed.content[0].text == "2"


@pytest.mark.asyncio
async def test_non_string_results_are_json_serialized():
    class Structs(LocalToolProvider):
        id = "structs"

        @tool
        def payload(self) -> dict:
            return {"ok": True}

        @tool
        def nothing(self) -> None:
            return None

    provider = Provider(
        spec=ProviderSpec(
            name="structs",
            kind="local",
            source="<test>",
            origin="builtin",
        ),
        tools={},
        instance=Structs(),
    )

    provider.tools = (
        provider.instance.build_tools()
    )

    result = await provider.call_tool("payload", {})

    assert result.content[0].text == '{"ok": true}'

    result = await provider.call_tool("nothing", {})

    assert result.content[0].text == "null"


@pytest.mark.asyncio
async def test_invalid_arguments_become_error_result():
    provider = Provider(
        spec=ProviderSpec(
            name="calculator",
            kind="local",
            source="<test>",
            origin="builtin",
        ),
        tools=CalculatorTools().build_tools(),
        instance=CalculatorTools(),
    )

    result = await provider.call_tool(
        "add",
        {"a": "not-a-number", "b": 3},
    )

    assert result.isError is True


@pytest.mark.asyncio
async def test_handler_exception_becomes_error_result():
    class Boom(LocalToolProvider):
        id = "boom"

        @tool
        def explode(self) -> str:
            raise RuntimeError("boom")

    provider = Provider(
        spec=ProviderSpec(
            name="boom",
            kind="local",
            source="<test>",
            origin="builtin",
        ),
        tools=Boom().build_tools(),
        instance=Boom(),
    )

    result = await provider.call_tool("explode", {})

    assert result.isError is True

    assert "boom" in result.content[0].text


@pytest.mark.asyncio
async def test_unknown_local_tool_is_an_error():
    provider = Provider(
        spec=ProviderSpec(
            name="calculator",
            kind="local",
            source="<test>",
            origin="builtin",
        ),
        tools=CalculatorTools().build_tools(),
        instance=CalculatorTools(),
    )

    result = await provider.call_tool("missing", {})

    assert result.isError is True


# ============================================================================
# Builtin registration
# ============================================================================


@pytest.mark.asyncio
async def test_builtin_local_provider_is_registered(tmp_path):
    runtime = make_local_runtime(
        tmp_path,
        builtin_tools=(CalculatorTools,),
    )

    await runtime._load_builtin_config()
    runtime._load_builtin_tools()

    provider = runtime.get_provider("calculator")

    assert provider is not None
    assert provider.kind == "local"

    assert set(provider.tools) == {
        "add",
        "greet",
        "delayed_sum",
    }

    assert runtime.provider_names() == (
        "calculator",
    )


@pytest.mark.asyncio
async def test_builtin_local_provider_without_tools_is_rejected(
    tmp_path,
):
    class Empty(LocalToolProvider):
        id = "empty"

    runtime = make_local_runtime(
        tmp_path,
        builtin_tools=(Empty,),
    )

    with pytest.raises(ValueError):
        runtime._load_builtin_tools()


@pytest.mark.asyncio
async def test_duplicate_builtin_ids_are_rejected(tmp_path):
    class First(CalculatorTools):
        pass

    First.id = "same"
    CalculatorTools.id = "same"

    try:
        runtime = make_local_runtime(
            tmp_path,
            builtin_tools=(First, CalculatorTools),
        )

        with pytest.raises(ValueError):
            runtime._load_builtin_tools()

    finally:
        CalculatorTools.id = "calculator"


# ============================================================================
# Workspace discovery and hot reload
# ============================================================================

TOOL_FILE_SOURCE = '''
# @tool

from src.nan_itself.tools import (
    LocalToolProvider,
    tool,
)


class Notes(LocalToolProvider):
    id = "notes"

    def __init__(self):
        super().__init__()

        self.lines = []

    @tool(description="Append one line.")
    def append(self, line: str) -> int:
        self.lines.append(line)

        return len(self.lines)

    @tool(description="Read all lines.")
    def read(self) -> list:
        return list(self.lines)
'''


def write_tool_file(
    path: Path,
    source: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        source,
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_workspace_local_tool_is_discovered_and_stateful(
    tmp_path,
):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "notes.py",
        TOOL_FILE_SOURCE,
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    provider = runtime.get_provider("notes")

    assert provider is not None

    spec = provider.spec

    assert spec.kind == "local"
    assert spec.origin == "workspace"
    assert spec.file == (local_dir / "notes.py").resolve()

    # Instance state persists across calls.
    await provider.call_tool(
        "append",
        {"line": "first"},
    )

    await provider.call_tool(
        "append",
        {"line": "second"},
    )

    result = await provider.call_tool("read", {})

    assert result.content[0].text == (
        '["first", "second"]'
    )


@pytest.mark.asyncio
async def test_workspace_local_tool_supports_nested_directories(
    tmp_path,
):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "sub" / "deep" / "notes.py",
        TOOL_FILE_SOURCE,
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    assert runtime.get_provider("notes") is not None


@pytest.mark.asyncio
async def test_file_without_tool_header_is_ignored(tmp_path):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "plain.py",
        '''
from src.nan_itself.tools import (
    LocalToolProvider,
    tool,
)


class Plain(LocalToolProvider):
    id = "plain"

    @tool
    def ping(self) -> str:
        return "pong"
''',
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    assert runtime.providers == {}


@pytest.mark.asyncio
async def test_underscore_prefixed_file_is_ignored(tmp_path):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "_hidden.py",
        "# @tool\n\nx = 1\n",
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    assert runtime.providers == {}


@pytest.mark.asyncio
async def test_hot_reload_replaces_the_provider_instance(
    tmp_path,
    monkeypatch,
):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    tool_file = local_dir / "notes.py"

    write_tool_file(tool_file, TOOL_FILE_SOURCE)

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    old_provider = runtime.get_provider("notes")

    connect_calls: list[str] = []

    original_build = local_backend.build_provider

    def counting_build(spec, cls):
        connect_calls.append(spec.name)

        return original_build(spec, cls)

    monkeypatch.setattr(
        local_backend,
        "build_provider",
        counting_build,
    )

    await runtime._scan_workspace()

    # Unchanged file is not reloaded.
    assert connect_calls == []

    updated_source = TOOL_FILE_SOURCE.replace(
        '''    @tool(description="Read all lines.")
    def read(self) -> list:
        return list(self.lines)
''',
        '''    @tool(description="Read all lines.")
    def read(self) -> list:
        return list(self.lines)

    @tool(description="Clear all lines.")
    def clear(self) -> int:
        count = len(self.lines)

        self.lines.clear()

        return count
''',
    )

    await asyncio.sleep(0.01)

    write_tool_file(tool_file, updated_source)

    await runtime._scan_workspace()

    assert connect_calls == ["notes"]

    new_provider = runtime.get_provider("notes")

    assert new_provider is not old_provider

    assert "clear" in new_provider.tools


@pytest.mark.asyncio
async def test_removed_file_removes_the_provider(tmp_path):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    tool_file = local_dir / "notes.py"

    write_tool_file(tool_file, TOOL_FILE_SOURCE)

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    assert runtime.get_provider("notes") is not None

    tool_file.unlink()

    await runtime._scan_workspace()

    assert runtime.get_provider("notes") is None


@pytest.mark.asyncio
async def test_broken_file_records_error_without_crashing(
    tmp_path,
):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    tool_file = local_dir / "broken.py"

    write_tool_file(
        tool_file,
        "# @tool\n\ndef broken(:\n",
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    assert runtime.providers == {}

    assert tool_file.resolve() in runtime._local_tracker.errors

    # Fixing the file recovers on the next scan.
    await asyncio.sleep(0.01)

    write_tool_file(tool_file, TOOL_FILE_SOURCE)

    await runtime._scan_workspace()

    assert runtime.get_provider("notes") is not None


@pytest.mark.asyncio
async def test_file_with_multiple_classes_is_rejected(tmp_path):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "multi.py",
        '''
# @tool

from src.nan_itself.tools import (
    LocalToolProvider,
    tool,
)


class A(LocalToolProvider):
    id = "a"

    @tool
    def ping(self) -> str:
        return "pong"


class B(LocalToolProvider):
    id = "b"

    @tool
    def pong(self) -> str:
        return "ping"
''',
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    assert runtime.providers == {}

    assert (local_dir / "multi.py").resolve() in (
        runtime._local_tracker.errors
    )


@pytest.mark.asyncio
async def test_workspace_cannot_override_builtin_local(tmp_path):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "calc.py",
        '''
# @tool

from src.nan_itself.tools import (
    LocalToolProvider,
    tool,
)


class Calc(LocalToolProvider):
    id = "calculator"

    @tool
    def sub(self, a: int, b: int) -> int:
        return a - b
''',
    )

    runtime = make_local_runtime(
        tmp_path,
        builtin_tools=(CalculatorTools,),
    )

    runtime._load_builtin_tools()

    await runtime._scan_workspace()

    provider = runtime.get_provider("calculator")

    assert provider.spec.origin == "builtin"

    assert set(provider.tools) == {
        "add",
        "greet",
        "delayed_sum",
    }


@pytest.mark.asyncio
async def test_cross_source_name_conflict_is_rejected(tmp_path):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "first.py",
        TOOL_FILE_SOURCE,
    )

    write_tool_file(
        local_dir / "second.py",
        TOOL_FILE_SOURCE,
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    # The first file wins; the second is recorded as an error.
    assert set(runtime.providers) == {"notes"}

    first = (local_dir / "first.py").resolve()
    second = (local_dir / "second.py").resolve()

    assert runtime._local_sources == {
        first: {"notes"},
    }

    assert second in runtime._local_tracker.errors
    assert second not in runtime._local_sources


# ============================================================================
# Routing integration
# ============================================================================


class FakeSession:
    def __init__(self, tools=None):
        from mcp.types import Tool

        self.tools = {
            tool_item.name: tool_item
            for tool_item in (tools or [])
        }

        self.calls = []

    async def initialize(self):
        pass

    async def list_tools(self):
        return SimpleNamespace(
            tools=list(self.tools.values())
        )

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))

        return {"tool": name}


class FakeStack:
    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_route_switches_between_mcp_and_local(tmp_path):
    from mcp.types import Tool

    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "notes.py",
        TOOL_FILE_SOURCE,
    )

    runtime = make_local_runtime(tmp_path)

    mcp_tool = Tool(
        name="read_file",
        description="fake",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    )

    runtime.providers["files"] = Provider(
        spec=ProviderSpec(
            name="files",
            command="fake",
            source="fake.yaml",
            origin="workspace",
        ),
        stack=FakeStack(),
        session=FakeSession([mcp_tool]),
        tools={"read_file": mcp_tool},
    )

    await runtime._scan_workspace()

    view = AgentToolView(runtime)

    # Only route is visible before activation.
    initial = await view.list_tools()

    assert [item.name for item in initial] == [
        "route",
    ]

    route_schema = initial[0].inputSchema

    assert set(
        route_schema["properties"]["provider_name"][
            "enum"
        ]
    ) == {"files", "notes"}

    # Route to the MCP provider.
    result = await view.call_tool(
        "route",
        {"provider_name": "files"},
    )

    assert not result.isError

    tools = await view.list_tools()

    assert {item.name for item in tools} == {
        "route",
        "read_file",
    }

    # Route to the local provider.
    result = await view.call_tool(
        "route",
        {"provider_name": "notes"},
    )

    assert not result.isError

    tools = await view.list_tools()

    assert {item.name for item in tools} == {
        "route",
        "append",
        "read",
    }

    # Call a local tool through the view.
    await view.call_tool(
        "append",
        {"line": "hello"},
    )

    result = await view.call_tool("read", {})

    assert result.content[0].text == '["hello"]'


@pytest.mark.asyncio
async def test_views_route_local_providers_independently(tmp_path):
    local_dir = (
        tmp_path / "workspace" / "tools" / "local"
    )

    write_tool_file(
        local_dir / "notes.py",
        TOOL_FILE_SOURCE,
    )

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    main_view = AgentToolView(runtime)
    other_view = AgentToolView(runtime)

    await main_view.call_tool(
        "route",
        {"provider_name": "notes"},
    )

    main_tools = await main_view.list_tools()

    other_tools = await other_view.list_tools()

    assert {item.name for item in main_tools} == {
        "route",
        "append",
        "read",
    }

    assert [item.name for item in other_tools] == [
        "route",
    ]


# ============================================================================
# Framework unit edges
# ============================================================================


def test_tool_decorator_rejects_non_function():
    import inspect

    with pytest.raises(TypeError, match="only decorate"):
        tool()(str.upper)


def test_duplicate_tool_name_across_hierarchy_is_rejected():
    class Base(LocalToolProvider):
        id = "dup_base"

        @tool(description="base")
        def read(self) -> str:
            return "base"

    class Child(Base):
        id = "dup_child"

        @tool(description="child")
        def read(self) -> str:
            return "child"

    with pytest.raises(ValueError, match="duplicates tool name"):
        Child()


def test_var_args_parameter_is_rejected():
    class Broken(LocalToolProvider):
        id = "varargs"

        @tool(description="bad")
        def collect(self, *parts: int) -> int:
            return sum(parts)

    with pytest.raises(ValueError, match="var args"):
        Broken()


def test_missing_annotation_is_rejected():
    class Broken(LocalToolProvider):
        id = "no_annot"

        @tool(description="bad")
        def half(self, x):  # noqa: ANN001 - deliberately unannotated
            return x / 2

    with pytest.raises(ValueError, match="type-annotated"):
        Broken()


def test_docstring_fallback_and_custom_name_and_schema_titles():
    class Calc(LocalToolProvider):
        id = "calc"

        @tool(name="double_it")
        def double(self, n: int) -> int:
            """Double an integer."""
            return n * 2

    instance = Calc()

    method = instance.get_tool_method("double_it")

    assert method is not None
    assert method.description == "Double an integer."

    schema = method.input_schema()

    assert "title" not in schema
    assert all(
        "title" not in prop
        for prop in schema["properties"].values()
    )
    assert set(schema["properties"]) == {"n"}
    assert schema["required"] == ["n"]


def test_validate_class_edges():
    import abc

    from src.nan_itself.tools.local import validate_class

    with pytest.raises(TypeError):
        validate_class(object())

    with pytest.raises(TypeError):
        validate_class(LocalToolProvider)

    class Abstract(LocalToolProvider, abc.ABC):
        id = "abstract"

        @abc.abstractmethod
        def must_impl(self) -> str: ...

    with pytest.raises(TypeError, match="concrete"):
        validate_class(Abstract)

    class EmptyId(LocalToolProvider):
        id = ""

    with pytest.raises(ValueError, match="non-empty string id"):
        validate_class(EmptyId)


def test_has_tool_header_only_scans_head_window(tmp_path):
    from src.nan_itself.tools.local import has_tool_header

    late = tmp_path / "late.py"
    late.write_text(
        "\n" * 30 + "# @tool\n",
        encoding="utf-8",
    )

    assert has_tool_header(late) is False
    assert has_tool_header(tmp_path / "missing.py") is False


def test_load_class_from_file_rejects_multiple_classes(tmp_path):
    import sys as _sys

    from src.nan_itself.tools.local import load_class_from_file

    path = tmp_path / "multi.py"

    content = (
        "# @tool\n"
        "from src.nan_itself.tools import LocalToolProvider\n"
        "\n"
        "class A(LocalToolProvider):\n"
        "    id = \"a\"\n"
        "\n"
        "class B(LocalToolProvider):\n"
        "    id = \"b\"\n"
    )

    path.write_text(content, encoding="utf-8")

    before = set(_sys.modules)

    with pytest.raises(RuntimeError, match="exactly one"):
        load_class_from_file(path)

    # Failed discovery must not leak its synthetic module.
    leaked = [
        name
        for name in _sys.modules
        if name not in before
        and name.startswith("_workspace_tool_")
    ]

    assert leaked == []


def test_load_class_from_file_syntax_error_cleans_up(tmp_path):
    import sys as _sys

    from src.nan_itself.tools.local import load_class_from_file

    path = tmp_path / "broken.py"

    path.write_text("def oops(:\n", encoding="utf-8")

    before = set(_sys.modules)

    with pytest.raises(SyntaxError):
        load_class_from_file(path)

    leaked = [
        name
        for name in _sys.modules
        if name not in before
        and name.startswith("_workspace_tool_")
    ]

    assert leaked == []


def test_local_provider_spec_shape():
    spec = local_backend.local_provider_spec(
        name="n",
        source="somewhere.py",
        origin="workspace",
    )

    assert spec.kind == "local"
    assert spec.file is None
    assert spec.source == "somewhere.py"


# ============================================================================
# Hot reload must never execute stale bytecode
# ============================================================================


@pytest.mark.asyncio
async def test_same_size_edit_still_reloads_fresh_code(tmp_path):
    """
    Regression: the dynamic loader used spec_from_file_location,
    whose bytecode cache validates on coarse mtime + size. A
    same-length edit within the same second executed STALE code.
    """
    import time

    local_dir = tmp_path / "workspace" / "tools" / "local"

    target = local_dir / "probe.py"

    V1 = (
        "# @tool\n\n"
        "from src.nan_itself.tools import LocalToolProvider, tool\n\n"
        "class Probe(LocalToolProvider):\n"
        "    id = \"probe\"\n\n"
        "    @tool(description=\"value\")\n"
        "    def value(self) -> str:\n"
        "        return \"1\"\n"
    )

    V2 = V1.replace('return "1"', 'return "2"')

    assert len(V1) == len(V2)

    write_tool_file(target, V1)

    runtime = make_local_runtime(tmp_path)

    await runtime._scan_workspace()

    result = await runtime.call_tool("probe", "value", {})

    assert result.content[0].text == "1"

    time.sleep(0.05)  # same wall-clock second, different intent

    write_tool_file(target, V2)

    await runtime._scan_workspace()

    result = await runtime.call_tool("probe", "value", {})

    assert result.content[0].text == "2"
