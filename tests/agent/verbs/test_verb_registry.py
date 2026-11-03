"""The verb registry: which capability set an agent gets, and the
protocol handed to the model."""

from __future__ import annotations

from nan.agent.verbs import CHILD_VERBS, ROOT_VERBS, VERBS, format_shell_output, verb_definitions

EXPECTED = {"spawn", "sleep", "shell", "send", "kill", "finish"}


def test_the_registry_is_the_whole_verb_set() -> None:
    assert set(VERBS) == EXPECTED


def test_the_root_cannot_finish() -> None:
    assert set(ROOT_VERBS) == EXPECTED - {"finish"}
    assert set(CHILD_VERBS) == EXPECTED


def test_verb_definitions_mirror_the_set() -> None:
    definitions = verb_definitions(ROOT_VERBS)

    assert [definition.name for definition in definitions] == list(ROOT_VERBS)
    assert all(definition.input_schema["type"] == "object" for definition in definitions)


def test_verb_definitions_are_the_verb_contracts() -> None:
    spawn = verb_definitions(ROOT_VERBS)[list(ROOT_VERBS).index("spawn")]

    assert spawn.input_schema["required"] == ["task"]
    assert spawn.input_schema["additionalProperties"] is False


def test_format_shell_output_keeps_the_longs_head_and_tail() -> None:
    output = "y" * 1000

    text = format_shell_output(output, 0, 100)

    assert text.startswith("exit code: 0\n")
    assert "truncated 900 chars" in text
    assert text.count("y") == 100


def test_format_shell_output_leaves_short_output_alone() -> None:
    assert format_shell_output("hi", 0, 100) == "exit code: 0\nhi"
    assert format_shell_output("", None, 100) == "exit code: None\n"
