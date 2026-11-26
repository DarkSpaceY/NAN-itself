"""The tree's state machine and rendering, without HTTP or FastAPI."""

from __future__ import annotations

import plan as tree


def test_add_assigns_sequential_ids():
    goals: list[tree.Goal] = []

    first = tree.add(goals, "page renders in the browser")

    second = tree.add(goals, "depth slider visible", first.id)

    assert (first.id, second.id) == ("g1", "g2")
    assert second.parent == "g1"
    assert [goal.status for goal in goals] == [tree.ACTIVE, tree.ACTIVE]


def test_add_rejects_an_unknown_parent():
    goals = [tree.Goal(id="g1", text="root")]

    try:
        tree.add(goals, "orphan", "g9")

    except KeyError as exc:
        assert exc.args[0] == "g9"

    else:
        raise AssertionError("unknown parent accepted")


def test_add_rejects_empty_text():
    try:
        tree.add([], "   ")

    except ValueError:
        return

    raise AssertionError("empty goal accepted")


def test_close_cascades_to_open_descendants():
    goals = [
        tree.Goal(id="g1", text="root"),
        tree.Goal(id="g2", text="child", parent="g1"),
        tree.Goal(id="g3", text="grandchild", parent="g2"),
        tree.Goal(id="g4", text="unrelated"),
    ]

    goal, closed = tree.close(goals, "g1", tree.DONE)

    assert goal.status == tree.DONE
    assert closed == 2
    assert {goal.id: goal.status for goal in goals} == {
        "g1": tree.DONE,
        "g2": tree.DONE,
        "g3": tree.DONE,
        "g4": tree.ACTIVE,
    }


def test_close_keeps_statuses_already_set():
    goals = [
        tree.Goal(id="g1", text="root"),
        tree.Goal(id="g2", text="abandoned", parent="g1", status=tree.DROPPED),
    ]

    _, closed = tree.close(goals, "g1", tree.DONE)

    assert closed == 0
    assert goals[1].status == tree.DROPPED


def test_close_rejects_an_unknown_id():
    try:
        tree.close([], "g7", tree.DONE)

    except KeyError:
        return

    raise AssertionError("unknown goal closed")


def test_render_full_indents_the_tree():
    goals = [
        tree.Goal(id="g1", text="page renders"),
        tree.Goal(id="g2", text="slider works", parent="g1"),
        tree.Goal(id="g3", text="readme updated", status=tree.DONE),
    ]

    assert tree.render_full(goals).splitlines() == [
        "- g1 active: page renders",
        "  - g2 active: slider works",
        "- g3 done: readme updated",
    ]


def test_render_delta_reports_only_changes():
    seen = {"g1": tree.ACTIVE, "g2": tree.ACTIVE}

    goals = [
        tree.Goal(id="g1", text="page renders", status=tree.DONE),
        tree.Goal(id="g3", text="export button", parent="g1"),
    ]

    assert tree.render_delta(seen, goals).splitlines() == [
        "- g1 done",
        "- g3 new (under g1): export button",
        "- g2 removed",
    ]


def test_render_delta_is_empty_without_changes():
    goals = [tree.Goal(id="g1", text="page renders")]

    assert tree.render_delta(tree.snapshot(goals), goals) == ""


def test_missing_or_broken_file_is_an_empty_plan(tmp_path):
    path = tmp_path / "plan.json"

    assert tree.load(path) == []

    path.write_text("{not json", encoding="utf-8")

    assert tree.load(path) == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "nested" / "plan.json"

    goals = [tree.Goal(id="g1", text="页面渲染正确"), tree.Goal(id="g2", text="x", parent="g1")]

    tree.save(path, goals)

    assert tree.load(path) == goals
    assert not list(path.parent.glob("*.tmp"))
