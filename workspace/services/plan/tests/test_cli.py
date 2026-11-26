"""The `plan` command: output, exit codes and the file it edits."""

from __future__ import annotations

import plan_cli


def _run(monkeypatch, tmp_path, *argv):
    monkeypatch.setenv("NAN_PLAN_FILE", str(tmp_path / "plan.json"))

    return plan_cli.main(list(argv))


def test_show_without_goals(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, tmp_path) == 0

    assert capsys.readouterr().out.strip() == "no goals"


def test_add_then_show(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, tmp_path, "add", "page renders") == 0
    assert _run(monkeypatch, tmp_path, "add", "slider works", "--under", "g1") == 0

    assert _run(monkeypatch, tmp_path) == 0

    assert capsys.readouterr().out.splitlines()[-2:] == [
        "g1 active: page renders",
        "  g2 active: slider works",
    ]


def test_done_reports_closed_children(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, "add", "root")
    _run(monkeypatch, tmp_path, "add", "child", "--under", "g1")
    capsys.readouterr()

    assert _run(monkeypatch, tmp_path, "done", "g1") == 0

    assert capsys.readouterr().out.strip() == "done g1 (1 children closed)"


def test_unknown_goal_exits_3(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, tmp_path, "done", "g9") == 3

    assert '"code": "unknown_goal"' in capsys.readouterr().err


def test_unknown_parent_exits_3(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, "add", "orphan", "--under", "g9") == 3


def test_missing_argument_exits_2(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, "done") == 2
