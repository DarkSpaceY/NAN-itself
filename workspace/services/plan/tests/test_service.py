"""What the hub receives for each turn: full tree, delta, or nothing."""

from __future__ import annotations

import plan as tree
from fastapi.testclient import TestClient
from main import HINT, PlanService, build_app
from nan_service_kit import ServiceConfig


def _service(tmp_path) -> PlanService:
    service = PlanService(ServiceConfig(service_id="plan", port=9003))

    service.path = tmp_path / "plan.json"

    return service


def test_the_app_answers_health_like_every_service(tmp_path):
    service = _service(tmp_path)

    assert TestClient(build_app(service)).get("/health").json() == {
        "status": "ok",
        "service_id": "plan",
    }


def _turn(**overrides) -> dict[str, object]:
    turn: dict[str, object] = {
        "agent_id": "root",
        "epoch": 1,
        "origin": "normal",
        "started_at": 10.0,
    }

    turn.update(overrides)

    return turn


def test_no_plan_still_announces_the_tool_once(tmp_path):
    service = _service(tmp_path)

    assert service.contribution(_turn(origin="first")) == HINT

    assert service.contribution(_turn(started_at=20.0)) == ""


def test_the_first_goal_after_the_hint_arrives_as_a_delta(tmp_path):
    service = _service(tmp_path)

    service.contribution(_turn(origin="first"))

    tree.save(service.path, [tree.Goal(id="g1", text="page renders")])

    assert service.contribution(_turn(started_at=20.0)) == "- g1 new: page renders"


def test_a_new_epoch_gets_the_whole_tree(tmp_path):
    service = _service(tmp_path)

    goals = [tree.Goal(id="g1", text="page renders")]

    tree.save(service.path, goals)

    text = service.contribution(_turn(origin="first"))

    assert text == HINT + "\n- g1 active: page renders"


def test_an_edit_inside_the_epoch_sends_only_the_delta(tmp_path):
    service = _service(tmp_path)

    tree.save(service.path, [tree.Goal(id="g1", text="page renders")])

    service.contribution(_turn(origin="first"))

    tree.save(
        service.path,
        [
            tree.Goal(id="g1", text="page renders", status=tree.DONE),
            tree.Goal(id="g2", text="export button", parent="g1"),
        ],
    )

    text = service.contribution(_turn(started_at=20.0))

    assert text.splitlines() == [
        "- g1 done",
        "- g2 new (under g1): export button",
    ]


def test_an_untouched_tree_injects_nothing(tmp_path):
    service = _service(tmp_path)

    tree.save(service.path, [tree.Goal(id="g1", text="page renders")])

    service.contribution(_turn(origin="first"))

    assert service.contribution(_turn(started_at=20.0)) == ""


def test_a_cleared_context_gets_the_whole_tree_again(tmp_path):
    service = _service(tmp_path)

    tree.save(service.path, [tree.Goal(id="g1", text="page renders")])

    service.contribution(_turn(origin="first"))

    text = service.contribution(_turn(origin="context_cleared", started_at=30.0))

    assert text.startswith(HINT)


def test_a_replayed_turn_gets_the_same_text(tmp_path):
    service = _service(tmp_path)

    tree.save(service.path, [tree.Goal(id="g1", text="page renders")])

    first = service.contribution(_turn(origin="first"))

    tree.save(service.path, [tree.Goal(id="g1", text="page renders", status=tree.DONE)])

    # The hub re-injects a parked turn with the same started_at.
    assert service.contribution(_turn(origin="first")) == first
