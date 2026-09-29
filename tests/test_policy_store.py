"""Changing policy at runtime: who may, within what, and with what record.

These thresholds decide what goes out without a person, so the tests pin the
safety properties first: no token no edit, nothing outside its bounds, no
unrecorded change, a revert that is itself recorded, and a preview that never
touches the live policy running cases read.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import config, policy_store
from sanwaad.graph.nodes import auto_post_allowed


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_store, "AUDIT_PATH", tmp_path / "policy_audit.jsonl")
    yield
    for name, values in policy_store.DEFAULTS.items():      # never leak a change
        for field, value in values.items():
            policy_store._set(name, field, value)


def _state(severity: int) -> dict:
    return {"triage": {"severity": severity, "category": "billing", "needs_private_data": False},
            "draft": {"text": "x", "promises_compensation": False},
            "grounding": {"grounded": True}}


def test_a_change_takes_effect_everywhere_the_policy_is_read():
    assert auto_post_allowed(_state(2))[0] is True            # 2 posts by default
    policy_store.apply("review", "auto_post_max_severity", 1, reason="tighten after an incident",
                       actor="ops lead")
    assert config.REVIEW.auto_post_max_severity == 1
    assert auto_post_allowed(_state(2))[0] is False           # the gate sees it at once


@pytest.mark.parametrize("policy,field,value,message", [
    ("review", "auto_post_max_severity", 4, "outside the allowed range"),
    ("review", "require_grounded", False, "not editable"),
    ("review", "forbid_auto_compensation", False, "not editable"),
    ("actions", "reversal_ceiling_inr", 50_000, "outside the allowed range"),
    ("crisis", "watch_cluster", 2.5, "whole number"),
])
def test_bounds_and_uneditable_rules_are_refused(policy, field, value, message):
    with pytest.raises(policy_store.PolicyError, match=message):
        policy_store.apply(policy, field, value, reason="testing the bounds", actor="t")


def test_rules_that_span_two_fields_hold():
    policy_store.apply("judge", "troll_ceiling", 0.5, reason="step one", actor="t")
    with pytest.raises(policy_store.PolicyError, match="below the audience floor"):
        policy_store.apply("judge", "audience_floor", 0.45, reason="crosses it", actor="t")
    with pytest.raises(policy_store.PolicyError, match="watch cluster < crisis cluster"):
        policy_store.apply("crisis", "watch_cluster", 6, reason="same as crisis", actor="t")


def test_a_change_without_a_reason_or_a_name_is_refused():
    with pytest.raises(policy_store.PolicyError, match="reason is required"):
        policy_store.apply("review", "auto_post_max_severity", 1, reason="", actor="ops")
    with pytest.raises(policy_store.PolicyError, match="who is making"):
        policy_store.apply("review", "auto_post_max_severity", 1, reason="tighten it", actor=" ")
    assert policy_store.history() == []                        # refused means unrecorded


def test_changes_are_logged_replayed_and_reverted_as_new_entries():
    first = policy_store.apply("crisis", "crisis_cluster", 8, reason="fewer false crises",
                               actor="ops lead")
    policy_store._set("crisis", "crisis_cluster", 6)           # as if the process restarted
    assert policy_store.replay() == []
    assert config.CRISIS.crisis_cluster == 8                   # survived the "restart"

    undo = policy_store.revert(first["id"], reason="crisis missed on Friday", actor="cto")
    assert config.CRISIS.crisis_cluster == 6
    assert undo["reverts"] == first["id"]
    assert [h["new"] for h in policy_store.history()] == [8, 6]   # nothing erased


def test_reverting_a_value_that_changed_again_is_refused():
    first = policy_store.apply("crisis", "crisis_cluster", 8, reason="step one", actor="a")
    policy_store.apply("crisis", "crisis_cluster", 10, reason="step two", actor="b")
    with pytest.raises(policy_store.PolicyError, match="changed again"):
        policy_store.revert(first["id"], reason="undo step one", actor="c")


def test_a_bad_log_entry_is_skipped_and_the_default_kept():
    policy_store.AUDIT_PATH.write_text(
        '{"policy": "review", "field": "auto_post_max_severity", "new": 9}\n'
        'not json at all\n', encoding="utf-8")
    problems = policy_store.replay()
    assert len(problems) == 1 and "auto_post_max_severity" in problems[0]
    assert config.REVIEW.auto_post_max_severity == policy_store.DEFAULTS["review"]["auto_post_max_severity"]


def test_preview_counts_changed_decisions_without_touching_the_live_policy():
    states = [_state(1), _state(2), _state(2), _state(3)]
    result = policy_store.preview("review", "auto_post_max_severity", 1, states)
    assert result["available"] and result["checked"] == 4
    assert result["changed"] == 2                               # the two severity-2 cases
    assert {e["after"] for e in result["examples"]} == {"held for a person"}
    assert config.REVIEW.auto_post_max_severity == 2            # live policy untouched


def test_preview_is_honest_where_it_cannot_replay():
    result = policy_store.preview("crisis", "crisis_cluster", 8, [_state(2)])
    assert result["available"] is False and "cannot be replayed" in result["why"]


# --- API ----------------------------------------------------------------------

def _client():
    from sanwaad.api import server

    return TestClient(server.app)


def test_editing_is_off_without_a_configured_token(monkeypatch):
    monkeypatch.delenv("SANWAAD_ADMIN_TOKEN", raising=False)
    r = _client().post("/api/settings/change", json={
        "policy": "review", "field": "auto_post_max_severity", "value": 1,
        "reason": "tighten", "actor": "x"})
    assert r.status_code == 403 and "disabled" in r.json()["detail"]
    assert _client().get("/api/settings").json()["editing_enabled"] is False


def test_the_wrong_token_is_refused_and_the_right_one_applies(monkeypatch):
    monkeypatch.setenv("SANWAAD_ADMIN_TOKEN", "s3cret")
    body = {"policy": "review", "field": "auto_post_max_severity", "value": 1,
            "reason": "tighten after an incident", "actor": "ops lead"}
    assert _client().post("/api/settings/change", json=body,
                          headers={"X-Admin-Token": "guess"}).status_code == 401
    assert config.REVIEW.auto_post_max_severity == 2
    r = _client().post("/api/settings/change", json=body, headers={"X-Admin-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["new"] == 1
    settings = _client().get("/api/settings").json()
    row = next(e for e in settings["editable"] if e["field"] == "auto_post_max_severity")
    assert (row["value"], row["default"]) == (1, 2)
    assert settings["history"][0]["reason"] == "tighten after an incident"


def test_an_out_of_bounds_value_through_the_api_is_a_400(monkeypatch):
    monkeypatch.setenv("SANWAAD_ADMIN_TOKEN", "s3cret")
    r = _client().post("/api/settings/change", headers={"X-Admin-Token": "s3cret"}, json={
        "policy": "actions", "field": "reversal_ceiling_inr", "value": 1e9,
        "reason": "let big refunds through", "actor": "x"})
    assert r.status_code == 400 and "outside the allowed range" in r.json()["detail"]
