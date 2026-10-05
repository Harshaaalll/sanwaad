"""The operator views: a case's explanation, the overview, and the policy page.

The property that matters most is that an explanation cannot disagree with the
decision it explains. Each is computed by the function that decides, and these
pin that: the gate's answer is the checklist's answer, the score is the sum of
its parts, and the voice reasons are the triggers that fired.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import config, pipeline
from sanwaad.explain import explain
from sanwaad.graph.nodes import (
    auto_post_allowed,
    escalation_checks,
    escalation_node,
    prioritise,
    review_checks,
)
from sanwaad.overview import status_of, summarise


def _state(**over) -> dict:
    base = {
        "case_id": "c1",
        "complaint": {"text": "charged twice", "channel": "reddit"},
        "triage": {"severity": 1, "category": "refund", "needs_private_data": False},
        "draft": {"text": "sorry", "promises_compensation": False},
        "grounding": {"grounded": True},
    }
    for key, value in over.items():
        base[key] = {**base[key], **value} if isinstance(base.get(key), dict) else value
    return base


GATE_CASES = [
    _state(),
    _state(grounding={"grounded": False}),
    _state(draft={"promises_compensation": True}),
    _state(injection_flagged=True),
    _state(pattern={"level": "crisis"}),
    _state(triage={"severity": 4}),
    _state(triage={"needs_private_data": True}, grounding={"grounded": False}),
    _state(actions=[{"proposal": {"risk": "write_high"}}]),
]


@pytest.mark.parametrize("state", GATE_CASES)
def test_the_checklist_and_the_gate_give_the_same_answer(state):
    checks, allowed, reason = review_checks(state)
    assert (allowed, reason) == auto_post_allowed(state)
    if not allowed:
        first_failure = next(c for c in checks if not c["passed"])
        assert first_failure["detail"] == reason


def test_every_failing_hard_rule_is_reported_not_just_the_first():
    checks, allowed, _ = review_checks(
        _state(triage={"needs_private_data": True}, grounding={"grounded": False}))
    failed = [c["rule"] for c in checks if not c["passed"]]
    assert not allowed
    assert failed == ["every claim grounded", "needs no private data"]


def test_the_score_is_the_sum_of_its_parts():
    p = prioritise({"is_complaint": True, "severity": 3},
                   {"authenticity": 0.9, "reach": 7000, "history_with_brand": 2},
                   {"level": "watch", "cluster_size": 3})
    assert p.components == {"severity": 30.0, "authenticity": 7.2, "reach": 7.0,
                            "pattern": 15.0, "history": 5.0}
    assert p.score == round(sum(p.components.values()), 1)


@pytest.mark.asyncio
async def test_voice_reasons_are_exactly_the_triggers_that_fired():
    state = _state(triage={"severity": 4, "category": "account_access"},
                   complaint={"text": "wallet frozen with ₹18,000, please call me"})
    fired = [c["detail"] for c in escalation_checks(state) if c["fired"]]
    out = await escalation_node(state)
    assert out["escalation"]["reasons"] == fired
    assert len(fired) == 4       # severity, category, amount, asked to talk
    assert len(escalation_checks(state)) == 5, "a trigger that did not fire is still listed"


def test_an_explanation_before_drafting_has_no_review_section():
    state = _state()
    state.pop("draft")
    state["priority"] = {"tier": "ignore", "score": 12.0, "reasons": ["not a complaint"]}
    x = explain(state)
    assert x["review"] is None
    assert x["priority"]["recomputed"], "old cases get their parts from the same function"


def test_a_held_case_reports_the_reason_it_was_held():
    x = explain(_state(triage={"severity": 4}), pending={"reason": "severity 4 is above 3"})
    assert x["review"]["at_the_time"] == {"allowed": False, "reason": "severity 4 is above 3"}


# --- overview -----------------------------------------------------------------

def _case(**kw) -> dict:
    return {"complaint": {"channel": "reddit"}, "triage": {"category": "refund", "severity": 3},
            **kw}


def test_status_follows_the_case_through_the_pipeline():
    assert status_of(_case(draft={"text": "x"})) == "needs_review"
    assert status_of(_case(draft={}, review={"auto": False},
                           escalation={"needed": True})) == "awaiting_call"
    assert status_of(_case(closure={"resolved": True})) == "resolved"
    assert status_of(_case(closure={"resolved": False})) == "closed_unresolved"


def test_an_approval_with_rewritten_text_counts_as_an_edit():
    cases = [_case(draft={"text": "Sorry about this."},
                   review={"auto": False, "decision": "approve", "final_text": "Sorry — refunded."}),
             _case(draft={"text": "Sorry about this."},
                   review={"auto": False, "decision": "approve", "final_text": " Sorry about this. "})]
    r = summarise(cases)["reviews"]
    assert (r["approved_unchanged"], r["edited"]) == (1, 1)


def test_overview_counts_are_traceable_to_cases():
    cases = [
        _case(draft={}, review={"auto": True}, closure={"resolved": True, "total_cost_inr": 0.02},
              cost_inr=0.02),
        _case(draft={}, review={"auto": False, "decision": "edit"},
              closure={"resolved": False, "total_cost_inr": 0.04, "degraded_steps": ["draft"]},
              cost_inr=0.04),
        _case(draft={"text": "x"}, pattern={"level": "crisis", "theme": "UPI down",
                                           "cluster_size": 6}, cost_inr=0.01),
    ]
    o = summarise(cases, dead_letters=1)
    assert o["status"] == {"resolved": 1, "closed_unresolved": 1, "needs_review": 1}
    assert o["reviews"]["auto_rate"] == 0.5 and o["reviews"]["edited"] == 1
    assert o["outcomes"] == {"closed": 2, "resolved": 1, "resolved_rate": 0.5, "degraded": 1}
    assert o["cost"]["avg_per_closed_case_inr"] == 0.03
    assert o["cost"]["total_inr"] == 0.07
    assert o["incidents"] == [{"theme": "UPI down", "level": "crisis", "cases": 1, "cluster_size": 6}]
    assert o["dead_letters"] == 1


# --- API -------------------------------------------------------------------------

def test_settings_report_the_live_policy_values():
    from sanwaad.api import server

    body = TestClient(server.app).get("/api/settings").json()
    review = next(v for k, v in body["policies"].items() if k.startswith("Review"))
    assert review["values"]["auto_post_max_severity"] == config.REVIEW.auto_post_max_severity
    assert body["fixed"]["autonomy_max_severity"] == 3


@pytest.mark.asyncio
async def test_a_case_from_the_api_carries_its_explanation(monkeypatch, tmp_path):
    from httpx import ASGITransport, AsyncClient

    from sanwaad.api import server
    from sanwaad.connectors import get_connector

    monkeypatch.setattr(pipeline, "CHECKPOINT_PATH", tmp_path / "x.sqlite")
    items = await get_connector("mock").fetch(limit=6)
    out = await pipeline.run_case(items[5])          # the ₹640 double debit
    async with AsyncClient(transport=ASGITransport(app=server.app), base_url="http://t") as client:
        body = (await client.get(f"/api/cases/{out['case_id']}")).json()
        overview = (await client.get("/api/overview")).json()
    x = body["explain"]
    assert x["priority"]["components"]
    assert x["review"]["at_the_time"]["allowed"] is False
    assert overview["total"] == 1
    await pipeline.close_sessions()


# --- trends ---------------------------------------------------------------------

def test_trends_bucket_by_day_and_compare_the_last_two_weeks():
    from datetime import datetime, timezone

    from sanwaad.overview import trends

    now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    cases = [
        _case(opened_at="2026-10-05T09:00:00+00:00", draft={"text": "x"}),               # today, waiting
        _case(opened_at="2026-10-03T09:00:00+00:00", closed_at="2026-10-04T10:00:00+00:00",
              closure={"resolved": True}, review={"decided_at": "2026-10-03T10:30:00+00:00"}),
        _case(opened_at="2026-09-28T09:00:00+00:00", triage={"category": "billing", "severity": 2},
              review={"decided_at": "2026-09-28T15:00:00+00:00"}),                       # last week
        _case(opened_at="2026-08-01T09:00:00", closure={"resolved": True}),              # too old, naive time
    ]
    t = trends(cases, now)
    assert t["days"][-1] == "2026-10-05" and len(t["days"]) == 14
    assert t["opened"][-1] == 1 and t["opened"][-3] == 1 and sum(t["opened"]) == 3
    assert t["resolved"][-2] == 1 and sum(t["resolved"]) == 1
    by = {m["category"]: m for m in t["momentum"]}
    assert by["refund"] == {"category": "refund", "this_week": 2, "last_week": 0, "change": 2}
    assert by["billing"]["change"] == -1
    assert t["oldest_waiting_at"] == "2026-10-05T09:00:00+00:00"


def test_first_response_reports_median_p90_and_share_within_target(monkeypatch):
    from datetime import datetime, timedelta, timezone

    from sanwaad import overview

    monkeypatch.setattr(overview, "SLA_FIRST_RESPONSE_MINUTES", 60)
    start = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    cases = [_case(opened_at=start.isoformat(),
                   review={"decided_at": (start + timedelta(minutes=m)).isoformat()})
             for m in (10, 20, 30, 50, 300)]
    fr = overview.trends(cases, start + timedelta(days=1))["first_response"]
    assert fr["median_minutes"] == 30.0
    assert fr["p90_minutes"] == 300.0          # nearest rank: the slowest of five
    assert fr["within_target"] == 0.8 and fr["count"] == 5


def test_a_case_with_no_timestamps_adds_nothing_to_the_trends():
    from sanwaad.overview import trends

    t = trends([_case(draft={"text": "x"})])
    assert sum(t["opened"]) == 0 and t["momentum"] == [] and t["first_response"]["count"] == 0
