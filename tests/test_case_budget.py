"""Tests for the per-case cost ceiling.

The support loop has had a budget since it was written, because a loop can
iterate freely and something has to stop it. The case graph never did: its
shape bounds its own length, so nothing could run away, and "nothing can run
away" was quietly taken to mean "no ceiling is needed".

Those are different claims. A ceiling here is a policy statement rather than a
runaway guard — past a certain spend, a person handling one case is cheaper
than the model continuing to try, and a case that has cost that much is a case
something unusual has happened to. These tests pin that it is enforced against
the same number the closure reports, and that going over degrades the way an
outage does rather than in some new way nothing downstream understands.
"""

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import llm
from sanwaad.config import MAX_CASE_COST_INR
from sanwaad.graph.nodes import _spent


class Tiny(BaseModel):
    label: str


def _no_model(monkeypatch):
    """A client that must never be reached."""
    def boom():
        raise AssertionError("the model was called after the budget was spent")

    monkeypatch.setattr(llm, "_get_client", boom)


@pytest.mark.asyncio
async def test_a_step_under_the_ceiling_runs_normally(monkeypatch):
    monkeypatch.setattr(llm, "_get_client", lambda: object())

    async def fake(*_a, **_kw):
        return object()

    monkeypatch.setattr(llm, "_generate", fake)
    monkeypatch.setattr(llm, "_parse", lambda *_a: Tiny(label="ok"))
    result, cost = await llm.structured(model="m", system="s", user="u", schema=Tiny,
                                        spent_inr=1.0, budget_inr=5.0)
    assert result.label == "ok"
    assert not cost["degraded"]


@pytest.mark.asyncio
async def test_a_spent_case_does_not_call_the_model_at_all(monkeypatch):
    """The ceiling has to be checked before the call, not after. Spending the
    money and then reporting that you were not allowed to is an audit trail,
    not a budget."""
    _no_model(monkeypatch)
    result, cost = await llm.structured(
        model="m", system="s", user="u", schema=Tiny,
        stage="draft", spent_inr=5.0, budget_inr=5.0,
        offline_fallback={"label": "holding"})
    assert result.label == "holding"
    assert cost["degraded"] and cost["model"] == "over_budget"
    assert cost["inr"] == 0.0


@pytest.mark.asyncio
async def test_over_budget_degrades_exactly_like_an_outage(monkeypatch):
    """Reusing the degraded path is the point: the grounding gate already
    sends a degraded check to a person, and a budget breach wants precisely
    that. A new marker would need every downstream branch taught about it."""
    _no_model(monkeypatch)
    _, cost = await llm.structured(
        model="m", system="s", user="u", schema=Tiny, stage="ground_check",
        spent_inr=9.0, budget_inr=5.0, offline_fallback={"label": "x"})
    assert cost["degraded"] is True
    # ...and it still says which kind of degradation it was, or a cost problem
    # would hide inside the availability metric.
    assert "budget" in cost["errors"][0]


@pytest.mark.asyncio
async def test_a_step_with_no_safe_default_stops_the_case(monkeypatch):
    _no_model(monkeypatch)
    with pytest.raises(llm.BudgetExceeded) as exc:
        await llm.structured(model="m", system="s", user="u", schema=Tiny,
                             stage="plan", spent_inr=6.0, budget_inr=5.0)
    assert exc.value.stage == "plan"
    assert exc.value.spent_inr == 6.0


@pytest.mark.asyncio
async def test_no_ceiling_means_no_ceiling(monkeypatch):
    """Passing no budget must not quietly apply a default. The evals and the
    loop call `structured` directly, and a surprise ceiling there would look
    like a model failure."""
    monkeypatch.setattr(llm, "_get_client", lambda: object())

    async def fake(*_a, **_kw):
        return object()

    monkeypatch.setattr(llm, "_generate", fake)
    monkeypatch.setattr(llm, "_parse", lambda *_a: Tiny(label="ok"))
    result, _ = await llm.structured(model="m", system="s", user="u", schema=Tiny,
                                     spent_inr=10_000.0)
    assert result.label == "ok"


def test_spend_is_read_from_the_entries_the_closure_reports():
    """One number, not two. A separate counter would eventually disagree with
    the one in the closure, and the ceiling would be enforced against the
    figure nobody was looking at."""
    state = {"costs": [{"inr": 0.5}, {"inr": 1.25}, {"inr": 0.0}]}
    assert _spent(state) == 1.75
    assert _spent({}) == 0.0
    assert _spent({"costs": [{"model": "cache"}]}) == 0.0


def test_the_default_ceiling_is_far_above_a_routine_case():
    """A ceiling a normal case can reach is a random escalation generator. The
    trajectory eval's mean cost per case is the number to compare against."""
    assert MAX_CASE_COST_INR >= 1.0
