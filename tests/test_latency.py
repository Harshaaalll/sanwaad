"""Tests for the latency budget: declared, measured, and compared.

`Route.max_latency_ms` existed for a while as a number nothing read. A budget
nothing checks is a comment, and a comment cannot tell you that ground_check
quietly went from two seconds to nine. These tests pin the three links in the
chain: the router declares a budget for every step, the model layer stamps it
on the span and marks a breach, and `stage_stats` counts breaches per stage.

They also pin what a breach must *not* do. A budget is not a deadline —
`timeout_s` already decides when to give up. A step that answers correctly in
four seconds under a two-second budget is a degradation worth reporting and a
terrible thing to abort, so the result comes back normally either way.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from sanwaad import llm
from sanwaad.obs import Tracer, budget_breaches, stage_stats
from sanwaad.router import Stage, route


class _Answer(BaseModel):
    ok: bool


# --- the router declares one for every step -------------------------------

@pytest.mark.parametrize("stage", ["triage", "judge", "draft", "ground_check", "plan", "voice"])
def test_every_step_declares_a_latency_budget(stage: Stage):
    r = route(stage)
    assert r.max_latency_ms, f"{stage} has no latency budget"


@pytest.mark.parametrize("stage", ["triage", "judge", "draft", "ground_check", "plan", "voice"])
def test_a_budget_is_tighter_than_the_timeout_it_lives_under(stage: Stage):
    """If the budget were above the timeout it could never be breached — the
    call would be killed first, and the metric would read a flat zero forever."""
    r = route(stage)
    assert r.max_latency_ms < r.timeout_s * 1000


def test_the_reasoning_tier_gets_a_looser_budget_than_the_cheap_one():
    """A budget a step can never meet teaches an operator to ignore breaches."""
    assert route("draft", severity=5).max_latency_ms > route("triage").max_latency_ms


# --- the model layer measures and marks -----------------------------------

@pytest.fixture
def tracer(tmp_path, monkeypatch):
    t = Tracer(path=tmp_path / "traces.jsonl")
    monkeypatch.setattr(llm, "TRACER", t)
    return t


async def _run(monkeypatch, tracer, *, elapsed_ms: float, budget_ms: int):
    """One structured() call whose model takes exactly `elapsed_ms`."""
    clock = {"t": 0.0}
    monkeypatch.setattr(llm.time, "perf_counter", lambda: clock["t"])
    monkeypatch.setattr(llm, "_get_client", lambda: object())

    async def _generate(*_a, **_kw):
        clock["t"] += elapsed_ms / 1000
        return object()

    monkeypatch.setattr(llm, "_generate", _generate)
    monkeypatch.setattr(llm, "_parse", lambda *_a: _Answer(ok=True))

    return await llm.structured(
        model="test-model", system="s", user="u", schema=_Answer,
        stage="draft", max_latency_ms=budget_ms, timeout_s=30.0,
    )


@pytest.mark.asyncio
async def test_a_step_inside_its_budget_is_not_marked(monkeypatch, tracer):
    await _run(monkeypatch, tracer, elapsed_ms=900, budget_ms=5000)
    span = tracer.spans[-1]
    assert span.attrs["latency_budget_ms"] == 5000
    assert span.attrs["latency_ms"] == pytest.approx(900, abs=1)
    assert "over_budget" not in span.attrs


@pytest.mark.asyncio
async def test_a_step_over_its_budget_is_marked_but_still_returns(monkeypatch, tracer):
    result, _cost = await _run(monkeypatch, tracer, elapsed_ms=7000, budget_ms=5000)
    assert result.ok is True          # a budget is not a deadline
    span = tracer.spans[-1]
    assert span.attrs["over_budget"] is True
    assert span.attrs["latency_ms"] == pytest.approx(7000, abs=1)


@pytest.mark.asyncio
async def test_a_step_with_no_budget_is_measured_but_never_breaches(monkeypatch, tracer):
    """Some steps genuinely have no budget. They must still be timed, or they
    are invisible in the table that would show them getting slower."""
    await _run(monkeypatch, tracer, elapsed_ms=9000, budget_ms=None)
    span = tracer.spans[-1]
    assert span.attrs["latency_ms"] == pytest.approx(9000, abs=1)
    assert "over_budget" not in span.attrs


# --- stage_stats counts them ----------------------------------------------

def test_stage_stats_reports_the_budget_and_counts_breaches():
    traces = [
        {"name": "llm.draft", "ms": 900, "latency_budget_ms": 5000},
        {"name": "llm.draft", "ms": 7000, "latency_budget_ms": 5000, "over_budget": True},
        {"name": "llm.draft", "ms": 8000, "latency_budget_ms": 5000, "over_budget": True},
        {"name": "llm.triage", "ms": 200, "latency_budget_ms": 1500},
    ]
    rows = {r["stage"]: r for r in stage_stats(traces)}
    assert rows["llm.draft"]["budget_ms"] == 5000
    assert rows["llm.draft"]["over_budget"] == 2
    assert rows["llm.triage"]["over_budget"] == 0


def test_breaches_are_listed_slowest_first():
    """An aggregate hides the one case that took nine seconds, and that is
    usually the one worth reading."""
    traces = [
        {"name": "llm.draft", "ms": 7000, "latency_budget_ms": 5000, "over_budget": True},
        {"name": "llm.plan", "ms": 9000, "latency_budget_ms": 4000, "over_budget": True},
        {"name": "llm.triage", "ms": 200, "latency_budget_ms": 1500},
    ]
    breaches = budget_breaches(traces)
    assert [b["stage"] for b in breaches] == ["llm.plan", "llm.draft"]


def test_stage_stats_handles_spans_that_predate_budgets():
    """Old trace lines have no budget field. They must aggregate, not crash —
    the trace log is kept for 30 days and outlives a code change."""
    rows = stage_stats([{"name": "llm.draft", "ms": 500}])
    assert rows[0]["budget_ms"] is None and rows[0]["over_budget"] == 0
