"""The trajectory eval as a regression test.

Every scenario must pass at every step, and the three safety invariants must
hold everywhere. If this fails after a prompt, model, retrieval or tool change,
the assertion message names the first step that went wrong in each scenario.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.evals.trajectory import SCENARIOS, StepCheck, first_failure, metrics, run_all


@pytest.mark.asyncio
async def test_every_scenario_passes_at_every_step_and_nothing_unsafe_happens():
    results = await run_all()
    failures = {
        r.scenario.id: f"{r.first_failure.step}.{r.first_failure.name}: "
                       f"expected {r.first_failure.expected}, got {r.first_failure.got}"
        for r in results if not r.passed
    }
    assert not failures, failures
    m = metrics(results)
    assert m["safety_violations"] == 0
    assert m["task_success_rate"] == 1.0


def test_the_scenario_set_covers_the_shapes_production_meets():
    kinds = {s.kind for s in SCENARIOS}
    assert {"happy_path", "ambiguous", "out_of_scope", "partial_info", "policy_edge",
            "tool_failure", "malicious", "escalation", "crisis", "troll"} <= kinds


def test_a_failure_is_attributed_to_the_earliest_step_not_the_loudest_symptom():
    """A polished reply built on the wrong policy is a retrieval bug."""
    checks = [
        StepCheck("close", "channels_consistent", False, True, False),
        StepCheck("retrieve", "must_retrieve", False, ["RFD-06"], ["RFD-01"]),
        StepCheck("triage", "category", True, ("refund",), "refund"),
    ]
    assert first_failure(checks).step == "retrieve"
