"""The loops around the loop.

Building software with agents is three nested loops, running at different
speeds and run by different people:

    agent loop       seconds to minutes   run by the agent    act, observe, verify, retry  (kernel.py)
    developer loop   minutes to hours     run by you          evals, the MINT ladder, a changed spec
    external loop    hours to weeks       run by the world    real conversations, reviewers, A/B tests

The further out a loop sits, the more its verification depends on human
judgement instead of automated checks. The agent can verify that a reply cites
the clause it retrieved. Only real customers can verify that the reply actually
helped. Humans do not leave the system as the inner loop gets better; they move
to the loops where they hold the context and the judgement.

This module is the machinery of the external loop, and its hand-off to the
developer loop:

- `trace_report` evaluates what REALLY ran — the recorded support-loop runs —
  as a system: how often loops stop cleanly, why they stop, what they cost.
  Scenarios test what someone thought of; traces test what happened.
- `regression_candidates` turns real runs that went wrong into draft eval
  scenarios. A person reviews them before they join the suite, because a
  failure copied blindly into a golden set becomes a wrong expectation.
- `assign_variant` and `compare_variants` split live traffic deterministically
  and compare outcomes, and refuse to declare a winner on too few runs.

    python -m sanwaad.loop.outer
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .budget import CLEAN_STOPS

MIN_RUNS_PER_VARIANT = 30

NESTED_LOOPS: list[dict] = [
    {"loop": "agent", "speed": "seconds to minutes", "run_by": "the agent",
     "verifies_with": "cheap in-loop verifiers", "in_sanwaad": "sanwaad/loop/kernel.py"},
    {"loop": "developer", "speed": "minutes to hours", "run_by": "you",
     "verifies_with": "trajectory eval, loop eval, MINT ladder, code review",
     "in_sanwaad": "sanwaad/evals/"},
    {"loop": "external", "speed": "hours to weeks", "run_by": "customers and reviewers",
     "verifies_with": "human judgement: outcomes, edits, A/B comparisons",
     "in_sanwaad": "sanwaad/loop/outer.py, sanwaad/feedback.py"},
]


# ---------------------------------------------------------------------------
# Reading what happened
# ---------------------------------------------------------------------------

def load_runs(path: Optional[Path] = None, since: Optional[datetime] = None) -> list[dict]:
    from . import kernel

    path = path or kernel.LOOP_RUNS_PATH
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            run = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is not None:
            try:
                if datetime.fromisoformat(run["at"]) < since:
                    continue
            except (KeyError, ValueError):
                continue
        runs.append(run)
    return runs


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


def _clean(run: dict) -> bool:
    return run.get("stop_reason") in {s.value for s in CLEAN_STOPS}


def trace_report(runs: Iterable[dict]) -> dict:
    """System-level metrics over recorded runs."""
    runs = list(runs)
    n = len(runs)
    if not n:
        return {"runs": 0}
    passes = [r.get("meter", {}).get("iterations", 0) for r in runs]
    costs = [r.get("meter", {}).get("cost_inr", 0.0) for r in runs]
    tool_steps = [s for r in runs for s in r.get("steps", []) if s.get("action") not in
                  ("final", "handoff", "propose_reversal", "consult_policy")]
    tool_errors = [s for s in tool_steps if str(s.get("outcome", "")).startswith("tool_error")]
    caught = [r for r in runs if r.get("stop_reason") == "done"
              and any(not v.get("passed", True) for v in r.get("verdicts", []))]
    over = [r for r in runs if r.get("window", {}).get("peak_tokens", 0)
            > r.get("config", {}).get("budget", {}).get("context_tokens", 10 ** 9)]
    return {
        "runs": n,
        "clean_stop_rate": round(sum(map(_clean, runs)) / n, 3),
        "stop_reasons": dict(Counter(r.get("stop_reason") for r in runs)),
        "needs_human_rate": round(sum(r.get("stop_reason") == "needs_human" for r in runs) / n, 3),
        "mean_passes": round(sum(passes) / n, 2),
        "p95_passes": _percentile(passes, 0.95),
        "mean_cost_inr": round(sum(costs) / n, 5),
        "p95_cost_inr": round(_percentile(costs, 0.95), 5),
        "tool_error_rate": round(len(tool_errors) / len(tool_steps), 3) if tool_steps else 0.0,
        "verifier_catch_rate": round(len(caught) / n, 3),
        "runs_over_context_budget": len(over),
        "by_workflow": dict(Counter(r.get("workflow") or "none" for r in runs)),
    }


# ---------------------------------------------------------------------------
# External loop → developer loop
# ---------------------------------------------------------------------------

def regression_candidates(runs: Iterable[dict]) -> list[dict]:
    """Real runs worth turning into eval scenarios, drafted for human review.

    Picked: loops that stopped badly, loops a verifier had to correct, and loops
    that met a tool error. The suggested expectation is a starting point only —
    a person decides what the right outcome was.
    """
    out = []
    for run in runs:
        stop = run.get("stop_reason")
        corrected = any(not v.get("passed", True) for v in run.get("verdicts", []))
        tool_error = any(str(s.get("outcome", "")).startswith("tool_error") for s in run.get("steps", []))
        if _clean(run) and not corrected and not tool_error:
            continue
        reason = ("stopped: " + stop) if not _clean(run) else (
            "a verifier corrected it" if corrected else "met a tool error")
        out.append({
            "id": f"prod-{run.get('run_id', '')[-6:]}",
            "source_run": run.get("run_id"),
            "handle": run.get("handle"),
            "message": run.get("message"),
            "observed_stop": stop,
            "observed_passes": run.get("meter", {}).get("iterations"),
            "why_selected": reason,
            "suggested_expect": {"stop_in": ["done", "needs_human"]},
            "review_required": True,
        })
    return out


# ---------------------------------------------------------------------------
# A/B: split traffic, compare outcomes, don't call it early
# ---------------------------------------------------------------------------

def assign_variant(key: str, experiment: str, variants: tuple[str, ...] = ("control", "treatment"),
                   weights: Optional[tuple[float, ...]] = None) -> str:
    """Deterministic assignment: the same customer always lands in the same arm
    of the same experiment, across restarts and machines, with nothing stored."""
    weights = weights or tuple(1.0 / len(variants) for _ in variants)
    point = int(hashlib.sha256(f"{experiment}\x00{key}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for variant, weight in zip(variants, weights):
        cumulative += weight
        if point <= cumulative:
            return variant
    return variants[-1]


def _two_proportion_p(successes_a: int, n_a: int, successes_b: int, n_b: int) -> Optional[float]:
    if not n_a or not n_b:
        return None
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = abs(successes_a / n_a - successes_b / n_b) / se
    return round(math.erfc(z / math.sqrt(2)), 4)


def compare_variants(runs: Iterable[dict], control: str = "control",
                     treatment: str = "treatment", min_runs: int = MIN_RUNS_PER_VARIANT) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        if run.get("variant"):
            groups[run["variant"]].append(run)
    summary = {name: {**trace_report(rs), "clean": sum(map(_clean, rs))} for name, rs in groups.items()}
    a, b = summary.get(control), summary.get(treatment)
    if not a or not b:
        return {"variants": summary, "verdict": "both arms need runs"}
    if a["runs"] < min_runs or b["runs"] < min_runs:
        return {"variants": summary,
                "verdict": f"not enough runs to call it (need {min_runs} per arm, have "
                           f"{a['runs']} and {b['runs']})"}
    p = _two_proportion_p(b["clean"], b["runs"], a["clean"], a["runs"])
    delta = round(b["clean_stop_rate"] - a["clean_stop_rate"], 3)
    verdict = (f"{treatment} {'better' if delta > 0 else 'worse'} on clean stops by {abs(delta):.3f} (p={p})"
               if p is not None and p < 0.05 else f"no clear difference in clean stops (p={p})")
    return {"variants": summary, "clean_stop_delta": delta, "p_value": p,
            "cost_delta_inr": round(b["mean_cost_inr"] - a["mean_cost_inr"], 5), "verdict": verdict}


def main() -> int:
    runs = load_runs()
    print("\nthe three loops")
    for row in NESTED_LOOPS:
        print(f"  {row['loop']:<10}{row['speed']:<22}{row['run_by']:<26}{row['verifies_with']}")
    print(f"\nrecorded runs: {len(runs)}")
    for key, value in trace_report(runs).items():
        print(f"  {key:<26}{value}")
    candidates = regression_candidates(runs)
    print(f"\nregression candidates for review: {len(candidates)}")
    for c in candidates[:8]:
        print(f"  {c['id']:<14}{c['why_selected']:<28}{(c['message'] or '')[:60]}")
    print(f"\nA/B: {compare_variants(runs)['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
