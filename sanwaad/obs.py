"""Observability: spans, costs and latency, recorded where they happen.

Three things get confused under this word, and only the third is hard:

  logging  — "what happened", one line at a time, no structure
  metrics  — "how much / how often", aggregated, cheap, no detail
  tracing  — "what happened *within one request*", nested and causal

An LLM pipeline needs tracing specifically, because the interesting failures
are relational: retrieval took 8ms and returned the wrong clause, so drafting
was fine but wrong, so the grounding check passed something that should not
have shipped. A log line per stage cannot express "so".

This is a deliberately small tracer — a few hundred bytes per span, written to
JSONL — rather than an SDK. It carries what a trace must carry: a shared
trace id, parent/child nesting, wall time, and the cost attributed to the span
that incurred it. Swapping in OpenTelemetry later is a matter of changing
`Span.finish`; every call site stays as written.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import DATA_DIR

TRACE_PATH = DATA_DIR / "traces.jsonl"

_current: ContextVar[Optional["Span"]] = ContextVar("current_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    started: float
    ended: Optional[float] = None
    attrs: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    # Wall-clock start. perf_counter is right for durations and useless for
    # "delete spans older than 30 days", which is a retention requirement.
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def ms(self) -> float:
        return ((self.ended or time.perf_counter()) - self.started) * 1000

    def set(self, **kw: Any) -> None:
        """Attach attributes. Called during the span, not after — an attribute
        recorded at the point it is known cannot drift from what happened."""
        self.attrs.update(kw)

    def to_dict(self) -> dict:
        return {
            "at": self.at,
            "trace_id": self.trace_id, "span_id": self.span_id,
            "parent_id": self.parent_id, "name": self.name,
            "ms": round(self.ms, 2), "error": self.error, **self.attrs,
        }


# How many finished spans stay in memory. The JSONL on disk is the record;
# this list only serves in-process readers like `summary()` and the evals. The
# API server runs cases for the lifetime of the process, and an unbounded list
# is a slow leak in the one component meant to stay up for weeks.
MAX_SPANS_IN_MEMORY = 5_000


class Tracer:
    def __init__(self, path: Path = TRACE_PATH, enabled: bool = True,
                 max_spans: int = MAX_SPANS_IN_MEMORY):
        self.path = path
        self.enabled = enabled
        self.max_spans = max_spans
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, trace_id: Optional[str] = None, **attrs: Any) -> Iterator[Span]:
        parent = _current.get()
        s = Span(
            name=name,
            trace_id=trace_id or (parent.trace_id if parent else uuid.uuid4().hex[:12]),
            span_id=uuid.uuid4().hex[:8],
            parent_id=parent.span_id if parent else None,
            started=time.perf_counter(),
            attrs=dict(attrs),
        )
        token = _current.set(s)
        try:
            yield s
        except Exception as exc:
            # Record the failure on the span before re-raising: a trace that
            # only contains successes is the one you cannot debug with.
            s.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.ended = time.perf_counter()
            _current.reset(token)
            self.spans.append(s)
            if len(self.spans) > self.max_spans:
                # Drop the oldest. A trace is read while it is recent, and
                # anything older is on disk anyway.
                del self.spans[:len(self.spans) - self.max_spans]
            if self.enabled:
                self._write(s)

    def _write(self, s: Span) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass  # never let telemetry break the request it is measuring

    # --- reading back -----------------------------------------------------

    def summary(self, trace_id: Optional[str] = None) -> dict:
        spans = [s for s in self.spans if not trace_id or s.trace_id == trace_id]
        if not spans:
            return {}
        by_name: dict[str, list[float]] = {}
        cost = 0.0
        for s in spans:
            by_name.setdefault(s.name, []).append(s.ms)
            cost += float(s.attrs.get("cost_inr", 0) or 0)
        root = min(spans, key=lambda s: s.started)
        return {
            "trace_id": root.trace_id,
            "total_ms": round(root.ms, 2),
            "cost_inr": round(cost, 4),
            "spans": len(spans),
            "errors": [s.name for s in spans if s.error],
            "by_stage": {k: round(sum(v), 2) for k, v in sorted(
                by_name.items(), key=lambda kv: -sum(kv[1]))},
        }


TRACER = Tracer()


def current_trace_id() -> Optional[str]:
    s = _current.get()
    return s.trace_id if s else None


def load_traces(path: Optional[Path] = None) -> list[dict]:
    # Resolved at call time, not bound as a default: a default argument freezes
    # TRACE_PATH at import and silently ignores anyone who redirects it.
    path = path or TRACE_PATH
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def stage_stats(traces: list[dict]) -> list[dict]:
    """Aggregate spans into a p50/p95 table, each stage against its budget.

    p95 rather than mean because the tail is what a caller waiting on the line
    actually experiences. The budget column is what turns a number into a
    verdict: `router.py` declares what each step is allowed to take, the model
    layer stamps that budget onto the span, and the comparison happens here. A
    budget nothing ever checks is a comment.
    """
    import statistics

    by: dict[str, list[float]] = {}
    cost: dict[str, float] = {}
    errs: dict[str, int] = {}
    breaches: dict[str, int] = {}
    budgets: dict[str, int] = {}
    for t in traces:
        name = t["name"]
        by.setdefault(name, []).append(t.get("ms", 0.0))
        cost[name] = cost.get(name, 0.0) + float(t.get("cost_inr", 0) or 0)
        if t.get("error"):
            errs[name] = errs.get(name, 0) + 1
        if t.get("over_budget"):
            breaches[name] = breaches.get(name, 0) + 1
        if t.get("latency_budget_ms"):
            # The widest budget seen for this stage, not the last one written.
            # `llm.draft` carries 8000ms when it routes to the reasoning tier
            # and 5000ms otherwise, under one span name, so last-wins reported
            # breaches against whichever trace happened to come last.
            budgets[name] = max(budgets.get(name, 0), int(t["latency_budget_ms"]))

    rows = []
    for name, ms in by.items():
        ms_sorted = sorted(ms)
        p95 = ms_sorted[min(len(ms_sorted) - 1, int(len(ms_sorted) * 0.95))]
        rows.append({
            "stage": name, "n": len(ms),
            "p50_ms": round(statistics.median(ms), 1),
            "p95_ms": round(p95, 1),
            "budget_ms": budgets.get(name),
            "over_budget": breaches.get(name, 0),
            "cost_inr": round(cost.get(name, 0.0), 4),
            "errors": errs.get(name, 0),
        })
    return sorted(rows, key=lambda r: -r["p95_ms"])


def budget_breaches(traces: list[dict]) -> list[dict]:
    """Every span that took longer than its step was budgeted.

    Kept separate from the table because an aggregate hides the single case
    that took nine seconds, and that one is usually the one worth reading.
    """
    return sorted(
        ({"stage": t["name"], "trace_id": t.get("trace_id"), "at": t.get("at"),
          "ms": t.get("ms"), "budget_ms": t.get("latency_budget_ms"),
          "model": t.get("model"), "attempts": t.get("attempts")}
         for t in traces if t.get("over_budget")),
        key=lambda r: -(r["ms"] or 0),
    )


def main(argv: list[str]) -> int:
    """`python -m sanwaad.obs` — what each step took, cost, and was allowed."""
    traces = load_traces()
    if not traces:
        print(f"\nNo traces yet at {TRACE_PATH}. Run `python -m sanwaad.demo` first.\n")
        return 0

    print(f"\n{'stage':<24}{'n':<6}{'p50 ms':<10}{'p95 ms':<10}{'budget':<10}"
          f"{'over':<7}{'errors':<8}cost ₹")
    print("-" * 92)
    for r in stage_stats(traces):
        budget = str(r["budget_ms"]) if r["budget_ms"] else "—"
        over = str(r["over_budget"]) if r["over_budget"] else ""
        errors = str(r["errors"]) if r["errors"] else ""
        print(f"{r['stage']:<24}{r['n']:<6}{r['p50_ms']:<10}{r['p95_ms']:<10}{budget:<10}"
              f"{over:<7}{errors:<8}{r['cost_inr']:.4f}")

    breaches = budget_breaches(traces)
    if breaches:
        print(f"\n{len(breaches)} step(s) over budget, slowest first:")
        for b in breaches[:10]:
            print(f"  {b['stage']:<22}{b['ms']:>8.0f}ms  against {b['budget_ms']}ms"
                  f"  {b['model'] or ''}  case {b['trace_id']}")
    elif any(t.get("latency_budget_ms") for t in traces):
        print("\nNo step went over its latency budget.")
    else:
        # Rather than print a column of dashes and let someone conclude the
        # budgets are not wired up: they are, and nothing here was measured
        # against one because nothing here called a model.
        print("\nNo budgets in this log. Latency budgets are declared per model"
              "\nstep in router.py, and offline runs never call a model —"
              "\nadd GOOGLE_API_KEY to .env and run the demo again.")
    print()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
