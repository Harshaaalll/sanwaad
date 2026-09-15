"""Model router: pick the cheapest model that can do this particular job.

The naive setup uses one model everywhere and is either needlessly expensive
on easy work or unreliable on hard work. A router turns "which model" into a
per-request decision driven by signals you already have.

Routing on *observable request features* (severity, language, whether a prior
attempt failed) rather than asking a model to judge difficulty first. A
classifier-based router costs an extra call and an extra failure mode to save
one; it only pays once the spread between tiers is large and the features are
genuinely uninformative. Here they are highly informative — triage has already
computed severity — so the routing is a function, and a function can be unit
tested, which a judgement call cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .config import TIER_DRAFT, TIER_REASONING, TIER_TRIAGE

Stage = Literal["triage", "judge", "draft", "ground_check", "plan", "voice"]


@dataclass(frozen=True)
class Route:
    model: str
    reason: str
    max_latency_ms: Optional[int] = None   # None = latency does not matter here
    # Where to go when `model` times out, errors, or keeps failing its schema.
    # Usually one tier up (a harder prompt is a reason to try a stronger
    # model); for the reasoning tier, one tier down (the likelier failure
    # there is availability, and a working answer beats no answer).
    fallback: Optional[str] = None
    # Output tokens are cost AND latency. Every stage returns a small JSON
    # object; a cap turns a runaway generation into a schema failure the LLM
    # layer already knows how to handle.
    max_output_tokens: Optional[int] = None
    timeout_s: float = 20.0


# The voice path is latency-bound and must never be routed to a reasoning
# model, whatever it would save. Everything else is throughput-bound.
VOICE_MODEL = "gemini-2.5-flash"
VOICE_TTFT_BUDGET_MS = 500


def route(stage: Stage, *, severity: int = 1, revision: int = 0,
          injection_flagged: bool = False, language: str = "en-IN") -> Route:
    if stage == "voice":
        # No fallback: on a live call, a second model attempt blows the latency
        # budget the caller is already waiting inside.
        return Route(VOICE_MODEL, "voice is latency-bound; no reasoning tier",
                     max_latency_ms=VOICE_TTFT_BUDGET_MS, max_output_tokens=150,
                     timeout_s=2.0)

    if stage == "triage":
        # Runs on every inbound item, so it is the single biggest lever on
        # spend. Nothing about triage justifies a larger model — it is a
        # short classification with a fixed label set.
        return Route(TIER_TRIAGE, "classification over a fixed label set",
                     fallback=TIER_DRAFT, max_output_tokens=300, timeout_s=8.0)

    if stage == "judge":
        return Route(TIER_TRIAGE, "ambiguous-band author read: a short classification",
                     fallback=TIER_DRAFT, max_output_tokens=200, timeout_s=8.0)

    if stage == "ground_check":
        return Route(TIER_DRAFT, "verification needs the drafting tier's reading ability",
                     fallback=TIER_REASONING, max_output_tokens=400, timeout_s=15.0)

    if stage == "plan":
        # Deliberately NOT the reasoning tier, even though this step proposes
        # moving money. Every proposal is validated by code against the ledger
        # before a human sees it, so a wrong proposal is caught, not executed.
        # Spending more on the model would buy accuracy the validator already
        # guarantees.
        return Route(TIER_DRAFT, "proposals are validated by code before anyone approves them",
                     fallback=TIER_REASONING, max_output_tokens=400, timeout_s=15.0)

    # draft
    strong = dict(fallback=TIER_DRAFT, max_output_tokens=400, timeout_s=25.0)
    if revision > 0:
        return Route(TIER_REASONING,
                     f"revision {revision}: the cheap tier already failed grounding once", **strong)
    if injection_flagged:
        return Route(TIER_REASONING,
                     "adversarial input: use the stronger model rather than the cheap one", **strong)
    if severity >= 4:
        return Route(TIER_REASONING, f"severity {severity}: high-stakes reply", **strong)
    return Route(TIER_DRAFT, f"severity {severity}: routine reply",
                 fallback=TIER_REASONING, max_output_tokens=400, timeout_s=20.0)


def explain() -> list[dict]:
    """The full routing table, for documentation and for the console."""
    cases = [
        ("triage",       dict()),
        ("judge",        dict()),
        ("draft",        dict(severity=2)),
        ("draft",        dict(severity=4)),
        ("draft",        dict(severity=2, revision=1)),
        ("draft",        dict(severity=1, injection_flagged=True)),
        ("ground_check", dict()),
        ("plan",         dict()),
        ("voice",        dict()),
    ]
    out = []
    for stage, kw in cases:
        r = route(stage, **kw)
        out.append({"stage": stage, "when": kw or {"always": True},
                    "model": r.model, "fallback": r.fallback,
                    "max_output_tokens": r.max_output_tokens, "timeout_s": r.timeout_s,
                    "reason": r.reason})
    return out


if __name__ == "__main__":
    for row in explain():
        when = ", ".join(f"{k}={v}" for k, v in row["when"].items())
        print(f"{row['stage']:<13}{when:<36}{row['model']:<24}"
              f"fallback {str(row['fallback']):<24}{row['max_output_tokens']!s:>4} tok  "
              f"{row['timeout_s']:>4}s  {row['reason']}")
