"""Verification: the hinge the whole loop turns on.

Some outputs are cheap and reliable to verify, and some are not, and that
asymmetry decides the shape of the loop:

- Where a CHEAP, DETERMINISTIC verifier exists, it goes INSIDE the loop. The
  agent drafts, the verifier checks, and a failure goes back to the agent as
  an observation it can act on. The agent corrects itself before anyone looks.
  Coding agents improved so fast because tests are exactly this kind of
  verifier.
- Where no cheap verifier exists — the judgement is subjective, novel, or the
  consequence is money or law — the loop does not guess. It stops with
  `needs_human`, and a person decides.

So verification is not a vibe check applied at the end. It is a design
decision made per step, and it is what tells you where automation ends and
review begins. `VERIFICATION_MAP` records that decision for every check in
Sanwaad, across all three loops: the agent's, the developer's and the world's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ..guardrails import check_reply
from .window import ContextWindow


class FinalAnswer(BaseModel):
    text: str
    cited_clauses: list[str] = Field(default_factory=list)


class VerifierCost(str, Enum):
    CHEAP = "cheap_deterministic"     # inside the loop, automatic retry
    MODEL = "model_judged"            # useful, imperfect, bounded retries
    HUMAN = "human_judgement"         # the loop stops and hands over


@dataclass(frozen=True)
class Verdict:
    verifier: str
    passed: bool
    feedback: str = ""


@dataclass(frozen=True)
class Verifier:
    name: str
    cost: VerifierCost
    catches: str
    check: Callable[[FinalAnswer, ContextWindow], Verdict]


_AMOUNT = re.compile(r"₹\s?([\d,]+(?:\.\d+)?)")
_REFERENCE = re.compile(r"\b(?:NP-TXN-[A-Z0-9-]+|REV-[A-F0-9]{8}|TKT-[A-F0-9]{8})\b")
_TIMELINE = re.compile(
    r"\b\d+\s*(?:working\s+|business\s+)?(?:hours?|hrs?|days?)\b|\bT\s*\+\s*\d\b", re.I)


def _amounts(text: str) -> set[float]:
    out = set()
    for raw in _AMOUNT.findall(text or ""):
        try:
            out.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Cheap, deterministic, in-loop verifiers
# ---------------------------------------------------------------------------

def check_guardrails(answer: FinalAnswer, window: ContextWindow) -> Verdict:
    blocking = [v for v in check_reply(answer.text).violations if v.severity == "block"]
    if blocking:
        return Verdict("reply_guardrails", False,
                       "Rewrite without: " + "; ".join(v.detail for v in blocking))
    return Verdict("reply_guardrails", True)


def check_facts_traced(answer: FinalAnswer, window: ContextWindow) -> Verdict:
    """Every amount and identifier in the reply must have been observed.

    The failure it exists for: the order lookup timed out, and the agent wrote a
    confident reply with a reference number anyway.
    """
    seen = window.all_seen_text()
    refs = [r for r in _REFERENCE.findall(answer.text) if r not in seen]
    seen_amounts = _amounts(seen)
    amounts = [a for a in _amounts(answer.text) if a not in seen_amounts]
    if refs or amounts:
        missing = refs + [f"₹{a:,.0f}" for a in amounts]
        return Verdict("facts_traced", False,
                       "Not in anything you observed — look it up or remove it: " + ", ".join(missing))
    return Verdict("facts_traced", True)


def check_timelines_cited(answer: FinalAnswer, window: ContextWindow) -> Verdict:
    if not _TIMELINE.search(answer.text):
        return Verdict("timelines_cited", True)
    if not any(c in window.clauses_seen for c in answer.cited_clauses):
        return Verdict("timelines_cited", False,
                       "You stated a timeline with no retrieved policy clause behind it. "
                       "Search the policy, then cite the clause that states it.")
    return Verdict("timelines_cited", True)


def check_citations_retrieved(answer: FinalAnswer, window: ContextWindow) -> Verdict:
    unknown = [c for c in answer.cited_clauses if c not in window.clauses_seen]
    if unknown:
        return Verdict("citations_retrieved", False,
                       f"You cited clauses you never retrieved: {', '.join(unknown)}")
    return Verdict("citations_retrieved", True)


IN_LOOP_VERIFIERS: list[Verifier] = [
    Verifier("reply_guardrails", VerifierCost.CHEAP,
             "money promises, banned phrasing, internal clause ids", check_guardrails),
    Verifier("facts_traced", VerifierCost.CHEAP,
             "amounts or references the agent never observed", check_facts_traced),
    Verifier("timelines_cited", VerifierCost.CHEAP,
             "a timeline stated without a policy clause", check_timelines_cited),
    Verifier("citations_retrieved", VerifierCost.CHEAP,
             "a clause cited that was never retrieved", check_citations_retrieved),
]


def run_verifiers(answer: FinalAnswer, window: ContextWindow,
                  verifiers: Optional[list[Verifier]] = None) -> list[Verdict]:
    return [v.check(answer, window) for v in (IN_LOOP_VERIFIERS if verifiers is None else verifiers)]


# ---------------------------------------------------------------------------
# Where verification is expensive, a person decides
# ---------------------------------------------------------------------------

_HUMAN_LANGUAGE = re.compile(
    r"\b(ombudsman|consumer\s*court|legal\s*notice|lawyer|police|fraud|scam|stolen)\b", re.I)


def needs_human(message: str, *, money_action_ready: bool) -> Optional[str]:
    """The judgements no cheap check can make. Returns why, or None."""
    if money_action_ready:
        return "a validated reversal is ready — moving money is a human decision"
    if _HUMAN_LANGUAGE.search(message or ""):
        return "regulatory or fraud language — no cheap verifier exists for this judgement"
    return None


# ---------------------------------------------------------------------------
# The verification map: every check, its cost, and which loop it lives in
# ---------------------------------------------------------------------------

VERIFICATION_MAP: list[dict] = [
    {"loop": "agent", "where": "support loop", "check": "reply_guardrails",
     "cost": VerifierCost.CHEAP.value, "on_fail": "agent rewrites from the feedback"},
    {"loop": "agent", "where": "support loop", "check": "facts_traced",
     "cost": VerifierCost.CHEAP.value, "on_fail": "agent looks it up or drops it"},
    {"loop": "agent", "where": "support loop", "check": "timelines_cited",
     "cost": VerifierCost.CHEAP.value, "on_fail": "agent searches policy, then cites"},
    {"loop": "agent", "where": "support loop", "check": "citations_retrieved",
     "cost": VerifierCost.CHEAP.value, "on_fail": "agent removes or retrieves the clause"},
    {"loop": "agent", "where": "support loop", "check": "needs_human",
     "cost": VerifierCost.HUMAN.value, "on_fail": "loop stops; a person takes over"},
    {"loop": "agent", "where": "case graph", "check": "ground_check",
     "cost": VerifierCost.MODEL.value, "on_fail": "redraft up to twice, then a person"},
    {"loop": "agent", "where": "case graph", "check": "validate_action",
     "cost": VerifierCost.CHEAP.value, "on_fail": "proposal blocked, never approvable"},
    {"loop": "agent", "where": "case graph", "check": "review_gate",
     "cost": VerifierCost.HUMAN.value, "on_fail": "approve, edit or reject"},
    {"loop": "developer", "where": "evals/", "check": "trajectory and loop evals",
     "cost": VerifierCost.CHEAP.value, "on_fail": "you fix the spec, prompt or tool"},
    {"loop": "external", "where": "loop/outer.py", "check": "recorded runs, reviewer edits, A/B",
     "cost": VerifierCost.HUMAN.value, "on_fail": "new regression scenarios, a changed spec"},
]
