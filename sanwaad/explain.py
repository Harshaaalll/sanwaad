"""Why a case is in the state it is in, in terms a reviewer can check.

Every piece here is computed by the same function the pipeline used to decide,
never re-described in prose: the priority parts come from `prioritise`, the
review rules from `review_checks`, the voice triggers from `escalation_checks`.
An explanation written separately from the decision is a second opinion that
drifts, and the first time it disagrees with what happened it is worse than
none.

One honest limit: earned autonomy changes as reviews accumulate, so the review
rules are evaluated *now*. The decision the gate made at the time is reported
next to them, and when the two differ the console says so.
"""

from __future__ import annotations

from .config import CRISIS, JUDGE, REVIEW
from .graph.nodes import escalation_checks, prioritise, review_checks


def _priority(state: dict) -> dict:
    triage = state.get("triage") or {}
    verdict = state.get("verdict") or {}
    pattern = state.get("pattern") or {}
    stored = state.get("priority") or {}
    components = stored.get("components")
    recomputed = False
    if not components and triage:
        # Cases from before the score was itemised: the same function on the
        # same inputs gives the same parts.
        components = prioritise(triage, verdict, pattern).components
        recomputed = True
    return {
        "tier": stored.get("tier"),
        "score": stored.get("score"),
        "reasons": stored.get("reasons") or [],
        "components": components or {},
        "recomputed": recomputed,
        "formula": "severity×10 + authenticity×8 + min(reach/1000, 20) "
                   "+ pattern (watch 15, crisis 40) + 5 if prior cases",
    }


def _author(state: dict) -> dict:
    verdict = state.get("verdict") or {}
    return {
        "class": verdict.get("author_class"),
        "authenticity": verdict.get("authenticity"),
        "reach": verdict.get("reach"),
        "evidence": verdict.get("evidence") or [],
        "reply_worthy": verdict.get("reply_worthy"),
        "bands": {
            "troll_at_or_below": JUDGE.troll_ceiling,
            "audience_at_or_above": JUDGE.audience_floor,
            "answer_if_authenticity_at_least": JUDGE.reply_worthy_authenticity,
            "or_reach_at_least": JUDGE.reply_worthy_reach,
        },
    }


def _pattern(state: dict) -> dict:
    pattern = state.get("pattern") or {}
    return {
        "level": pattern.get("level", "none"),
        "cluster_size": pattern.get("cluster_size"),
        "velocity_per_hour": pattern.get("velocity_per_hour"),
        "thresholds": {"watch_at": CRISIS.watch_cluster, "crisis_at": CRISIS.crisis_cluster,
                       "or_per_hour": CRISIS.crisis_velocity_per_hour,
                       "window_minutes": CRISIS.window_minutes},
    }


def _review(state: dict, pending: dict | None) -> dict | None:
    if not state.get("draft") or not state.get("triage"):
        return None      # closed before drafting: the gate never ran
    checks, allowed, reason = review_checks(state)
    review = state.get("review") or {}
    if pending and pending.get("await") != "voice_call":
        at_the_time = {"allowed": False, "reason": pending.get("reason")}
    elif review:
        at_the_time = {"allowed": bool(review.get("auto")),
                       "reason": review.get("note") if review.get("auto")
                       else f"reviewed by {review.get('reviewer')}"}
    else:
        at_the_time = None
    return {"checks": checks, "allowed_now": allowed, "reason_now": reason,
            "at_the_time": at_the_time,
            "ceiling": REVIEW.auto_post_max_severity}


def explain(state: dict, pending: dict | None = None) -> dict:
    out = {
        "priority": _priority(state),
        "author": _author(state),
        "pattern": _pattern(state),
        "review": _review(state, pending),
        "escalation": None,
    }
    if state.get("triage") and state.get("complaint"):
        out["escalation"] = {"checks": escalation_checks(state),
                             "recorded": state.get("escalation")}
    return out
