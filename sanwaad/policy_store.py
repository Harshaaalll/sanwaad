"""Changing a policy threshold at runtime: bounded, reasoned, logged, reversible.

The policies in config.py decide what goes out without a person, so a change
to one is a change to the product's safety, and it is treated that way:

- Only the fields listed in EDITABLE can change, each inside hard bounds.
  Some rules are deliberately absent: turning off the grounding requirement,
  or letting a draft promise compensation on its own, is a code change with a
  review, never a form. Where the code default is the safe limit (the reversal
  ceiling), the bound stops at it.
- Every change carries a reason and a name, and goes to an append-only log.
  The log is the source of truth: on start the server replays it, so changes
  survive a restart, and a revert is one more entry rather than an erasure.
- The policies are frozen dataclasses imported by name all over the codebase,
  so replacing the objects would leave every importer holding the old one.
  They are changed in place, and only here, through `_set`.

Previews answer "what would this have changed?" by re-running the deciding
function with a candidate copy of the policy on stored cases. They are offered
only where stored state reproduces the decision exactly; crisis thresholds
depend on when earlier complaints arrived, which a case does not store, so
those say so instead of guessing.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from . import config
from .config import DATA_DIR

AUDIT_PATH = DATA_DIR / "policy_audit.jsonl"

POLICIES = {"review": config.REVIEW, "judge": config.JUDGE,
            "crisis": config.CRISIS, "actions": config.ACTIONS}


class PolicyError(ValueError):
    """A change that is not allowed, and the message says why."""


@dataclass(frozen=True)
class Bound:
    kind: type
    low: float
    high: float
    note: str = ""


def _autonomy_max() -> int:
    from .graph.nodes import AUTONOMY_MAX_SEVERITY

    return AUTONOMY_MAX_SEVERITY


# What may change, and within what. Anything not listed is not editable.
EDITABLE: dict[tuple[str, str], Bound] = {
    ("review", "auto_post_max_severity"): Bound(int, 0, 3, "never above the autonomy maximum"),
    ("review", "escalate_to_voice_min_severity"): Bound(int, 2, 5),
    ("judge", "reply_worthy_authenticity"): Bound(float, 0.1, 0.9),
    ("judge", "reply_worthy_reach"): Bound(int, 500, 1_000_000),
    ("judge", "troll_ceiling"): Bound(float, 0.1, 0.5, "must stay below the audience floor"),
    ("judge", "audience_floor"): Bound(float, 0.4, 0.9, "must stay above the troll ceiling"),
    ("crisis", "watch_cluster"): Bound(int, 2, 20, "must stay below the crisis cluster"),
    ("crisis", "crisis_cluster"): Bound(int, 3, 50, "must stay above the watch cluster"),
    ("crisis", "crisis_velocity_per_hour"): Bound(float, 1.0, 100.0),
    ("crisis", "similarity"): Bound(float, 0.5, 0.95),
    ("crisis", "window_minutes"): Bound(int, 15, 1440),
    ("actions", "reversal_ceiling_inr"): Bound(float, 0.0, 25_000.0,
                                               "never above the code default"),
    ("actions", "ticket_min_severity"): Bound(int, 1, 5),
}

# The values the code ships with, captured before any override is replayed.
DEFAULTS = {name: dataclasses.asdict(obj) for name, obj in POLICIES.items()}

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce(bound: Bound, raw: Any) -> Any:
    try:
        value = bound.kind(raw)
        if bound.kind is int and float(raw) != int(float(raw)):
            raise ValueError
    except (TypeError, ValueError):
        raise PolicyError(f"expected {'a whole number' if bound.kind is int else 'a number'}, "
                          f"got {raw!r}") from None
    if not bound.low <= value <= bound.high:
        raise PolicyError(f"{value} is outside the allowed range {bound.low:g}–{bound.high:g}")
    return value


def _check_consistency(policy: str, field: str, value: Any) -> None:
    """Rules that span two fields, or another module."""
    current = dataclasses.asdict(POLICIES[policy]) | {field: value}
    if policy == "review" and field == "auto_post_max_severity" and value > _autonomy_max():
        raise PolicyError(f"cannot exceed the autonomy maximum of {_autonomy_max()}")
    if policy == "judge" and current["troll_ceiling"] >= current["audience_floor"]:
        raise PolicyError("the troll ceiling must stay below the audience floor")
    if policy == "crisis" and current["watch_cluster"] >= current["crisis_cluster"]:
        raise PolicyError("a watch must fire before a crisis: watch cluster < crisis cluster")


def validate(policy: str, field: str, raw: Any) -> Any:
    bound = EDITABLE.get((policy, field))
    if bound is None:
        raise PolicyError(f"{policy}.{field} is not editable here; it changes through code review")
    value = _coerce(bound, raw)
    _check_consistency(policy, field, value)
    return value


def _set(policy: str, field: str, value: Any) -> None:
    # The only place a frozen policy is written. See the module docstring.
    object.__setattr__(POLICIES[policy], field, value)


def _append(entry: dict) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def history() -> list[dict]:
    """Every change ever made, oldest first."""
    if not AUDIT_PATH.exists():
        return []
    out = []
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue       # a torn last line must not hide every change before it
    return out


def apply(policy: str, field: str, raw: Any, *, reason: str, actor: str,
          reverts: Optional[str] = None) -> dict:
    reason, actor = (reason or "").strip(), (actor or "").strip()
    if len(reason) < 5:
        raise PolicyError("a reason is required (at least a few words)")
    if not actor:
        raise PolicyError("say who is making the change")
    with _LOCK:
        value = validate(policy, field, raw)
        old = getattr(POLICIES[policy], field)
        if old == value:
            raise PolicyError(f"{policy}.{field} is already {value}")
        entry = {"id": uuid.uuid4().hex[:12], "at": _now(), "policy": policy, "field": field,
                 "old": old, "new": value, "reason": reason, "actor": actor}
        if reverts:
            entry["reverts"] = reverts
        _append(entry)          # logged before it takes effect: no unrecorded change
        _set(policy, field, value)
    return entry


def revert(change_id: str, *, reason: str, actor: str) -> dict:
    change = next((c for c in history() if c.get("id") == change_id), None)
    if change is None:
        raise PolicyError(f"no change {change_id}")
    current = getattr(POLICIES[change["policy"]], change["field"])
    if current != change["new"]:
        raise PolicyError(f"{change['policy']}.{change['field']} has changed again since "
                          f"(now {current}); revert the later change first")
    return apply(change["policy"], change["field"], change["old"],
                 reason=reason, actor=actor, reverts=change_id)


def replay() -> list[str]:
    """Re-apply the log on start. Returns problems, never raises: a bad entry
    is skipped and reported, and the server still comes up on safe defaults
    for that field."""
    problems = []
    latest: dict[tuple[str, str], Any] = {}
    for c in history():
        latest[(c.get("policy"), c.get("field"))] = c.get("new")
    with _LOCK:
        for (policy, field), value in latest.items():
            try:
                _set(policy, field, validate(policy, field, value))
            except (PolicyError, KeyError) as exc:
                problems.append(f"{policy}.{field}: {exc}")
    return problems


def describe() -> list[dict]:
    """Every editable field: its value now, the shipped default, and its bounds."""
    rows = []
    for (policy, field), b in EDITABLE.items():
        rows.append({"policy": policy, "field": field,
                     "value": getattr(POLICIES[policy], field),
                     "default": DEFAULTS[policy][field],
                     "low": b.low, "high": b.high, "kind": b.kind.__name__, "note": b.note,
                     "previewable": (policy, field) in _PREVIEWS})
    return rows


# ---------------------------------------------------------------------------
# Preview: what a candidate value would have changed, on stored cases
# ---------------------------------------------------------------------------

def _review_outcome(state: dict, review) -> Optional[str]:
    from .graph.nodes import review_checks

    if not state.get("draft") or not state.get("triage"):
        return None
    _, allowed, _ = review_checks(state, review=review)
    return "posts on its own" if allowed else "held for a person"


def _voice_outcome(state: dict, review) -> Optional[str]:
    from .graph.nodes import escalation_checks

    if not state.get("triage") or not state.get("complaint"):
        return None
    fired = any(c["fired"] for c in escalation_checks(state, review=review))
    return "call offered" if fired else "no call"


def _tier_outcome(state: dict, judge) -> Optional[str]:
    from .graph.nodes import prioritise

    triage, verdict = state.get("triage"), dict(state.get("verdict") or {})
    if not triage or not verdict:
        return None
    # reply_worthy was computed at judge time from these same two thresholds;
    # recompute it from the stored authenticity and reach.
    verdict["reply_worthy"] = (
        verdict.get("author_class") != "bot"
        and (float(verdict.get("authenticity", 0)) >= judge.reply_worthy_authenticity
             or int(verdict.get("reach", 0)) >= judge.reply_worthy_reach))
    return prioritise(triage, verdict, state.get("pattern") or {}, judge=judge).tier


_PREVIEWS = {
    ("review", "auto_post_max_severity"): _review_outcome,
    ("review", "escalate_to_voice_min_severity"): _voice_outcome,
    ("judge", "reply_worthy_authenticity"): _tier_outcome,
    ("judge", "reply_worthy_reach"): _tier_outcome,
}


def preview(policy: str, field: str, raw: Any, states: list[dict]) -> dict:
    """Decisions that would differ under the candidate value, on these cases.

    Uses a copy of the policy; the live one, which running cases read, is
    never touched.
    """
    value = validate(policy, field, raw)
    fn = _PREVIEWS.get((policy, field))
    if fn is None:
        return {"available": False,
                "why": "this threshold depends on data a stored case does not keep "
                       "(for crisis: when earlier complaints arrived), so its effect "
                       "cannot be replayed exactly"}
    live = POLICIES[policy]
    candidate = dataclasses.replace(live, **{field: value})
    changed, checked = [], 0
    for state in states:
        before, after = fn(state, live), fn(state, candidate)
        if before is None:
            continue
        checked += 1
        if before != after:
            changed.append({"case_id": state.get("case_id"), "before": before, "after": after,
                            "category": (state.get("triage") or {}).get("category"),
                            "severity": (state.get("triage") or {}).get("severity")})
    return {"available": True, "checked": checked, "changed": len(changed),
            "examples": changed[:20], "value": value,
            "note": "evaluated now; earned autonomy may differ from when each case was decided"}
