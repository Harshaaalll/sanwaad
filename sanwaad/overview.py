"""The operator's view across cases: what is waiting, what is on fire, what it costs.

A case list shows one case at a time and only the ones that worked. This is
the other view: counts a team lead would put on a wall, computed from the same
case state the list shows, so a number here can always be traced to cases.
"""

from __future__ import annotations

from collections import Counter

from .config import HUMAN_COST_INR


def status_of(case: dict) -> str:
    """Where a case is, from the fields list_cases returns."""
    if case.get("closure"):
        return "resolved" if case["closure"].get("resolved") else "closed_unresolved"
    if case.get("draft") and not case.get("review"):
        return "needs_review"
    if case.get("review") and (case.get("escalation") or {}).get("needed"):
        return "awaiting_call"
    return "in_progress"


def summarise(cases: list[dict], *, dead_letters: int = 0,
              autonomy: list[dict] | None = None) -> dict:
    statuses = Counter(status_of(c) for c in cases)
    triages = [c.get("triage") or {} for c in cases]
    reviews = [c["review"] for c in cases if c.get("review")]
    auto = sum(1 for r in reviews if r.get("auto"))
    human = [r for r in reviews if not r.get("auto")]
    decisions = Counter(r.get("decision") for r in human)
    closed = [c for c in cases if c.get("closure")]
    closed_costs = [float(c["closure"].get("total_cost_inr", 0.0)) for c in closed]
    degraded = sum(1 for c in closed if c["closure"].get("degraded_steps"))

    # Incidents: every case the pattern agent put in a watch or crisis cluster,
    # grouped by the theme it named, so ten reports read as one line.
    incidents: dict[str, dict] = {}
    for c in cases:
        p = c.get("pattern") or {}
        if p.get("level") in ("watch", "crisis"):
            key = p.get("theme") or p.get("cluster_id") or "unnamed cluster"
            entry = incidents.setdefault(key, {"theme": key, "level": p["level"], "cases": 0,
                                               "cluster_size": 0})
            entry["cases"] += 1
            entry["cluster_size"] = max(entry["cluster_size"], int(p.get("cluster_size") or 0))
            if p["level"] == "crisis":
                entry["level"] = "crisis"

    avg_cost = sum(closed_costs) / len(closed_costs) if closed_costs else None
    return {
        "total": len(cases),
        "status": dict(statuses),
        "by_category": dict(Counter(t.get("category", "untriaged") for t in triages).most_common()),
        "by_severity": {str(s): n for s, n in sorted(Counter(t.get("severity") for t in triages
                                                             if t.get("severity")).items())},
        "by_tier": dict(Counter((c.get("priority") or {}).get("tier", "—") for c in cases)),
        "by_author": dict(Counter((c.get("verdict") or {}).get("author_class", "—") for c in cases)),
        "by_channel": dict(Counter((c.get("complaint") or {}).get("channel", "—") for c in cases)),
        "untagged": sum(1 for c in cases if (c.get("complaint") or {}).get("tagged") is False),
        "reviews": {
            "total": len(reviews),
            "auto": auto,
            "auto_rate": round(auto / len(reviews), 4) if reviews else None,
            "human": len(human),
            "approved_unchanged": decisions.get("approve", 0),
            "edited": decisions.get("edit", 0),
            "rejected": decisions.get("reject", 0),
        },
        "outcomes": {
            "closed": len(closed),
            "resolved": statuses.get("resolved", 0),
            "resolved_rate": round(statuses.get("resolved", 0) / len(closed), 4) if closed else None,
            "degraded": degraded,
        },
        "cost": {
            "avg_per_closed_case_inr": round(avg_cost, 4) if avg_cost is not None else None,
            "total_inr": round(sum(float(c.get("cost_inr") or 0.0) for c in cases), 4),
            "human_reply_inr": HUMAN_COST_INR["social_reply"],
        },
        "incidents": sorted(incidents.values(), key=lambda i: (i["level"] != "crisis", -i["cases"])),
        "dead_letters": dead_letters,
        "autonomy": autonomy or [],
    }
