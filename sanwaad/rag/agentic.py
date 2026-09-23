"""Agentic RAG: let retrieval decide whether it succeeded, and retry if not.

Ordinary RAG retrieves once and hopes. It has no way to notice that it pulled
five plausible-looking clauses, none of which answer the question — and the
generator downstream will write a confident reply from whatever it was handed.

Agentic RAG closes that loop: retrieve, *assess coverage*, and if the answer
is not in hand, reformulate and go again. The word "agentic" just means the
retrieval step gets to make a decision instead of being a fixed pipeline
stage.

The cost discipline that makes it practical: assess with a free heuristic
first and only spend a model call when the heuristic is uncertain. A design
that calls an LLM to grade every retrieval doubles your per-case spend to fix
the minority of cases that were wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..models import Citation

# Vocabulary that signals a clause family. Cheap, auditable, and enough to
# answer "did we retrieve anything from the right area of the policy?"
_TOPIC_TERMS: dict[str, set[str]] = {
    "RFD": {"refund", "reversal", "reverse", "chargeback", "money back", "wapas",
            "returned", "failed transaction", "debited", "t+3"},
    "BIL": {"charge", "charged", "fee", "gst", "invoice", "deduct", "mandate",
            "subscription", "percent", "%"},
    "KYC": {"kyc", "freeze", "frozen", "block", "blocked", "hold", "lien",
            "access", "locked", "verification"},
    "ESC": {"ombudsman", "escalate", "callback", "nodal", "complaint", "sla",
            "court", "legal", "police", "third time"},
    "PRV": {"privacy", "otp", "fraud", "phishing", "data", "leak", "scam"},
}


@dataclass
class Coverage:
    covered: bool
    confidence: float           # 0..1, heuristic
    missing_topics: list[str] = field(default_factory=list)
    reason: str = ""


def assess(query: str, citations: list[Citation]) -> Coverage:
    """Free coverage check: do the retrieved clauses come from the families the
    query's vocabulary points at?

    Returns low confidence rather than a verdict when it cannot tell — that is
    the signal to spend a model call, not to guess.
    """
    q = query.lower()
    wanted = {
        prefix for prefix, terms in _TOPIC_TERMS.items()
        if any(t in q for t in terms)
    }
    if not wanted:
        return Coverage(True, 0.3, [], "no topic vocabulary matched; cannot judge cheaply")

    got = {c.clause_id.split("-")[0] for c in citations}
    missing = sorted(wanted - got)
    if not missing:
        return Coverage(True, 0.9, [], f"retrieved from every indicated family: {sorted(wanted)}")
    if wanted & got:
        return Coverage(True, 0.6, missing,
                        f"partial: got {sorted(wanted & got)}, missing {missing}")
    return Coverage(False, 0.85, missing,
                    f"nothing retrieved from the indicated families {sorted(wanted)}")


def reformulate(query: str, missing_topics: list[str]) -> str:
    """Rewrite the query toward the families that came back empty.

    Deterministic on purpose. An LLM rewrite is more flexible but costs a call
    and can wander off the original intent; appending the vocabulary of the
    missing family is enough to move a lexical retriever, which is usually the
    one that failed.
    """
    extra = []
    for prefix in missing_topics:
        extra.extend(sorted(_TOPIC_TERMS.get(prefix, set()))[:4])
    return f"{query} {' '.join(extra)}".strip()


def retrieve_with_retry(
    query: str,
    retriever: Callable[[str], list[Citation]],
    *,
    max_attempts: int = 2,
    min_confidence: float = 0.5,
) -> tuple[list[Citation], list[dict]]:
    """Retrieve, assess, reformulate, retry. Returns (citations, trail).

    The trail is kept because "why did this case get these clauses" is the
    first question asked when a reply is wrong, and reconstructing it after
    the fact is impossible.
    """
    trail: list[dict] = []
    current = query
    best: list[Citation] = []

    for attempt in range(1, max_attempts + 1):
        cites = retriever(current)
        cov = assess(current, cites)
        trail.append({
            "attempt": attempt, "query": current,
            "got": [c.clause_id for c in cites],
            "covered": cov.covered, "confidence": round(cov.confidence, 2),
            "reason": cov.reason,
        })

        # Union rather than replace: a second pass that finds the missing
        # family should not discard the correct clauses the first pass found.
        seen = {c.clause_id for c in best}
        best.extend(c for c in cites if c.clause_id not in seen)

        # Partial coverage is not success. A complaint spanning two policy
        # areas that retrieved only one of them will produce a reply that
        # confidently answers half the question, which is the failure mode
        # this loop exists to catch — so `missing_topics` gates the exit, not
        # confidence alone.
        if cov.covered and not cov.missing_topics and cov.confidence >= min_confidence:
            break
        if attempt < max_attempts and cov.missing_topics:
            current = reformulate(query, cov.missing_topics)
        else:
            break

    return best, trail
