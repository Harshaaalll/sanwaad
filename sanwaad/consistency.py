"""The consistency receipt — the product's central claim, made checkable.

A support organisation's real failure mode is not a wrong answer. It is two
answers: the social team quotes seven days, the call centre quotes three, and
the customer, reasonably, concludes the company is lying to them.

Because both channels here retrieve from one clause index, we can record which
clauses each channel actually relied on and compare the sets. A clause used on
the call but never on the thread is not automatically wrong — a call goes
deeper — but a clause that CONTRADICTS one already cited in public is a defect
we can detect mechanically rather than discover in a screenshot.
"""

from __future__ import annotations

from .models import ConsistencyReceipt

# Clause pairs that must never both be relied on for the same case. These are
# the genuine contradictions in the policy set: they answer the same customer
# question with different money or different timelines.
CONTRADICTORY_PAIRS: set[frozenset[str]] = {
    frozenset({"RFD-01", "RFD-06"}),  # T+3 auto-reversal vs 24h duplicate-debit reversal
    frozenset({"RFD-04", "RFD-03"}),  # merchant's problem vs we owe TAT compensation
    frozenset({"KYC-02", "KYC-04"}),  # resolve it yourself in 10 min vs funds are liened
}


def build_receipt(text_clauses: list[str], voice_clauses: list[str]) -> ConsistencyReceipt:
    text_set, voice_set = set(text_clauses), set(voice_clauses)

    # A case that never went to voice cannot be inconsistent with itself.
    if not voice_set:
        return ConsistencyReceipt(
            shared_clauses=sorted(text_set),
            text_only=[], voice_only=[], consistent=True,
        )

    shared = text_set & voice_set
    text_only = text_set - voice_set
    voice_only = voice_set - text_set

    contradicted = any(
        pair <= (text_set | voice_set) and not pair <= shared
        for pair in CONTRADICTORY_PAIRS
    )

    # Divergence alone is fine; a call legitimately goes further than a tweet.
    # Inconsistency means the channels leaned on clauses that cannot both hold,
    # or the call ignored the public commitment entirely.
    consistent = not contradicted and bool(shared or not text_set)

    return ConsistencyReceipt(
        shared_clauses=sorted(shared),
        text_only=sorted(text_only),
        voice_only=sorted(voice_only),
        consistent=consistent,
    )
