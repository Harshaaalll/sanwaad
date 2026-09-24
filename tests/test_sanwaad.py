"""Tests for the invariants that would be expensive to get wrong in production.

These deliberately avoid asserting on model output. What is tested is the
machinery around the model: routing, gating, grounding enforcement and the
consistency receipt. Those are the parts that must hold even when the model
has a bad day, and they are the parts a reviewer can verify without a key.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.config import REVIEW
from sanwaad.consistency import build_receipt
from sanwaad.graph.graph import _after_ground_check, _after_prioritise
from sanwaad.graph.nodes import (
    _largest_rupee_amount,
    _offline_triage,
    auto_post_allowed,
    prioritise,
)
from sanwaad.rag.store import CATEGORY_CLAUSES, get_store
from sanwaad.voice.brief import CitationTracker

# --- Routing safety --------------------------------------------------------

# The drop decision moved from `_after_triage` to `_after_prioritise` when the
# judge and pattern agents landed: triage alone can no longer see enough to
# close a case. The invariants below are the same two that mattered before.

_CREDIBLE = {"authenticity": 0.9, "reach": 10, "reply_worthy": True,
             "author_class": "customer", "history_with_brand": 0}
_QUIET = {"level": "none", "cluster_size": 1}


def test_high_severity_is_never_dropped_even_if_miscategorised():
    """The failure that ends a pilot: a regulatory threat classified as
    off-topic and silently closed with no reply."""
    triage = {"is_complaint": False, "severity": 5, "category": "off_topic"}
    p = prioritise(triage, _CREDIBLE, _QUIET)
    assert p.drafting
    assert _after_prioritise({"priority": p.model_dump()}) == "retrieve"


def test_genuine_off_topic_still_closes_cheaply():
    triage = {"is_complaint": False, "severity": 1, "category": "off_topic"}
    p = prioritise(triage, _CREDIBLE, _QUIET)
    assert not p.drafting
    assert _after_prioritise({"priority": p.model_dump()}) == "close"


def test_regulatory_keywords_force_severity_5():
    for text in ["Filing with the RBI Ombudsman this week",
                 "taking this to consumer court",
                 "sending a legal notice"]:
        assert _offline_triage(text)["severity"] == 5, text


def test_ungrounded_draft_is_revised_then_handed_to_a_human():
    ungrounded = {"grounding": {"grounded": False}}
    assert _after_ground_check({**ungrounded, "revision_count": 1}) == "draft"
    # Two failures means the policy index lacks the answer. Stop burning tokens.
    # It goes on to planning and then the review gate, which holds it for a human.
    assert _after_ground_check({**ungrounded, "revision_count": 2}) == "plan"


# --- The auto-post gate ----------------------------------------------------

def _state(**over):
    base = {
        "triage": {"severity": 1, "needs_private_data": False},
        "draft": {"promises_compensation": False},
        "grounding": {"grounded": True},
    }
    for key, value in over.items():
        base[key] = {**base[key], **value} if key in base else value
    return base


def test_benign_grounded_reply_may_post_unattended():
    allowed, _ = auto_post_allowed(_state())
    assert allowed


def test_never_auto_posts_a_money_promise():
    allowed, reason = auto_post_allowed(_state(draft={"promises_compensation": True}))
    assert not allowed and "RFD-05" in reason


def test_never_auto_posts_an_ungrounded_claim():
    allowed, reason = auto_post_allowed(_state(grounding={"grounded": False}))
    assert not allowed and "unsupported" in reason


def test_never_auto_posts_above_the_severity_ceiling_without_a_track_record():
    """Above the fixed ceiling a reply is a person's call until the capability
    has earned otherwise — and with no ledger there is nothing earned."""
    allowed, reason = auto_post_allowed(
        _state(triage={"severity": REVIEW.auto_post_max_severity + 1})
    )
    assert not allowed
    assert "has not earned it" in reason


def test_high_severity_is_never_autonomous_however_good_the_record(tmp_path, monkeypatch):
    """Autonomy widens the ordinary case. It does not reach the cases where
    being wrong is expensive, and no amount of agreement moves that line."""
    from datetime import datetime, timedelta, timezone

    import sanwaad.autonomy as autonomy_mod
    from sanwaad.graph.nodes import AUTONOMY_MAX_SEVERITY

    ledger = autonomy_mod.AutonomyLedger(path=tmp_path / "autonomy.jsonl")
    monkeypatch.setattr(autonomy_mod, "LEDGER", ledger)
    now = datetime.now(timezone.utc)
    for i in range(200):                       # a spotless record
        ledger.record("reply.billing", agreed=True, at=now - timedelta(hours=i))

    allowed, reason = auto_post_allowed(
        _state(triage={"severity": AUTONOMY_MAX_SEVERITY + 1, "category": "billing"})
    )
    assert not allowed
    assert "whatever the track record says" in reason


def test_a_upi_address_is_redacted_like_any_other_identifier():
    """In a payments product the VPA is the identifier a customer is likeliest
    to type, and the email pattern misses every one: a UPI handle has no dot
    after the @. It reached the pattern store, the audit log and the
    dead-letter excerpts that /api/delivery renders."""
    from sanwaad.guardrails import redact

    for said in ("refund to rohit.mhrs@okhdfcbank please", "my vpa is anita92@ybl"):
        clean, kinds = redact(said)
        assert "@" not in clean and kinds == ["vpa"]

    # A phone-shaped VPA loses the whole address, not just its digits.
    assert redact("9876543210@paytm")[0] == "[upi id]"
    # And the brand mention that starts half the complaints is not an address.
    assert redact("@NimbusPay double debit ₹640")[0] == "@NimbusPay double debit ₹640"


# --- Consistency receipt ---------------------------------------------------

def test_call_going_deeper_than_the_tweet_is_not_a_divergence():
    r = build_receipt(["RFD-01"], ["RFD-01", "RFD-02"])
    assert r.consistent and r.voice_only == ["RFD-02"]


def test_contradictory_clauses_across_channels_are_flagged():
    # Public: "that's the merchant's problem". Call: "we owe you TAT compensation".
    assert not build_receipt(["RFD-04"], ["RFD-03"]).consistent


def test_case_without_a_call_cannot_be_inconsistent():
    assert build_receipt(["RFD-01"], []).consistent


def test_channels_sharing_no_clause_at_all_is_flagged():
    assert not build_receipt(["KYC-02"], ["KYC-04"]).consistent


# --- Retrieval -------------------------------------------------------------

def test_category_boost_surfaces_the_right_clause_family():
    store = get_store()
    hits = store.search(
        "customer says refund has not arrived 5 days after a failed UPI transaction",
        k=3, boost_prefixes=CATEGORY_CLAUSES["refund"],
    )
    assert all(c.clause_id.startswith("RFD-") for c in hits)


def test_devanagari_retrieves_the_account_freeze_policy():
    hits = get_store().search("मेरा अकाउंट फ्रीज हो गया है", k=3)
    assert "KYC-01" in [c.clause_id for c in hits]


def test_constitutional_clauses_are_always_present():
    ids = [c.clause_id for c in get_store().always_on()]
    assert "BV-01" in ids and "PRV-01" in ids


def test_every_clause_id_is_unique():
    ids = [c["clause_id"] for c in get_store().clauses]
    assert len(ids) == len(set(ids))


# --- Voice citation tracking ----------------------------------------------

def test_tracker_detects_clauses_used_in_hinglish_speech():
    store = get_store()
    tracker = CitationTracker([store.get("RFD-01"), store.get("RFD-02"), store.get("KYC-04")])
    tracker.observe("Failed transfer teen working days mein auto reverse ho jata hai")
    tracker.observe("Nahi aaya to hum chargeback raise karenge sponsor bank ke saath")
    used = tracker.used()
    assert "RFD-01" in used and "RFD-02" in used
    assert "KYC-04" not in used  # never mentioned


# --- Amount parsing (drives escalation) ------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("₹4,500 transfer kiya tha", 4500.0),
    ("Rs. 18,000 stuck", 18000.0),
    ("INR 640 taken twice", 640.0),
    ("charged 15 rupees", None),          # no currency marker before the number
    ("₹1,200 and also ₹9,999 later", 9999.0),
])
def test_largest_rupee_amount(text, expected):
    assert _largest_rupee_amount(text) == expected
