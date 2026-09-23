"""Tests for authority the system earns rather than is given.

The goal is a system that runs without a human touch. The failure mode is
getting there by removing the human, because an agent's first day and its
thousandth look identical from the inside and only one of them has a track
record. So autonomy is a measured property of one capability, it climbs on
evidence, and it falls the moment the evidence stops — which is what makes it
safe to grant at all.

The tests that matter most here are the ones about falling, not climbing.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.autonomy import AutonomyLedger, Level, ceiling_for

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def ledger(tmp_path):
    return AutonomyLedger(path=tmp_path / "autonomy.jsonl")


def _history(ledger, capability, *, agreed: int, disagreed: int = 0,
             days_ago_start: int = 1):
    at = NOW - timedelta(days=days_ago_start)
    for _ in range(agreed):
        ledger.record(capability, agreed=True, at=at)
    for _ in range(disagreed):
        ledger.record(capability, agreed=False, at=at)


# --- climbing --------------------------------------------------------------

def test_a_new_capability_starts_with_a_person(ledger):
    """Day one has no evidence, and the absence of evidence is not a licence."""
    v = ledger.verdict("reply.billing", now=NOW)
    assert v.level is Level.ASSISTED
    assert not v.acts_without_a_person
    assert "no track record" in v.reason


def test_agreement_over_enough_decisions_buys_supervision(ledger):
    _history(ledger, "reply.billing", agreed=25)
    v = ledger.verdict("reply.billing", now=NOW)
    assert v.level is Level.SUPERVISED
    assert v.acts_without_a_person


def test_acting_unannounced_takes_twice_as_much_evidence(ledger):
    _history(ledger, "reply.billing", agreed=45)
    assert ledger.verdict("reply.billing", now=NOW).level is Level.AUTONOMOUS


def test_a_good_afternoon_is_not_a_track_record(ledger):
    """Nineteen perfect decisions is still nineteen."""
    _history(ledger, "reply.billing", agreed=19)
    assert ledger.verdict("reply.billing", now=NOW).level is Level.ASSISTED


# --- falling, which is the half that makes it safe -------------------------

def test_authority_contracts_when_agreement_drops(ledger):
    """The property a fixed policy cannot have: the day a model, a prompt or a
    policy clause changes, agreement falls and authority follows it down
    without anyone having to notice the regression first."""
    _history(ledger, "reply.billing", agreed=45)
    assert ledger.verdict("reply.billing", now=NOW).level is Level.AUTONOMOUS

    _history(ledger, "reply.billing", agreed=0, disagreed=6)      # 88% agreement
    assert ledger.verdict("reply.billing", now=NOW).level is Level.ASSISTED


def test_a_capability_that_is_usually_wrong_stops_deciding(ledger):
    _history(ledger, "reply.billing", agreed=10, disagreed=20)
    v = ledger.verdict("reply.billing", now=NOW)
    assert v.level is Level.SHADOW
    assert not v.acts_without_a_person


def test_one_consequential_mistake_outranks_a_good_average(ledger):
    """An average is exactly where a rare expensive mistake hides. A reply that
    went out and had to be retracted is not 1% of a bad week."""
    _history(ledger, "reply.billing", agreed=200)
    ledger.record("reply.billing", agreed=False, consequential=True,
                  at=NOW - timedelta(days=1))
    v = ledger.verdict("reply.billing", now=NOW)
    assert v.level is Level.ASSISTED
    assert "consequential" in v.reason


def test_the_cooloff_expires_so_one_mistake_is_not_a_life_sentence(ledger):
    _history(ledger, "reply.billing", agreed=200, days_ago_start=2)
    ledger.record("reply.billing", agreed=False, consequential=True,
                  at=NOW - timedelta(days=30))
    assert ledger.verdict("reply.billing", now=NOW).level is Level.AUTONOMOUS


def test_evidence_goes_stale(ledger):
    """A record from last quarter says nothing about the model running today."""
    _history(ledger, "reply.billing", agreed=100, days_ago_start=400)
    v = ledger.verdict("reply.billing", now=NOW)
    assert v.level is Level.ASSISTED
    assert v.decisions == 0


# --- the line that is not for sale -----------------------------------------

def test_a_tool_that_needs_a_person_caps_the_capability():
    """`initiate_reversal` is not auto_approvable. No track record buys past
    that, because a track record is evidence about the common case and an
    irreversible transfer of someone else's money is not the case you want to
    be average about."""
    assert ceiling_for(("initiate_reversal",)) is Level.ASSISTED
    assert ceiling_for(("post_reply", "open_ticket")) is Level.AUTONOMOUS
    # Reads and low-risk writes were never gated on a person to begin with, so
    # they cap nothing: open_ticket is WRITE_LOW and needs no approval at all.
    assert ceiling_for(("lookup_transaction", "open_ticket")) is Level.AUTONOMOUS


def test_the_ceiling_is_applied_and_said_out_loud(ledger):
    _history(ledger, "refund.reverse", agreed=200)
    v = ledger.verdict("refund.reverse", ceiling=Level.ASSISTED, now=NOW)
    assert v.level is Level.ASSISTED
    assert "capped" in v.reason


def test_autonomy_can_be_switched_off_entirely(ledger, monkeypatch):
    """A pilot customer who wants nothing automatic gets that from one
    variable, rather than being told the system cannot do what it was for."""
    import sanwaad.autonomy as autonomy_mod

    _history(ledger, "reply.billing", agreed=200)
    monkeypatch.setattr(autonomy_mod, "ENABLED", False)
    assert ledger.verdict("reply.billing", now=NOW).level is Level.ASSISTED


# --- bookkeeping -----------------------------------------------------------

def test_capabilities_are_counted_separately(ledger):
    _history(ledger, "reply.billing", agreed=45)
    _history(ledger, "reply.refund", agreed=2)
    assert ledger.verdict("reply.billing", now=NOW).level is Level.AUTONOMOUS
    assert ledger.verdict("reply.refund", now=NOW).level is Level.ASSISTED


def test_a_failed_write_never_fails_the_case(ledger, monkeypatch):
    """Missing evidence can only slow a promotion, never cause one, so the safe
    direction on a bookkeeping error is to carry on."""
    def boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", boom)
    ledger.record("reply.billing", agreed=True)      # must not raise


def test_the_report_names_every_capability(ledger):
    _history(ledger, "reply.billing", agreed=45)
    _history(ledger, "reply.refund", agreed=3)
    names = [r["capability"] for r in ledger.report(now=NOW)]
    assert names == ["reply.billing", "reply.refund"]
