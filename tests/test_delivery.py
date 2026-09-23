"""Tests for the floor under the listener's generosity.

The rule that an item is only marked seen after its handler returns is right,
and on its own it is unbounded: an item that fails deterministically is
refetched every cycle forever, costs a graph run each time, and is never
visible anywhere except as another identical log line.

These tests pin the bound. A transient failure must still be retried — that is
the whole reason for the rule. A permanent one must stop, land somewhere a
person can read it, and be recoverable once the cause is fixed.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.delivery import DEAD, RETRYING, DeliveryLog, delivery_key
from sanwaad.listener import Listener, SeenStore
from sanwaad.models import Channel, Complaint


def _complaint(external_id: str = "mock_x1", text: str = "money gone, ₹640") -> Complaint:
    return Complaint(external_id=external_id, channel=Channel.MOCK, author="u/test",
                     text=text, url="https://example.invalid/1",
                     created_at=datetime.now(timezone.utc).isoformat())


@pytest.fixture
def log(tmp_path):
    return DeliveryLog(path=tmp_path / "delivery.json", max_attempts=3)


# --- counting -------------------------------------------------------------

def test_a_failure_is_retried_before_it_is_given_up_on(log):
    """The first failures are usually a rate limit or a cold dependency, and
    those clear on their own. Giving up on the first one would lose complaints
    to a hiccup."""
    c = _complaint()
    first = log.record_failure(c, RuntimeError("upstream 503"))
    assert first.attempts == 1
    assert first.status == RETRYING
    assert not first.dead


def test_it_gives_up_at_the_limit(log):
    c = _complaint()
    for _ in range(3):
        failure = log.record_failure(c, ValueError("malformed record"))
    assert failure.attempts == 3
    assert failure.status == DEAD
    assert log.dead() and not log.retrying()


def test_attempts_survive_a_restart(log, tmp_path):
    """The counter is useless in memory: a crash loop restarts the process,
    and an in-memory count would reset with it and never reach the limit."""
    c = _complaint()
    log.record_failure(c, ValueError("boom"))
    log.record_failure(c, ValueError("boom"))

    reopened = DeliveryLog(path=tmp_path / "delivery.json", max_attempts=3)
    assert reopened.record_failure(c, ValueError("boom")).status == DEAD


def test_success_forgets_the_earlier_failures(log):
    """Two failures a week apart are not one item on its third life."""
    c = _complaint()
    log.record_failure(c, RuntimeError("timeout"))
    assert log.clear(c) is True
    assert log.record_failure(c, RuntimeError("timeout")).attempts == 1


def test_distinct_errors_are_kept_but_repeats_are_not(log):
    """"Failed three times the same way" and "failed three different ways" are
    different bugs. The second is the one worth seeing."""
    c = _complaint()
    log.record_failure(c, ValueError("same"))
    log.record_failure(c, ValueError("same"))
    log.record_failure(c, KeyError("different"))
    assert len(log.dead()[0].errors) == 2


def test_the_stored_excerpt_is_redacted(log):
    log.record_failure(_complaint(text="call me on 9876543210 about ₹640"),
                       ValueError("boom"))
    stored = json.loads(log.path.read_text())[0]["excerpt"]
    assert "9876543210" not in stored
    assert "[phone]" in stored


def test_an_unreadable_log_does_not_stop_delivery(log):
    """Failing closed here would stop the listener entirely because the file
    that records failures is itself broken."""
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text("{ this is not json")
    assert log.load() == []
    assert log.record_failure(_complaint(), ValueError("boom")).attempts == 1


# --- the loop that used to be infinite ------------------------------------

@pytest.mark.asyncio
async def test_a_poisoned_item_stops_being_refetched(tmp_path, monkeypatch):
    """The defect this exists for: before the bound, a handler that always
    raised left the item unmarked, so every poll refetched it, ran the graph on
    it and failed again — forever, invisibly."""
    seen = SeenStore(path=tmp_path / "seen.json")
    log = DeliveryLog(path=tmp_path / "delivery.json", max_attempts=3)
    listener = Listener(channels=["mock"], seen=seen, delivery=log)

    poison = _complaint("mock_poison")
    attempts = {"n": 0}

    async def _poll(limit=20):
        from sanwaad.listener import Heard
        fresh = seen.filter_new([poison])
        return Heard(complaints=fresh, skipped_seen=0, skipped_unrelated=0)

    async def handler(_c):
        attempts["n"] += 1
        raise ValueError("this will never work")

    monkeypatch.setattr(listener, "poll", _poll)
    await listener.watch(handler, interval_s=0, max_cycles=6)

    # Three attempts, then dead-lettered and marked seen — not six.
    assert attempts["n"] == 3
    assert delivery_key(poison) in seen.load()
    assert [f.key for f in log.dead()] == [delivery_key(poison)]


@pytest.mark.asyncio
async def test_a_transient_failure_still_gets_its_replay(tmp_path, monkeypatch):
    """The bound must not cost us the behaviour it bounds: an item that fails
    once and then succeeds is handled, marked seen, and leaves nothing behind."""
    seen = SeenStore(path=tmp_path / "seen.json")
    log = DeliveryLog(path=tmp_path / "delivery.json", max_attempts=3)
    listener = Listener(channels=["mock"], seen=seen, delivery=log)

    c = _complaint("mock_flaky")
    calls = {"n": 0}

    async def _poll(limit=20):
        from sanwaad.listener import Heard
        return Heard(complaints=seen.filter_new([c]), skipped_seen=0, skipped_unrelated=0)

    async def handler(_c):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("ledger warming up")

    monkeypatch.setattr(listener, "poll", _poll)
    await listener.watch(handler, interval_s=0, max_cycles=3)

    assert calls["n"] == 2                      # failed once, then succeeded
    assert delivery_key(c) in seen.load()
    assert log.load() == []                     # and left no trace behind


def test_a_partially_written_file_is_never_read(tmp_path):
    """The listener and the API are separate processes against one file.
    `write_text` truncates then writes, and `load` treats unparseable JSON as
    an empty queue — so a reader landing in that window silently wiped every
    attempt counter and restarted the unbounded retry loop this module exists
    to stop. Writing to a temp file and renaming makes the swap atomic.
    """
    log = DeliveryLog(path=tmp_path / "delivery.json", max_attempts=3)
    log.record_failure(_complaint("mock_a"), ValueError("boom"))

    seen = []
    for _ in range(40):
        log.record_failure(_complaint("mock_b"), ValueError("boom"))
        seen.append(len(DeliveryLog(path=log.path).load()))   # a concurrent reader
    assert 0 not in seen          # never observed as empty mid-write


def test_truncation_keeps_what_is_closest_to_giving_up(tmp_path):
    """Dropping the oldest records drops the longest-running failures, which
    are exactly the ones nearest the limit. Their counters would reset and the
    item would start its three lives over."""
    log = DeliveryLog(path=tmp_path / "delivery.json", max_attempts=3, max_records=5)
    old_one = _complaint("mock_old")
    log.record_failure(old_one, ValueError("boom"))
    log.record_failure(old_one, ValueError("boom"))        # two attempts in
    for i in range(20):
        log.record_failure(_complaint(f"mock_new_{i}"), ValueError("boom"))
    assert any(f.key == delivery_key(old_one) for f in log.load())


def test_the_path_is_resolved_at_construction(tmp_path, monkeypatch):
    """Binding DELIVERY_PATH as a default argument freezes it at import, so the
    eval harness redirecting it would be silently ignored."""
    import sanwaad.delivery as delivery_mod

    monkeypatch.setattr(delivery_mod, "DELIVERY_PATH", tmp_path / "redirected.json")
    assert delivery_mod.DeliveryLog().path == tmp_path / "redirected.json"


# --- the operator's side --------------------------------------------------

def test_requeue_clears_both_stores_or_the_item_never_returns(tmp_path, monkeypatch):
    """Dead-lettering marks the item seen. Clearing only the failure record
    would leave it seen forever, which looks like a requeue and is not one."""
    import sanwaad.delivery as delivery_mod

    seen = SeenStore(path=tmp_path / "seen.json")
    log = DeliveryLog(path=tmp_path / "delivery.json", max_attempts=1)
    monkeypatch.setattr(delivery_mod, "DELIVERY", log)
    monkeypatch.setattr("sanwaad.listener.SEEN_PATH", seen.path)

    c = _complaint("mock_dead")
    log.record_failure(c, ValueError("boom"))
    seen.mark([c])
    assert log.dead()

    assert delivery_mod.requeue_everywhere(delivery_key(c)) is True
    assert log.dead() == []
    assert delivery_key(c) not in seen.load()


def test_retention_prunes_the_delivery_log_like_every_other_store(tmp_path, monkeypatch):
    """It is a dated store, so it must be covered by the same retention pass —
    an operational queue nobody prunes is a disk-usage bug waiting to happen."""
    import sanwaad.delivery as delivery_mod
    from sanwaad.memory import MEMORY_MAP, prune

    tier = next(t for t in MEMORY_MAP if t.name == "Delivery failures")
    assert tier.prunable and tier.retention_days == 90

    path = tmp_path / "delivery.json"
    monkeypatch.setattr(delivery_mod, "DELIVERY_PATH", path)
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    path.write_text(json.dumps([{"key": "mock:old", "at": old}]))

    row = next(r for r in prune(dry_run=True) if r["tier"] == "Delivery failures")
    assert row["removed"] == 1
