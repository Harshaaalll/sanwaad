"""Tests for the circuit breaker: stop calling what is already failing.

A retry loop protects one call from one bad moment. It cannot know that the
last hundred calls also failed, so against a dependency that is simply down it
makes things worse: every case spends its full retry allowance rediscovering
the outage, and the backend is hammered while it tries to come back.

Two things have to be true for a breaker to be worth having, and both are
easy to get wrong:

- it must open on the failures that mean the dependency is unwell, and on no
  others. One that counts a business conflict opens on healthy traffic and
  takes the system down in the name of protecting it.
- it must close by itself. Otherwise the recovery path is a person at 3am.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.tools import REGISTRY, ErrorCode, ToolError
from sanwaad.tools.breaker import CLOSED, HALF_OPEN, OPEN, BreakerPolicy, CircuitBreaker


@pytest.fixture
def breaker():
    return CircuitBreaker(BreakerPolicy(threshold=3, window_s=60, cooldown_s=30))


# --- opening --------------------------------------------------------------

def test_it_opens_once_the_failures_pile_up(breaker):
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    assert breaker.state("ledger", now=100) == OPEN
    assert breaker.allows("ledger", now=100) is False


def test_it_stays_closed_below_the_threshold(breaker):
    for _ in range(2):
        breaker.record_outcome("ledger", ErrorCode.TIMEOUT, now=100)
    assert breaker.state("ledger", now=100) == CLOSED
    assert breaker.allows("ledger", now=100) is True


@pytest.mark.parametrize("code", [
    ErrorCode.CONFLICT,        # already reversed: the backend answered
    ErrorCode.NOT_FOUND,       # no such transaction: the backend answered
    ErrorCode.INVALID_OUTPUT,  # answered, in the wrong shape: still answered
])
def test_an_answer_we_dislike_is_not_an_outage(breaker, code):
    """The question is not "did the call succeed" but "did anyone answer".
    Counting these would open the circuit on perfectly healthy traffic."""
    for _ in range(10):
        breaker.record_outcome("ledger", code, now=100)
    assert breaker.state("ledger", now=100) == CLOSED


def test_old_failures_fall_out_of_the_window(breaker):
    """Two failures a day apart are not a pattern. Without a window they would
    accumulate forever and the breaker would eventually open on nothing."""
    breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=0)
    breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=10)
    breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=1000)
    assert breaker.state("ledger", now=1000) == CLOSED


def test_a_lucky_success_does_not_empty_the_window(breaker):
    """This test used to assert the opposite, and the opposite is a bug.

    Clearing the window on every success makes the breaker a consecutive
    counter — precisely what `BreakerPolicy`'s own comment says a window exists
    to avoid. A dependency dropping one request in five produces a lucky
    success often enough that the count never reaches the threshold, so the
    circuit stayed closed through exactly the outage it was built for.
    """
    for t in range(100, 102):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=t)
    breaker.record_outcome("ledger", now=102)                 # one lucky success
    breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=103)
    assert breaker.state("ledger", now=103) == OPEN


def test_a_half_dead_dependency_still_opens_the_circuit():
    """The shipped policy against the traffic it was written for: a backend
    dropping about one request in five, which never fails five times in a row."""
    b = CircuitBreaker()                       # threshold 5, window 60s
    t = 0.0
    for i in range(20):
        b.record_outcome("ledger", ErrorCode.UPSTREAM, now=t)
        t += 1
        if i % 4 == 3:
            b.record_outcome("ledger", now=t)  # the lucky one
            t += 1
    assert b.report(now=t)[0]["trips"] >= 1


def test_a_healthy_tool_never_trips(breaker):
    """The other direction: failures age out, so a slow drip that never
    reaches the threshold inside one window must leave the circuit closed."""
    for t in range(0, 500, 100):               # one failure every 100s
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=float(t))
    assert breaker.state("ledger", now=500.0) == CLOSED


def test_one_tool_failing_does_not_stop_another(breaker):
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    assert breaker.allows("ledger", now=100) is False
    assert breaker.allows("ticket_desk", now=100) is True


# --- recovering -----------------------------------------------------------

def test_it_half_opens_when_the_cooldown_expires(breaker):
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    assert breaker.state("ledger", now=120) == OPEN
    assert breaker.state("ledger", now=131) == HALF_OPEN


def test_only_one_call_probes_a_half_open_circuit(breaker):
    """Letting the whole backlog through the instant the cooldown expires is a
    retry storm on a timer. Exactly one call finds out."""
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    assert breaker.allows("ledger", now=131) is True     # the probe
    assert breaker.allows("ledger", now=131) is False    # everyone else waits


def test_a_successful_probe_closes_the_circuit(breaker):
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    breaker.allows("ledger", now=131)
    breaker.record_outcome("ledger", now=131)
    assert breaker.state("ledger", now=131) == CLOSED
    assert breaker.allows("ledger", now=131) is True


def test_a_failed_probe_buys_another_full_cooldown(breaker):
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    breaker.allows("ledger", now=131)
    breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=131)
    assert breaker.state("ledger", now=140) == OPEN
    assert breaker.state("ledger", now=162) == HALF_OPEN


def test_a_probe_that_gets_any_answer_closes_the_circuit(breaker):
    """A probe returning CONFLICT means the backend is alive and arguing. That
    is recovery. It also must not leave the probe flag set, or the tool stays
    half-open forever with its one probe already spent."""
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    breaker.allows("ledger", now=131)
    breaker.record_outcome("ledger", ErrorCode.CONFLICT, now=131)
    assert breaker.state("ledger", now=131) == CLOSED
    assert breaker.allows("ledger", now=131) is True


# --- through the registry, which is where it has to work ------------------

@pytest.mark.asyncio
async def test_the_registry_stops_calling_a_dead_tool(monkeypatch):
    """The point of the whole thing: a hundred cases against a dead ledger
    must not be three hundred doomed requests."""
    monkeypatch.setattr(REGISTRY, "breaker",
                        CircuitBreaker(BreakerPolicy(threshold=3, window_s=60, cooldown_s=30)))
    REGISTRY.inject_fault(
        "lookup_transaction",
        ToolError(code=ErrorCode.UPSTREAM, message="ledger down", retryable=True),
        times=50)
    try:
        codes = []
        for _ in range(4):
            result = await REGISTRY.call(
                "lookup_transaction", {"handle": "u/karthik_rn", "amount_inr": 640},
                agent="plan")
            codes.append(result.error.code)
        # The first calls spend their retries discovering the outage; by the
        # fourth the breaker answers without calling the backend at all.
        assert codes[-1] is ErrorCode.CIRCUIT_OPEN
        assert ErrorCode.UPSTREAM in codes
    finally:
        REGISTRY.clear_faults()


@pytest.mark.asyncio
async def test_a_refused_call_costs_no_attempts(monkeypatch):
    """Failing fast has to actually be fast: no retries, no timeout waited."""
    breaker = CircuitBreaker(BreakerPolicy(threshold=1, window_s=60, cooldown_s=30))
    monkeypatch.setattr(REGISTRY, "breaker", breaker)
    REGISTRY.inject_fault(
        "lookup_transaction",
        ToolError(code=ErrorCode.UPSTREAM, message="down", retryable=True), times=50)
    try:
        await REGISTRY.call("lookup_transaction", {"handle": "u/karthik_rn"}, agent="plan")
        refused = await REGISTRY.call("lookup_transaction", {"handle": "u/karthik_rn"},
                                      agent="plan")
        assert refused.error.code is ErrorCode.CIRCUIT_OPEN
        assert refused.attempts == 0
    finally:
        REGISTRY.clear_faults()


def test_an_abandoned_probe_is_handed_back(breaker):
    """A cancelled task unwinds without an outcome. If the probe flag stayed
    set the tool would be refused for the life of the process — one client
    disconnect and the ledger is unreachable until a restart."""
    for _ in range(3):
        breaker.record_outcome("ledger", ErrorCode.UPSTREAM, now=100)
    assert breaker.allows("ledger", now=131) is True     # probe taken
    assert breaker.allows("ledger", now=131) is False    # nobody else may
    breaker.abandon("ledger")
    assert breaker.allows("ledger", now=131) is True     # available again


@pytest.mark.asyncio
async def test_a_cancelled_call_does_not_strand_the_probe(monkeypatch):
    """The same thing through the registry, where the cancellation happens."""
    import asyncio

    breaker = CircuitBreaker(BreakerPolicy(threshold=1, window_s=60, cooldown_s=0))
    monkeypatch.setattr(REGISTRY, "breaker", breaker)
    breaker.record_outcome("lookup_transaction", ErrorCode.UPSTREAM)

    async def _hang(_inp):
        await asyncio.sleep(3600)

    # ToolSpec is frozen, so swap in a copy that hangs, under the same name —
    # the name is what the agent's contract is checked against.
    import dataclasses

    spec = REGISTRY.get("lookup_transaction")
    monkeypatch.setitem(REGISTRY._tools, "lookup_transaction",
                        dataclasses.replace(spec, handler=_hang))

    task = asyncio.create_task(
        REGISTRY.call("lookup_transaction", {"handle": "u/karthik_rn"}, agent="plan"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The probe was handed back, so the next caller may take one.
    assert breaker.allows("lookup_transaction") is True


@pytest.mark.asyncio
async def test_clearing_injected_faults_also_forgets_the_circuits_they_tripped():
    """Otherwise the next eval scenario starts against a tool that is still
    failing fast because of an outage the previous scenario invented."""
    REGISTRY.breaker.record_outcome("lookup_transaction", ErrorCode.UPSTREAM)
    REGISTRY.clear_faults()
    assert REGISTRY.breaker.report() == []
    result = await REGISTRY.call("lookup_transaction", {"handle": "u/karthik_rn"},
                                 agent="plan")
    assert result.ok
