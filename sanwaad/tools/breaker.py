"""A circuit breaker per tool: stop calling what is already failing.

Retries protect one call from one bad moment. They do nothing about the case
where a dependency is simply down, and they make it worse: every case that
arrives spends its full retry allowance discovering the same outage. A hundred
complaints against a dead ledger is three hundred doomed requests, three
hundred timeouts of latency paid by customers, and a backend being hammered
while it tries to come back up.

The breaker remembers across calls what a retry loop cannot. After enough
failures in a window it **opens**: further calls return immediately with
`CIRCUIT_OPEN` instead of waiting for a timeout that is now predictable. The
system stays up and degrades — Sanwaad already knows how to open a ticket when
a lookup is unavailable, and it can do that in a millisecond instead of nine
seconds.

Three states, and the third is the one that matters:

    closed      calls go through; failures are counted
    open        calls fail fast, for `cooldown_s`
    half-open   exactly one call is allowed through to find out

Half-open is what makes it recover by itself. Without it, something has to
decide when the outage is over, and that something is a person at 3am.

## What counts as a failure

Only TIMEOUT and UPSTREAM. Those say the dependency is unwell. An invalid
argument, a missing approval or a business conflict all say the *request* was
wrong, and no amount of not-calling will fix a wrong request. A breaker that
counted those would open on perfectly healthy traffic and take the system down
in the name of protecting it, which is the classic way this pattern is got
wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from .contracts import ErrorCode

# Failures that mean "the dependency is unwell" rather than "you asked wrong".
TRIPPING_CODES = frozenset({ErrorCode.TIMEOUT, ErrorCode.UPSTREAM})

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass
class BreakerPolicy:
    # Counted over attempts rather than calls: one call that retries three
    # times against a dead backend is three dead requests, and the breaker
    # should notice at the speed the damage is actually happening.
    threshold: int = 5
    # A window, not a consecutive count. Consecutive counting is reset by a
    # single lucky success, which is exactly what a half-dead dependency
    # produces.
    window_s: float = 60.0
    cooldown_s: float = 30.0


@dataclass
class _State:
    failures: list[float] = field(default_factory=list)
    opened_at: Optional[float] = None
    probing: bool = False
    trips: int = 0


class CircuitBreaker:
    """Per-tool state. One instance lives on the registry."""

    def __init__(self, policy: Optional[BreakerPolicy] = None):
        self.policy = policy or BreakerPolicy()
        self._states: dict[str, _State] = {}

    # --- what the registry asks --------------------------------------------

    def state(self, name: str, now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        st = self._states.get(name)
        if st is None or st.opened_at is None:
            return CLOSED
        if now - st.opened_at < self.policy.cooldown_s:
            return OPEN
        return HALF_OPEN

    def allows(self, name: str, now: Optional[float] = None) -> bool:
        """May this call go through?

        In half-open exactly one probe is allowed. Letting the whole backlog
        through the moment the cooldown expires would re-open the breaker with
        another hundred failures, which is a retry storm on a timer.
        """
        status = self.state(name, now)
        if status == CLOSED:
            return True
        if status == OPEN:
            return False
        st = self._states[name]
        if st.probing:
            return False
        st.probing = True
        return True

    # --- what the registry reports back -------------------------------------

    def record_outcome(self, name: str, code: Optional[ErrorCode] = None,
                       now: Optional[float] = None) -> None:
        """The one call the registry makes after every attempt.

        The question a breaker asks is not "did this call succeed" but "did the
        dependency answer at all". A CONFLICT or a NOT_FOUND is an answer: the
        backend is alive and disagreeing with us, which is health, not illness.
        Only a timeout or an upstream error says nobody is home.

        Routing every outcome through here also keeps half-open from
        deadlocking. The probe sets a flag that only an outcome clears, so an
        exit path that reported nothing would leave the tool permanently
        half-open with its one probe already spent.
        """
        if code is not None and code in TRIPPING_CODES:
            self.record_failure(name, code, now)
        else:
            self.record_success(name)

    def record_success(self, name: str) -> None:
        st = self._states.get(name)
        if st is None:
            return
        if st.opened_at is not None:
            # A probe got an answer: the outage is over, so the slate is clean.
            logger.info(f"circuit closed for {name}: a probe got an answer")
            self._states[name] = _State(trips=st.trips)
            return
        # Closed, and this is the important case. A success does NOT empty the
        # failure window. Emptying it turns the whole thing into a consecutive
        # counter, which is what the policy above says a window exists to avoid:
        # a dependency dropping one request in five produces a lucky success
        # often enough that the count never reaches the threshold, and the
        # circuit stays closed through exactly the outage it was built for.
        # Failures leave the window by ageing out of it, and by nothing else.
        st.probing = False

    def abandon(self, name: str) -> None:
        """Give back a probe we took and never got an answer for.

        A cancelled task — a client disconnecting, a shutdown, an outer
        timeout — unwinds without an outcome. Without this the probe flag stays
        set, `state()` reports half-open forever and every later call to that
        tool is refused for the life of the process: one cancellation and the
        tool is dead. Handing the probe back is honest, because we learned
        nothing either way.
        """
        st = self._states.get(name)
        if st is not None:
            st.probing = False

    def record_failure(self, name: str, code: ErrorCode,
                       now: Optional[float] = None) -> None:
        if code not in TRIPPING_CODES:
            return
        now = time.monotonic() if now is None else now
        st = self._states.setdefault(name, _State())

        # A failed probe re-opens the circuit for a fresh cooldown rather than
        # letting the next caller probe again immediately.
        if st.opened_at is not None:
            st.opened_at = now
            st.probing = False
            return

        st.failures = [t for t in st.failures if now - t < self.policy.window_s]
        st.failures.append(now)
        if len(st.failures) >= self.policy.threshold:
            st.opened_at = now
            st.trips += 1
            # The window is NOT cleared here. While open there is nothing to
            # count (calls are refused), a successful probe wipes the state
            # anyway, and leaving it means `report()` can say how many failures
            # tripped this circuit instead of always saying nought.
            logger.warning(
                f"circuit opened for {name}: {self.policy.threshold} failures in "
                f"{self.policy.window_s:g}s; failing fast for {self.policy.cooldown_s:g}s")

    # --- operators and tests -------------------------------------------------

    def reset(self, name: Optional[str] = None) -> None:
        if name is None:
            self._states.clear()
        else:
            self._states.pop(name, None)

    def report(self, now: Optional[float] = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        return [
            {"tool": name, "state": self.state(name, now), "trips": st.trips,
             "recent_failures": len([t for t in st.failures
                                     if now - t < self.policy.window_s])}
            for name, st in sorted(self._states.items())
        ]
