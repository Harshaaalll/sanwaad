"""How much of this system may be happening at once.

Nothing here limited concurrency, and the shape of the traffic is what makes
that dangerous rather than merely untidy. Complaints do not arrive evenly.
They arrive in bursts, because the thing people are complaining about is one
outage — which is precisely what the pattern agent exists to detect. So the
moment this system is most useful is the moment the most work arrives at once,
and without a bound, a crisis becomes a thundering herd: every case racing for
the same model quota and the same ledger, every one of them slower, some of
them timing out, the retry logic turning each timeout into three more
requests.

A semaphore is not a performance feature. It is the difference between a queue
that drains slowly and a system that falls over while claiming to be fine.

Two pools, because the resources are not alike:

- CASES bound how many complaints run through the graph at once. A case that
  waits is a case that finishes late, which is fine — the customer is reading
  a public thread, not holding a line.
- CALLS bound live voice calls, and this one **refuses** rather than queues.
  Someone is on the phone. A call that waits for a slot is a caller listening
  to silence, and a busy signal is more honest than dead air.

That difference is the whole reason these are two pools and not one number.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

# Defaults sized for one laptop and a free-tier key: enough that a burst is
# absorbed rather than serialised, few enough that the provider's rate limit is
# not the thing that discovers the burst for us.
MAX_CONCURRENT_CASES = int(os.getenv("SANWAAD_MAX_CONCURRENT_CASES", "4"))
MAX_CONCURRENT_CALLS = int(os.getenv("SANWAAD_MAX_CONCURRENT_CALLS", "2"))


class AtCapacity(RuntimeError):
    """Raised by a pool that refuses rather than queues."""

    def __init__(self, pool: str, limit: int):
        super().__init__(f"{pool} is at capacity ({limit} in flight)")
        self.pool = pool
        self.limit = limit


@dataclass
class _LoopSlots:
    """One event loop's share of a pool: its semaphore and its own counters."""

    sem: asyncio.Semaphore
    running: int = 0
    waiting: int = 0


@dataclass
class Pool:
    """A bounded number of slots, and an honest count of who is waiting.

    The waiting count is the part worth having. A saturated pool and a healthy
    one look identical from the outside until something reports the queue, and
    by then the only signal is latency, which arrives too late to act on.

    Everything the loop owns is kept *per loop*, not just the semaphore. An
    asyncio primitive belongs to the loop that first awaited it, so the
    semaphore has to be per-loop; the first version of this kept the semaphore
    per-loop and the counters global, and rebuilding one zeroed the other. A
    holder on the old loop then ran `running -= 1` against counters that had
    been reset, so `running` went to -1 and stayed there — and
    `/api/offer` gates live calls on `CALLS.running >= CALLS.limit`, which can
    never be true again once the count is negative. A metrics bug that silently
    removes the limit it is metering.
    """

    name: str
    limit: int

    def __post_init__(self) -> None:
        # Weak keys: a process runs many short-lived loops (every asyncio.run,
        # every test) and a plain dict would hold every one of them forever.
        self._by_loop: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self.refused = 0
        self.peak_running = 0

    def _slots(self) -> _LoopSlots:
        loop = asyncio.get_running_loop()
        slots = self._by_loop.get(loop)
        if slots is None:
            slots = _LoopSlots(asyncio.Semaphore(self.limit))
            self._by_loop[loop] = slots
        return slots

    # Totals across every live loop. In the deployed system there is exactly
    # one, so these read as you would expect; in a test process they are the
    # honest sum, and neither can go negative because each loop only ever
    # decrements its own.
    @property
    def running(self) -> int:
        return sum(s.running for s in self._by_loop.values())

    @property
    def waiting(self) -> int:
        return sum(s.waiting for s in self._by_loop.values())

    @asynccontextmanager
    async def slot(self, *, wait: bool = True) -> AsyncIterator[None]:
        slots = self._slots()
        if not wait and slots.sem.locked():
            self.refused += 1
            raise AtCapacity(self.name, self.limit)
        slots.waiting += 1
        try:
            await slots.sem.acquire()
        finally:
            slots.waiting -= 1
        slots.running += 1
        self.peak_running = max(self.peak_running, self.running)
        try:
            yield
        finally:
            slots.running -= 1
            slots.sem.release()

    async def acquire_now(self) -> bool:
        """Take a slot immediately, or report that there is none.

        For a caller that must decide before it can do anything else — an HTTP
        handler that either starts a call or answers busy. Awaiting an
        unlocked semaphore does not yield, so the check and the take cannot be
        raced by another task. The caller MUST call `release_slot()`, which is
        why `slot()` is the right tool everywhere the work is in one place.
        """
        slots = self._slots()
        if slots.sem.locked():
            self.refused += 1
            return False
        await slots.sem.acquire()
        slots.running += 1
        self.peak_running = max(self.peak_running, self.running)
        return True

    def release_slot(self) -> None:
        slots = self._slots()
        slots.running -= 1
        slots.sem.release()

    def report(self) -> dict:
        return {"pool": self.name, "limit": self.limit, "running": self.running,
                "waiting": self.waiting, "peak_running": self.peak_running,
                "refused": self.refused}


CASES = Pool("cases", MAX_CONCURRENT_CASES)
CALLS = Pool("calls", MAX_CONCURRENT_CALLS)


def report(pools: Optional[list[Pool]] = None) -> list[dict]:
    return [p.report() for p in (pools or [CASES, CALLS])]
