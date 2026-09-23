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
class Pool:
    """A bounded number of slots, and an honest count of who is waiting.

    The waiting count is the part worth having. A saturated pool and a healthy
    one look identical from the outside until something reports the queue, and
    by then the only signal is latency, which arrives too late to act on.
    """

    name: str
    limit: int

    def __post_init__(self) -> None:
        self._sem: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = 0
        self.waiting = 0
        self.refused = 0
        self.peak_running = 0

    def _semaphore(self) -> asyncio.Semaphore:
        """The semaphore for the loop we are actually running in.

        An asyncio primitive belongs to the loop that first awaited it. These
        pools are module singletons, and a process runs more than one loop —
        every `asyncio.run`, every test. A semaphore carried across loops
        raises instead of limiting anything, so it is rebuilt when the loop
        underneath it changes.
        """
        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self._sem = asyncio.Semaphore(self.limit)
            self._loop = loop
            self.running = self.waiting = 0
        return self._sem

    @asynccontextmanager
    async def slot(self, *, wait: bool = True) -> AsyncIterator[None]:
        sem = self._semaphore()
        if not wait and sem.locked():
            self.refused += 1
            raise AtCapacity(self.name, self.limit)
        self.waiting += 1
        try:
            await sem.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        self.peak_running = max(self.peak_running, self.running)
        try:
            yield
        finally:
            self.running -= 1
            sem.release()

    def report(self) -> dict:
        return {"pool": self.name, "limit": self.limit, "running": self.running,
                "waiting": self.waiting, "peak_running": self.peak_running,
                "refused": self.refused}


CASES = Pool("cases", MAX_CONCURRENT_CASES)
CALLS = Pool("calls", MAX_CONCURRENT_CALLS)


def report(pools: Optional[list[Pool]] = None) -> list[dict]:
    return [p.report() for p in (pools or [CASES, CALLS])]
