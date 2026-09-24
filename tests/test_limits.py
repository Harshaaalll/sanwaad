"""Tests for the bound on how much happens at once.

Complaints do not arrive evenly. They arrive in bursts, because the thing
people are complaining about is one outage — which is exactly what the pattern
agent exists to detect. So the moment this system is most useful is the moment
the most work arrives, and without a bound that moment is a thundering herd:
every case racing for the same model quota, each one slower, some timing out,
and the retry logic turning every timeout into three more requests.

The two pools behave differently on purpose, and that difference is what these
tests are mostly about. A case that waits finishes late, which is fine. A call
that waits is a caller listening to silence, so it is refused instead.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.limits import AtCapacity, Pool, report


@pytest.mark.asyncio
async def test_a_burst_runs_at_the_limit_not_all_at_once():
    pool = Pool("test", 3)
    seen: list[int] = []

    async def work():
        async with pool.slot():
            seen.append(pool.running)
            await asyncio.sleep(0.01)

    await asyncio.gather(*(work() for _ in range(12)))
    assert max(seen) == 3
    assert pool.peak_running == 3


@pytest.mark.asyncio
async def test_everything_still_gets_done():
    """A bound must not drop work. Twelve complaints arriving at once is
    twelve cases, four at a time."""
    pool = Pool("test", 2)
    done = []

    async def work(n):
        async with pool.slot():
            done.append(n)

    await asyncio.gather(*(work(n) for n in range(12)))
    assert sorted(done) == list(range(12))


@pytest.mark.asyncio
async def test_the_pool_empties_again_afterwards():
    pool = Pool("test", 2)
    async with pool.slot():
        assert pool.running == 1
    assert pool.running == 0 and pool.waiting == 0


@pytest.mark.asyncio
async def test_a_slot_is_released_even_when_the_work_raises():
    """Otherwise one failing case permanently shrinks the pool, and enough of
    them deadlock the system — a bug that only appears under the load the
    bound was added for."""
    pool = Pool("test", 1)
    with pytest.raises(ValueError):
        async with pool.slot():
            raise ValueError("case blew up")
    assert pool.running == 0
    async with pool.slot():            # the slot came back
        assert pool.running == 1


@pytest.mark.asyncio
async def test_a_refusing_pool_says_no_instead_of_queueing():
    """The voice pool. A call that waits is dead air on the line."""
    pool = Pool("calls", 1)
    async with pool.slot(wait=False):
        with pytest.raises(AtCapacity) as exc:
            async with pool.slot(wait=False):
                pass
    assert exc.value.pool == "calls" and exc.value.limit == 1
    assert pool.refused == 1


@pytest.mark.asyncio
async def test_a_waiting_pool_queues_rather_than_refusing():
    """The case pool. Finishing late is the right answer for a public thread."""
    pool = Pool("cases", 1)
    order = []

    async def work(n):
        async with pool.slot():
            order.append(n)
            await asyncio.sleep(0.01)

    await asyncio.gather(work(1), work(2))
    assert sorted(order) == [1, 2]
    assert pool.refused == 0


@pytest.mark.asyncio
async def test_waiting_is_visible_while_it_is_happening():
    """A saturated pool and an idle one look identical from outside until
    something reports the queue."""
    pool = Pool("test", 1)
    peak_waiting = 0

    async def work():
        nonlocal peak_waiting
        async with pool.slot():
            peak_waiting = max(peak_waiting, pool.waiting)
            await asyncio.sleep(0.01)

    await asyncio.gather(*(work() for _ in range(4)))
    assert peak_waiting >= 1
    assert pool.report()["limit"] == 1


def test_counters_survive_a_slot_held_across_two_live_loops():
    """The failure this pins: the semaphore was per-loop and the counters were
    global, so rebuilding one zeroed the other. A holder on the old loop then
    decremented counters that had been reset, `running` went to -1, and stayed
    there — and /api/offer gates live calls on `CALLS.running >= CALLS.limit`,
    which can never be true again once the count is negative. A metrics bug
    that silently removes the limit it is metering.
    """
    import threading

    pool = Pool("cases", 2)
    held, release = threading.Event(), threading.Event()

    async def holder():
        async with pool.slot():
            held.set()
            await asyncio.to_thread(release.wait)

    t = threading.Thread(target=lambda: asyncio.run(holder()))
    t.start()
    held.wait(timeout=5)
    assert pool.running == 1

    async def other_loop():
        async def work():
            async with pool.slot():
                await asyncio.sleep(0.01)
        await asyncio.gather(*(work() for _ in range(4)))

    asyncio.run(other_loop())          # a whole second loop, start to finish
    release.set()
    t.join(timeout=5)

    assert pool.running == 0
    assert pool.waiting == 0


@pytest.mark.asyncio
async def test_acquire_now_takes_a_slot_or_says_there_is_none():
    """The voice path needs to decide before it can answer at all, so it takes
    the slot itself rather than checking and hoping."""
    pool = Pool("calls", 1)
    assert await pool.acquire_now() is True
    assert pool.running == 1
    assert await pool.acquire_now() is False      # refused, not queued
    assert pool.refused == 1
    pool.release_slot()
    assert pool.running == 0
    assert await pool.acquire_now() is True       # the slot came back


def test_a_pool_survives_being_used_by_a_second_event_loop():
    """Two loops one after another, which is the easy half: a semaphore carried
    into a loop that did not create it raises instead of limiting.

    This cannot detect loss of per-loop isolation — a single shared state
    passes it — so the overlapping case is covered separately by
    `test_counters_survive_a_slot_held_across_two_live_loops`, which is the one
    that actually contends."""
    pool = Pool("test", 2)

    async def use():
        async with pool.slot():
            return pool.running

    assert asyncio.run(use()) == 1
    assert asyncio.run(use()) == 1          # a different loop entirely


@pytest.mark.asyncio
async def test_the_report_names_both_pools():
    names = [row["pool"] for row in report()]
    assert names == ["cases", "calls"]
