"""What happens to an inbound item that cannot be handled.

The listener's rule is deliberately generous: an item is marked seen only after
its handler returns, so a crash between fetching and handling replays the item
rather than losing it. A duplicate reply is embarrassing; a complaint that
silently evaporated is the failure the whole system exists to prevent, so the
tie goes to replaying.

That rule has no floor. An item that fails for a reason that will never go
away — a malformed record, a text that trips an unhandled edge — is refetched
on the next poll, fails again, and is never marked seen. Forever. Each cycle
spends a full graph run on it, and the only trace is another identical line in
the log. Nothing gives up and nothing surfaces it, which is the worst of both:
the item is not handled, and it is not visible either.

So this module puts a bound on generosity. Failures are counted across
restarts, retried a few times, and then declared **dead**: written down with
the error that killed them and marked seen, so the poll loop stops paying for
them. Dead is not lost. The record holds who posted, what they said and what
broke, and `requeue` puts it back after the bug is fixed — which is the part
that makes this a queue rather than a log file.

    python -m sanwaad.delivery                     what is stuck, and why
    python -m sanwaad.delivery --requeue KEY       try it again after a fix
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import DATA_DIR
from .guardrails import redact
from .models import Complaint

DELIVERY_PATH = DATA_DIR / "delivery_failures.json"

# Three attempts, because the failures worth retrying are the transient ones —
# a rate limit, a cold index, a dropped connection — and those clear within a
# poll cycle or two. A deterministic failure is not more likely to succeed on
# the tenth attempt than the third; it is just ten times as expensive.
MAX_ATTEMPTS = int(os.getenv("SANWAAD_MAX_DELIVERY_ATTEMPTS", "3"))

RETRYING = "retrying"
DEAD = "dead"

_EXCERPT_CHARS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Failure:
    key: str
    channel: str
    external_id: str
    author: str
    excerpt: str
    error: str
    attempts: int
    status: str
    first_at: str
    at: str                       # last attempt; `at` is the field prune() reads
    errors: list[str] = field(default_factory=list)

    @property
    def dead(self) -> bool:
        return self.status == DEAD


class DeliveryLog:
    """Failures on the way from a connector to a handled case.

    Stored as a JSON list rather than a map so the retention pass in memory.py
    can prune it like every other dated store, with no special case.
    """

    def __init__(self, path: Optional[Path] = None, max_attempts: int = MAX_ATTEMPTS,
                 max_records: int = 500):
        # Resolved here, not bound as a default: a default argument freezes
        # DELIVERY_PATH at import, so anything that redirects it — the eval
        # harness, a test — is silently ignored by every store built afterwards.
        self.path = path or DELIVERY_PATH
        self.max_attempts = max_attempts
        self.max_records = max_records

    # --- storage ------------------------------------------------------------

    def load(self) -> list[Failure]:
        if not self.path.exists():
            return []
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Same call as the seen-store makes: an unreadable file must not
            # stop the listener. The cost is re-counting attempts from zero.
            logger.warning("delivery log unreadable; treating as empty")
            return []
        out = []
        for row in rows:
            try:
                out.append(Failure(**row))
            except TypeError:
                continue      # a record from an older shape: ignore, don't crash
        return out

    def save(self, failures: list[Failure]) -> None:
        """Write via a temporary file and rename over the old one.

        `write_text` truncates and then writes, so a reader between the two
        sees a partial document. `load` treats unparseable JSON as an empty
        queue — a deliberate choice, so that a corrupt file cannot stop the
        listener — and the two together are worse than either: every attempt
        counter silently restarts at zero and the unbounded retry loop this
        module exists to stop comes back. The listener and the API server are
        separate processes against one file, so that window is real. A rename
        within a directory is atomic, so a reader sees the old file or the new
        one and never half of either.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncation drops the OLDEST records, which are the longest-running
        # failures — exactly the ones closest to being dead-lettered. Keep the
        # dead ones and the highest attempt counts instead of the newest.
        kept = sorted(failures, key=lambda f: (f.status == DEAD, f.attempts),
                      reverse=True)[:self.max_records]
        rows = [asdict(f) for f in sorted(kept, key=lambda f: f.first_at)]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # --- the two calls the poll loop makes ----------------------------------

    def record_failure(self, complaint: Complaint, exc: BaseException) -> Failure:
        """Count one failed delivery, and decide whether this one is finished.

        Returns the updated record. A caller should stop replaying the item
        once `status` is DEAD — it is written down here instead.
        """
        key = delivery_key(complaint)
        error = f"{type(exc).__name__}: {exc}"[:300]
        failures = self.load()
        existing = next((f for f in failures if f.key == key), None)

        if existing is None:
            existing = Failure(
                key=key,
                channel=complaint.channel.value,
                external_id=complaint.external_id,
                author=complaint.author,
                excerpt=redact(complaint.text or "")[0][:_EXCERPT_CHARS],
                error=error, attempts=1, status=RETRYING,
                first_at=_now(), at=_now(), errors=[error],
            )
            failures.append(existing)
        else:
            existing.attempts += 1
            existing.error = error
            existing.at = _now()
            # Keep the distinct errors, not every repetition: "it failed three
            # times the same way" and "it failed three different ways" are
            # different bugs, and the second is the one worth seeing.
            if error not in existing.errors:
                existing.errors.append(error)

        if existing.attempts >= self.max_attempts:
            existing.status = DEAD

        self.save(failures)
        return existing

    def clear(self, complaint: Complaint) -> bool:
        """Forget a past failure because the item finally went through."""
        key = delivery_key(complaint)
        failures = self.load()
        remaining = [f for f in failures if f.key != key]
        if len(remaining) == len(failures):
            return False
        self.save(remaining)
        return True

    # --- the operator's side ------------------------------------------------

    def is_dead(self, complaint: Complaint) -> bool:
        """Has this item already given up? Callers must not replay one."""
        key = delivery_key(complaint)
        return any(f.key == key and f.status == DEAD for f in self.load())

    def dead(self) -> list[Failure]:
        return [f for f in self.load() if f.status == DEAD]

    def retrying(self) -> list[Failure]:
        return [f for f in self.load() if f.status == RETRYING]

    def requeue(self, key: str) -> bool:
        """Drop a dead record so the item is retried on the next poll.

        The caller must also forget the id in the listener's seen-store, or the
        item is never fetched again; `requeue_everywhere` does both.
        """
        failures = self.load()
        remaining = [f for f in failures if f.key != key]
        if len(remaining) == len(failures):
            return False
        self.save(remaining)
        return True


DELIVERY = DeliveryLog()


def delivery_key(complaint: Complaint) -> str:
    """The same key the seen-store uses, so the two agree about identity."""
    return f"{complaint.channel.value}:{complaint.external_id}"


def requeue_everywhere(key: str) -> bool:
    """Undo a dead letter completely: forget the failure and the seen-mark.

    Both are needed. Clearing only the failure leaves the item marked seen and
    it is never fetched again; clearing only the seen-mark leaves the dead
    record behind to confuse the next person reading the queue.
    """
    from .listener import SeenStore

    removed = DELIVERY.requeue(key)
    seen = SeenStore()
    ids = seen.load()
    if key in ids:
        seen.save([i for i in ids if i != key])
        removed = True
    return removed


def main(argv: list[str]) -> int:
    if "--requeue" in argv:
        key = argv[argv.index("--requeue") + 1]
        ok = requeue_everywhere(key)
        print(f"{'requeued' if ok else 'not found'}: {key}")
        return 0 if ok else 1

    dead, retrying = DELIVERY.dead(), DELIVERY.retrying()
    if not dead and not retrying:
        print("\nNothing stuck. Every item that was fetched was handled.\n")
        return 0

    print(f"\n{'key':<28}{'attempts':<10}{'status':<10}error")
    print("-" * 96)
    for f in retrying + dead:
        print(f"{f.key:<28}{f.attempts:<10}{f.status:<10}{f.error[:44]}")
    for f in dead:
        print(f"\n  {f.key}  by {f.author}, first seen {f.first_at[:16]}")
        print(f"  “{f.excerpt}”")
        for e in f.errors:
            print(f"    {e[:100]}")
    if dead:
        print(f"\nRetry one after fixing the cause:"
              f"\n  python -m sanwaad.delivery --requeue {dead[0].key}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
