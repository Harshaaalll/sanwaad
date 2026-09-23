"""The listener: hear the complaints that were never addressed to you.

The comments that damage a brand are almost never the ones in the mentions
tab. Someone writes "this app just ate ₹4,500" under a stranger's post, gets
forty upvotes, and no notification is ever generated — because the brand was
described, not tagged. A support queue built on mentions is a queue built on
the subset of angry customers who were polite enough to address you directly.

So the listener does two things a webhook does not:

1.  It searches for the brand being *named* as well as tagged, and marks which
    happened. Untagged is not a lesser signal; it is the normal one, and it is
    the one nobody else is watching.
2.  It remembers what it has already seen, across restarts. Every connector
    returns overlapping pages on every poll, and a queue that re-opens a case
    each time is worse than no queue — it double-replies, which is the one
    public mistake a support account cannot take back.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional

from loguru import logger

from .config import DATA_DIR
from .connectors import get_connector
from .delivery import DELIVERY, DeliveryLog
from .models import Complaint

SEEN_PATH = DATA_DIR / "listener_seen.json"

# The brand, and the ways people actually write it. Misspellings matter more
# than the canonical form: nobody checks their spelling while angry.
BRAND = os.getenv("SANWAAD_BRAND", "NimbusPay")
BRAND_ALIASES = tuple(
    a.strip() for a in os.getenv(
        "SANWAAD_BRAND_ALIASES", "NimbusPay,Nimbus Pay,Nimbuspay,nimbus,निंबस"
    ).split(",") if a.strip()
)


def _alias_pattern() -> re.Pattern:
    parts = sorted({re.escape(a) for a in (BRAND, *BRAND_ALIASES)}, key=len, reverse=True)
    return re.compile(r"(?<![\w@/])(" + "|".join(parts) + r")\b", re.IGNORECASE)


_MENTION = _alias_pattern()
_TAGGED = re.compile(
    r"(?:@|u/|r/)\s?(" + "|".join(re.escape(a).replace(r"\ ", r"\s?")
                                  for a in (BRAND, *BRAND_ALIASES)) + r")\b",
    re.IGNORECASE,
)


def is_tagged(text: str) -> bool:
    """Did they actually address us, or just talk about us?"""
    return bool(_TAGGED.search(text or ""))


def mentions_brand(text: str) -> bool:
    return bool(_MENTION.search(text or "") or is_tagged(text or ""))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class SeenStore:
    """Ids already handed downstream. Small, boring, and load-bearing."""

    def __init__(self, path=None, max_ids: int = 5000):
        # Resolved here rather than as a default argument, so that patching
        # SEEN_PATH — which the eval harness and the tests do — is actually
        # seen by a store built afterwards.
        self.path = path or SEEN_PATH
        self.max_ids = max_ids

    def load(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            return list(json.loads(self.path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            # Failing closed here would replay the whole feed and double-reply
            # in public. Failing open costs us one poll's worth of memory.
            logger.warning("listener seen-store unreadable; treating as empty")
            return []

    def save(self, ids: Iterable[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(list(ids)[-self.max_ids:]), encoding="utf-8")

    def filter_new(self, complaints: list[Complaint]) -> list[Complaint]:
        seen = set(self.load())
        fresh = [c for c in complaints if _key(c) not in seen]
        return fresh

    def mark(self, complaints: list[Complaint]) -> None:
        ids = self.load()
        known = set(ids)
        for c in complaints:
            k = _key(c)
            if k not in known:
                known.add(k)
                ids.append(k)
        self.save(ids)


def _key(c: Complaint) -> str:
    return f"{c.channel.value}:{c.external_id}"


# ---------------------------------------------------------------------------
# Listening
# ---------------------------------------------------------------------------


@dataclass
class Heard:
    complaints: list[Complaint]
    skipped_seen: int
    skipped_unrelated: int

    @property
    def untagged(self) -> int:
        return sum(1 for c in self.complaints if not c.tagged)


# Channels that are already ours: our Play Store listing, our own support
# feed. Everything posted there is about us by construction, and demanding the
# brand name would discard every review that just says "this app". The name
# filter is only meaningful on the open web, where we are one subject among
# millions.
OWNED_CHANNELS = frozenset({"mock", "playstore"})


class Listener:
    """Poll every configured channel and emit what is new and about us."""

    def __init__(self, channels: Optional[list[str]] = None,
                 seen: Optional[SeenStore] = None,
                 owned: Optional[frozenset] = None,
                 delivery: Optional["DeliveryLog"] = None):
        self.channels = channels or [
            c.strip() for c in os.getenv("SANWAAD_CHANNELS", "mock").split(",") if c.strip()
        ]
        self.seen = seen or SeenStore()
        self.owned = OWNED_CHANNELS if owned is None else owned
        self.delivery = delivery or DELIVERY

    def needs_mention(self, channel: str) -> bool:
        return channel not in self.owned

    async def poll(self, limit: int = 20) -> Heard:
        gathered: list[Complaint] = []
        for channel in self.channels:
            try:
                connector = get_connector(channel)
                gathered.extend(await connector.fetch(limit=limit))
            except Exception as exc:
                # One dead channel must not stop the others. A listener that
                # goes silent because Reddit rate-limited us is a listener that
                # misses the Play Store review that mattered.
                logger.warning(f"listener: channel {channel!r} failed: {exc}")

        related, unrelated = [], 0
        for c in gathered:
            body = f"{c.text}\n{c.parent_text or ''}"
            if self.needs_mention(c.channel.value) and not mentions_brand(body):
                unrelated += 1
                continue
            # On an owned feed everything is addressed to us in effect, so
            # `tagged` records only whether they actually wrote the name — the
            # untagged count stays an honest measure of what a mentions-based
            # queue would have missed.
            c.tagged = is_tagged(body)
            related.append(c)

        fresh = self.seen.filter_new(related)
        return Heard(complaints=fresh,
                     skipped_seen=len(related) - len(fresh),
                     skipped_unrelated=unrelated)

    async def watch(self, handler: Callable[[Complaint], Awaitable[None]],
                    interval_s: float = 60.0, limit: int = 20,
                    max_cycles: Optional[int] = None) -> None:
        """Poll forever, handing each new complaint to `handler`.

        An item is marked seen only after its handler returns. A crash between
        fetching and handling should replay the item, not lose it — a duplicate
        reply is embarrassing, but a complaint that silently evaporated is the
        failure this whole system exists to prevent, so the tie goes to
        replaying.

        Replaying needs a floor, though. An item that fails for a reason that
        will never clear is otherwise refetched every cycle forever, spending a
        whole graph run each time and leaving nothing behind but another
        identical log line. After `max_attempts` it is declared dead: written
        to the delivery log with the errors that killed it, and marked seen so
        the loop stops paying for it. Dead is not lost — `python -m
        sanwaad.delivery` lists it with the customer's words and the error, and
        `--requeue` puts it back once the cause is fixed.
        """
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            heard = await self.poll(limit=limit)
            if heard.complaints:
                logger.info(
                    f"listener: {len(heard.complaints)} new "
                    f"({heard.untagged} untagged), {heard.skipped_seen} already seen"
                )
            for complaint in heard.complaints:
                try:
                    await handler(complaint)
                    self.seen.mark([complaint])
                    self.delivery.clear(complaint)
                except Exception as exc:
                    failure = self.delivery.record_failure(complaint, exc)
                    if failure.dead:
                        self.seen.mark([complaint])
                        logger.error(
                            f"listener: {complaint.external_id} dead-lettered after "
                            f"{failure.attempts} attempts: {exc}")
                    else:
                        logger.warning(
                            f"listener: handler failed for {complaint.external_id} "
                            f"(attempt {failure.attempts}/{self.delivery.max_attempts}, "
                            f"will retry): {exc}")
            cycles += 1
            if max_cycles is None or cycles < max_cycles:
                await asyncio.sleep(interval_s)
