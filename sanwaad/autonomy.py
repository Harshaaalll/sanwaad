"""Autonomy the system earns, one capability at a time.

The goal is a system that runs without a human touch. The mistake is to build
that by removing the human, because an agent's first day and its thousandth
look identical from the inside and only one of them has a track record. What
actually gets you there is making autonomy a **measured property of a specific
capability** rather than a global switch someone flips.

So every capability starts supervised and climbs:

    SHADOW      it decides and records; a person does the work
    ASSISTED    a person approves before it acts
    SUPERVISED  it acts, tells a person, and stays reversible for a window
    AUTONOMOUS  it acts and reports in aggregate

A capability moves up by agreeing with people often enough, over enough
decisions, recently enough — and falls the moment it stops. That is the whole
mechanism, and it has a property a fixed policy cannot have: the day a model,
a prompt or a policy clause changes, agreement drops and authority contracts
on its own, before anyone notices the regression.

## Why the ledger is fed by work that was happening anyway

Every human review already produces the pair this needs: what the system
proposed, and what the person actually sent. Approving unchanged is agreement.
Editing or rejecting is not. Nothing extra is asked of the reviewer, which
matters — a promotion scheme that needs its own labelling effort is one that
stops being fed in month two.

## The line this does not cross

A capability's level is capped by what its tools allow, and a tool that is not
`auto_approvable` can never be executed without a person no matter how good the
record is. That is `initiate_reversal`: moving someone's money. The cap is not
timidity, it is the argument the rest of the system is built on — code
validates, a person approves the exact arguments, the executor re-checks. A
track record is evidence about the *common* case, and an irreversible transfer
of someone else's money is not the case you want to be average about.

Everything else — replying, ticketing, triaging, routing, calling back — is
reachable all the way to AUTONOMOUS, which is most of the work.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

from .config import DATA_DIR

AUTONOMY_PATH = DATA_DIR / "autonomy.jsonl"


class Level(IntEnum):
    SHADOW = 0
    ASSISTED = 1
    SUPERVISED = 2
    AUTONOMOUS = 3

    @property
    def acts_without_a_person(self) -> bool:
        return self >= Level.SUPERVISED


@dataclass(frozen=True)
class AutonomyPolicy:
    """How much evidence buys how much authority.

    The numbers are product decisions, so they live together where someone can
    argue with them rather than scattered through the scoring.
    """

    window_days: int = 30
    # Recent decisions required before a capability may leave ASSISTED. Twenty
    # is small enough to reach in a pilot and large enough that one good
    # afternoon is not a track record.
    min_decisions: int = 20
    # ...and twice that before it may act unannounced.
    min_for_autonomous: int = 40
    promote_at: float = 0.95
    demote_at: float = 0.80
    # A disagreement on a decision marked consequential stops the clock: the
    # capability returns to ASSISTED until it has earned its way back. Rare
    # expensive mistakes are exactly what an average hides.
    consequential_cooloff_days: int = 7


POLICY = AutonomyPolicy()

# Turning this off pins every capability at ASSISTED. A pilot customer who
# wants nothing automatic gets that by setting one variable, rather than by
# being told the system cannot do what it was bought for.
ENABLED = os.getenv("SANWAAD_AUTONOMY", "on").lower() not in ("0", "off", "false")


@dataclass
class Verdict:
    capability: str
    level: Level
    reason: str
    decisions: int
    agreement: Optional[float]
    ceiling: Level

    @property
    def acts_without_a_person(self) -> bool:
        return self.level.acts_without_a_person


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class AutonomyLedger:
    """What each capability has done, and what a person made of it.

    Append-only, because the question it answers is historical and because a
    file that is only ever appended to cannot be corrupted by a crash halfway
    through a promotion.
    """

    def __init__(self, path: Optional[Path] = None, policy: Optional[AutonomyPolicy] = None):
        # Resolved here rather than bound as a default, so redirecting
        # AUTONOMY_PATH — the eval harness does — is actually seen.
        self.path = path or AUTONOMY_PATH
        self.policy = policy or POLICY

    # --- writing ------------------------------------------------------------

    def record(self, capability: str, *, agreed: bool, consequential: bool = False,
               case_id: str = "", note: str = "", at: Optional[datetime] = None) -> None:
        """One decision, and whether the person agreed with it."""
        row = {
            "at": (at or _now()).isoformat(),
            "capability": capability,
            "agreed": bool(agreed),
            "consequential": bool(consequential),
            "case_id": case_id,
            "note": note[:200],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Never let bookkeeping fail a case. The cost is one decision's
            # worth of evidence, and the direction of that error is safe:
            # missing evidence can only slow a promotion, never cause one.
            logger.warning(f"autonomy ledger write failed: {exc}")

    # --- reading ------------------------------------------------------------

    def rows(self, capability: Optional[str] = None,
             since: Optional[datetime] = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if capability and row.get("capability") != capability:
                continue
            at = _parse(row.get("at", ""))
            if since and (at is None or at < since):
                continue
            out.append(row)
        return out

    def verdict(self, capability: str, *, ceiling: Level = Level.AUTONOMOUS,
                now: Optional[datetime] = None) -> Verdict:
        """How much authority this capability has earned, capped by `ceiling`."""
        now = now or _now()
        p = self.policy

        if not ENABLED:
            return Verdict(capability, min(Level.ASSISTED, ceiling),
                           "autonomy is switched off (SANWAAD_AUTONOMY)", 0, None, ceiling)

        rows = self.rows(capability, since=now - timedelta(days=p.window_days))
        n = len(rows)
        if n == 0:
            return Verdict(capability, min(Level.ASSISTED, ceiling),
                           "no track record yet; a person decides", 0, None, ceiling)

        agreement = sum(1 for r in rows if r.get("agreed")) / n

        # A recent consequential disagreement outranks the average. An average
        # is exactly where a rare expensive mistake hides.
        cutoff = now - timedelta(days=p.consequential_cooloff_days)
        recent_bad = [r for r in rows
                      if r.get("consequential") and not r.get("agreed")
                      and (_parse(r.get("at", "")) or now) >= cutoff]
        if recent_bad:
            return Verdict(capability, min(Level.ASSISTED, ceiling),
                           f"a consequential disagreement {len(recent_bad)} time(s) "
                           f"in the last {p.consequential_cooloff_days} days",
                           n, agreement, ceiling)

        if agreement < p.demote_at:
            level, why = Level.SHADOW, f"agreement {agreement:.0%} below {p.demote_at:.0%}"
        elif agreement < p.promote_at or n < p.min_decisions:
            level, why = Level.ASSISTED, (
                f"agreement {agreement:.0%} over {n} decisions; needs "
                f"{p.promote_at:.0%} over {p.min_decisions}")
        elif n < p.min_for_autonomous:
            level, why = Level.SUPERVISED, (
                f"agreement {agreement:.0%} over {n} decisions; acts and tells someone")
        else:
            level, why = Level.AUTONOMOUS, (
                f"agreement {agreement:.0%} over {n} decisions")

        capped = min(level, ceiling)
        if capped < level:
            why += f"; capped at {capped.name} by what this capability's tools allow"
        return Verdict(capability, capped, why, n, agreement, ceiling)

    def report(self, now: Optional[datetime] = None) -> list[dict]:
        """Every capability and where it stands, for an operator."""
        names = sorted({r.get("capability", "") for r in self.rows()} - {""})
        out = []
        for name in names:
            v = self.verdict(name, now=now)
            out.append({"capability": name, "level": v.level.name,
                        "decisions": v.decisions,
                        "agreement": round(v.agreement, 3) if v.agreement is not None else None,
                        "reason": v.reason})
        return out


LEDGER = AutonomyLedger()


def ceiling_for(tools: Iterable[str]) -> Level:
    """The highest level a capability using these tools may ever reach.

    The rule is the registry's own, not a second one beside it: the registry
    demands an approval for a WRITE_HIGH tool, and `auto_approvable` says
    whether a policy gate may stand in for the person giving it. A WRITE_HIGH
    tool that is not auto-approvable therefore needs a human by contract, and
    no track record buys past a contract. Reads and low-risk writes were never
    gated on a person in the first place.
    """
    from .tools import REGISTRY, Risk

    for name in tools:
        spec = REGISTRY.get(name)
        if spec is not None and spec.risk is Risk.WRITE_HIGH and not spec.auto_approvable:
            return Level.ASSISTED
    return Level.AUTONOMOUS


def main(argv: list[str]) -> int:
    """`python -m sanwaad.autonomy` — what the system has earned the right to do."""
    rows = LEDGER.report()
    if not rows:
        print("\nNo decisions recorded yet. Every capability is ASSISTED: a person"
              "\ndecides, and the system learns from what they decide.\n")
        return 0
    print(f"\n{'capability':<28}{'level':<13}{'decisions':<11}{'agreement':<11}why")
    print("-" * 100)
    for r in rows:
        agreement = f"{r['agreement']:.0%}" if r["agreement"] is not None else "—"
        print(f"{r['capability']:<28}{r['level']:<13}{r['decisions']:<11}"
              f"{agreement:<11}{r['reason'][:44]}")
    print()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
