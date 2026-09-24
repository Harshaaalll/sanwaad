"""Declared missions: the same machinery, pointed at a different job.

Sanwaad's case graph handles complaints, and its shape is baked into
`graph/graph.py` because that shape was known in advance. A lead pipeline has
the same bones and a different route — qualify, then enrich or disqualify,
then follow up, then close — and rebuilding the harness for it would be the
mistake. Everything a second workflow needs already exists: typed tools with a
risk ladder, a circuit breaker, bounded concurrency, a cost ceiling, traces, a
dead-letter queue, and authority that is earned per capability.

What was missing was a way to say what the route *is* without writing edges.
That is `handoff.py`, and this file is the proof it generalises: a mission is
a declaration, the agents propose where work goes next, and code decides
whether they may.

    python -m sanwaad.missions        run the lead pipeline over sample leads

## The handlers here are rules, deliberately

Every decision below is a rule, not a model call, for the same reason the
judge agent is: a qualification threshold is a business decision that should
live where someone can argue with it. Swapping any handler for a model call is
a one-line change, and the routing stays exactly as safe either way — which is
the point of validating the route rather than trusting it.
"""

from __future__ import annotations

import sys

from .handoff import Done, Handoff, Mission, Step, run_mission

# ---------------------------------------------------------------------------
# The lead pipeline
# ---------------------------------------------------------------------------

LEAD_PIPELINE = Mission(
    name="lead",
    entry="qualify",
    steps=(
        Step(
            agent="qualify",
            does="Decide whether this lead is worth anyone's time, and how urgently",
            expects=("company", "seats"),
            hands_to=("enrich", "nurture", "disqualified"),
        ),
        Step(
            agent="enrich",
            does="Fill in what the qualifier needed and did not have",
            expects=("company",),
            hands_to=("reach_out", "nurture"),
            tools=("search_policy",),
        ),
        Step(
            agent="reach_out",
            does="Make contact, in the channel the lead came from",
            expects=("company", "contact"),
            hands_to=("booked", "nurture"),
            # post_reply is WRITE_HIGH but auto-approvable, so this step can
            # earn its way to acting alone. A step that moved money could not.
            tools=("post_reply",),
        ),
        Step(agent="nurture", does="Not now: park it with a reason and a date",
             terminal=True),
        Step(agent="booked", does="A person is meeting them", terminal=True),
        Step(agent="disqualified", does="Not a fit, recorded so it is not re-worked",
             terminal=True),
    ),
).check()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

BIG_ENOUGH = 25          # seats below which we do not staff a human
HOT = 200                # seats above which it skips the queue


async def qualify(payload: dict):
    seats = int(payload.get("seats") or 0)
    if seats < BIG_ENOUGH:
        return Done("disqualified", {"why": f"{seats} seats is below {BIG_ENOUGH}"})
    if not payload.get("contact"):
        return Handoff("enrich", f"{seats} seats, but nobody to contact",
                       {"priority": "high" if seats >= HOT else "normal"})
    return Handoff("reach_out", f"{seats} seats and a named contact",
                   {"priority": "high" if seats >= HOT else "normal"})


async def enrich(payload: dict):
    # A real one would call a data provider through the tool registry, which is
    # what makes it a tool call and not a side effect. The mock stands in.
    found = _DIRECTORY.get(payload["company"].lower())
    if not found:
        return Handoff("nurture", "no contact found; nothing to reach out to", {})
    return Handoff("reach_out", f"found {found}", {"contact": found})


async def reach_out(payload: dict):
    if payload.get("priority") == "high":
        return Done("booked", {"why": f"high priority, contacted {payload['contact']}"})
    return Handoff("nurture", "contacted, no reply yet; park and follow up", {})


def _terminal(name: str):
    """A step that only records the outcome it is named for."""
    async def handler(_payload: dict):
        return Done(name, {})

    return handler


_DIRECTORY = {
    "nimbus retail": "ops@nimbusretail.example",
    "kirana connect": "founder@kiranaconnect.example",
}

LEAD_HANDLERS = {
    "qualify": qualify,
    "enrich": enrich,
    "reach_out": reach_out,
    "nurture": _terminal("nurture"),
    "booked": _terminal("booked"),
    "disqualified": _terminal("disqualified"),
}


MISSIONS = {LEAD_PIPELINE.name: LEAD_PIPELINE}
HANDLERS = {LEAD_PIPELINE.name: LEAD_HANDLERS}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

_SAMPLE = [
    {"company": "Nimbus Retail", "seats": 400},
    {"company": "Kirana Connect", "seats": 60},
    {"company": "Two Person Studio", "seats": 4},
    {"company": "Unlisted Co", "seats": 80},
]

BOLD, DIM, GREEN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[0m"


async def _demo() -> int:
    from .autonomy import LEDGER, ceiling_for

    print(f"\n{BOLD}Mission: {LEAD_PIPELINE.name}{RESET}  "
          f"{DIM}{len(LEAD_PIPELINE.steps)} steps, at most {LEAD_PIPELINE.max_hops} hops{RESET}")
    for s in LEAD_PIPELINE.steps:
        arrow = " → " + ", ".join(s.hands_to) if s.hands_to else " (terminal)"
        level = LEDGER.verdict(f"{LEAD_PIPELINE.name}.{s.agent}",
                               ceiling=ceiling_for(s.tools)).level.name
        print(f"  {s.agent:<14}{DIM}{s.does[:52]:<54}{RESET}{level:<11}{DIM}{arrow}{RESET}")

    print()
    for lead in _SAMPLE:
        run = await run_mission(LEAD_PIPELINE, LEAD_HANDLERS, lead)
        colour = GREEN if run.finished else YELLOW
        print(f"  {lead['company']:<20}{DIM}{lead['seats']:>4} seats{RESET}  "
              f"{run.path():<44} {colour}{run.outcome or run.stopped}{RESET}")
        for hop in run.trail:
            print(f"      {DIM}{hop.frm} → {hop.to}: {hop.reason}{RESET}")

    print(f"\n{DIM}A step proposes where work goes next; handoff.py decides whether it may."
          f"\nRefusals are outcomes with reasons, not exceptions.{RESET}\n")
    return 0


def main(argv: list[str]) -> int:
    import asyncio

    return asyncio.run(_demo())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
