"""Tests for agents handing work to each other without handing over control.

A self-routing multi-agent system fails in a small number of specific ways:
two agents pass work back and forth until something dies, an agent sends work
somewhere it was never meant to go, a step receives work it cannot act on, and
the reason any of it happened exists only in a transcript.

Every one of those is a refusal here, so most of these tests are about a
handoff *not* happening. The ones that matter are the ones where the handler
proposes something reasonable-looking and the system says no anyway — that is
the whole difference between validating a route and trusting one.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.handoff import (
    Done,
    Handoff,
    Mission,
    MissionError,
    Step,
    refusal_for,
    run_mission,
)

TINY = Mission(
    name="tiny",
    entry="a",
    steps=(
        Step("a", "first", hands_to=("b", "end")),
        Step("b", "second", expects=("note",), hands_to=("a", "end")),
        Step("end", "finished", terminal=True),
    ),
    max_hops=4,
).check()


def _handlers(**overrides):
    async def finish(_p):
        return Done("finished", {})

    base = {"a": finish, "b": finish, "end": finish}
    base.update(overrides)
    return base


def _always(result):
    async def handler(_payload):
        return result

    return handler


# --- the declaration is checked, not trusted --------------------------------

def test_a_handoff_to_a_step_that_does_not_exist_fails_at_declaration():
    """A typo in a routing table should fail on import, not when a lead hits
    that branch at two in the morning."""
    with pytest.raises(MissionError, match="unknown"):
        Mission("m", "a", (Step("a", "x", hands_to=("typo",)),
                           Step("end", "e", terminal=True))).check()


def test_a_step_nobody_can_reach_fails_at_declaration():
    with pytest.raises(MissionError, match="cannot be reached"):
        Mission("m", "a", (Step("a", "x", hands_to=("end",)),
                           Step("orphan", "never", hands_to=("end",)),
                           Step("end", "e", terminal=True))).check()


def test_a_mission_that_cannot_finish_fails_at_declaration():
    with pytest.raises(MissionError, match="no terminal step"):
        Mission("m", "a", (Step("a", "x", hands_to=("b",)),
                           Step("b", "y", hands_to=("a",)))).check()


def test_a_dead_end_that_is_not_terminal_fails_at_declaration():
    """A step with nowhere to send work and no terminal flag swallows whatever
    reaches it, which is a routing table that lies about being one."""
    with pytest.raises(MissionError, match="hands to nobody"):
        Mission("m", "a", (Step("a", "x", hands_to=("b",)),
                           Step("b", "dead end"),
                           Step("end", "e", terminal=True))).check()


# --- refusals ---------------------------------------------------------------

def test_an_agent_may_not_invent_a_route():
    """`a` may hand to `b` or `end`. Proposing anything else is the model
    deciding control flow, which is the thing this exists to prevent."""
    refusal = refusal_for(TINY, "a", Handoff("nowhere", "because"), ["a"])
    assert "not a step" in refusal


def test_an_agent_may_not_use_a_route_it_was_not_given():
    """`end` exists and `b` is real, but `b -> a -> b` is not declared from
    every direction. An undeclared edge between two real steps is the subtle
    version of the same mistake."""
    narrow = Mission("n", "a", (Step("a", "x", hands_to=("end",)),
                                Step("b", "y", hands_to=("end",)),
                                Step("end", "e", terminal=True)))
    # `a` may reach `end` but was never given `b`.
    assert "may not hand to b" in refusal_for(narrow, "a", Handoff("b", "why"), ["a"])


def test_a_step_does_not_receive_work_it_cannot_act_on():
    """`b` needs a note. Handing it work without one means discovering the gap
    inside b's own logic, which is where a missing field turns into a guess."""
    refusal = refusal_for(TINY, "a", Handoff("b", "over to you", {}), ["a"])
    assert "needs ['note']" in refusal


def test_a_step_may_not_be_revisited():
    """Two agents passing work back and forth is the multi-agent failure mode.
    `b` is allowed to hand back to `a` by the table, and still may not."""
    refusal = refusal_for(TINY, "b", Handoff("a", "your turn again", {"note": "n"}), ["a", "b"])
    assert "already seen this work" in refusal


def test_a_chain_stops_at_the_mission_limit():
    long_trail = ["a", "b", "c", "d"]
    refusal = refusal_for(TINY, "a", Handoff("end", "done", {}), long_trail)
    assert "its limit" in refusal


def test_a_legal_handoff_is_not_refused():
    assert refusal_for(TINY, "a", Handoff("b", "over to you", {"note": "n"}), ["a"]) is None


# --- running ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_refused_handoff_stops_the_mission_with_a_reason():
    """A refusal is an outcome, not an exception: the mission stopped, the
    trail says where, and a person can read why."""
    run = await run_mission(TINY, _handlers(a=_always(Handoff("nowhere", "guessing"))), {})
    assert run.stopped == "refused"
    assert "not a step" in run.detail
    assert run.visited == ["a"]
    assert not run.finished


@pytest.mark.asyncio
async def test_the_trail_records_why_each_hop_happened():
    """The reason a piece of work ended up somewhere has to outlive the run,
    or the only answer to "why is this lead here" is a transcript."""
    run = await run_mission(
        TINY,
        _handlers(a=_always(Handoff("b", "needs a second look", {"note": "n"}))),
        {})
    assert run.finished
    assert run.path() == "a → b"
    assert [(h.frm, h.to, h.reason) for h in run.trail] == [
        ("a", "b", "needs a second look")]


@pytest.mark.asyncio
async def test_payload_accumulates_along_the_chain():
    """What one agent learned is what the next one acts on."""
    run = await run_mission(
        TINY,
        _handlers(a=_always(Handoff("b", "enriched", {"note": "n", "found": "x"}))),
        {"lead": "acme"})
    assert run.payload["lead"] == "acme" and run.payload["found"] == "x"


@pytest.mark.asyncio
async def test_a_handler_that_returns_nonsense_stops_the_mission():
    run = await run_mission(TINY, _handlers(a=_always("just a string")), {})
    assert run.stopped == "error"
    assert "not a handoff" in run.detail


@pytest.mark.asyncio
async def test_a_bouncing_pair_cannot_run_forever():
    """The failure this is all for. Two handlers that each insist on the other
    are stopped by the revisit rule, at the first bounce."""
    run = await run_mission(
        TINY,
        {"a": _always(Handoff("b", "you take it", {"note": "n"})),
         "b": _always(Handoff("a", "no, you take it", {"note": "n"})),
         "end": _always(Done("finished", {}))},
        {})
    assert run.stopped == "refused"
    assert "already seen this work" in run.detail
    assert run.visited == ["a", "b"]


# --- the declared lead pipeline ---------------------------------------------

@pytest.mark.asyncio
async def test_the_lead_pipeline_routes_by_what_the_qualifier_found():
    """The point of the whole thing: one declaration, and the route depends on
    what an agent discovered rather than on an edge someone drew."""
    from sanwaad.missions import LEAD_HANDLERS, LEAD_PIPELINE

    big = await run_mission(LEAD_PIPELINE, LEAD_HANDLERS,
                            {"company": "Nimbus Retail", "seats": 400})
    assert big.path() == "qualify → enrich → reach_out"
    assert big.outcome == "booked"

    small = await run_mission(LEAD_PIPELINE, LEAD_HANDLERS,
                              {"company": "Two Person Studio", "seats": 4})
    assert small.path() == "qualify"
    assert small.outcome == "disqualified"

    unknown = await run_mission(LEAD_PIPELINE, LEAD_HANDLERS,
                                {"company": "Unlisted Co", "seats": 80})
    assert unknown.path() == "qualify → enrich → nurture"


@pytest.mark.asyncio
async def test_work_that_arrives_without_what_the_entry_step_needs_is_refused():
    from sanwaad.missions import LEAD_HANDLERS, LEAD_PIPELINE

    run = await run_mission(LEAD_PIPELINE, LEAD_HANDLERS, {"company": "No Seats Co"})
    assert run.stopped == "refused"
    assert "seats" in run.detail


def test_a_step_that_moves_money_could_not_run_itself():
    """The mission layer inherits the same limit as everything else: a step's
    autonomy is capped by its tools, so one wired to initiate_reversal stays
    ASSISTED whatever its record."""
    from sanwaad.autonomy import Level, ceiling_for
    from sanwaad.missions import LEAD_PIPELINE

    reach_out = LEAD_PIPELINE.step("reach_out")
    assert ceiling_for(reach_out.tools) is Level.AUTONOMOUS
    assert ceiling_for(("initiate_reversal",)) is Level.ASSISTED
