"""Tests for the listener, the judge and the pattern agent.

Same discipline as the rest of the suite: nothing here asserts on model output.
What is pinned is the judgement each agent makes from signals — who counts as a
troll, when a set of complaints becomes an incident, what gets answered anyway.
Those are product decisions, and a product decision that is not tested is a
product decision that changes by accident.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.graph.nodes import auto_post_allowed, prioritise
from sanwaad.judge import assess, needs_a_second_opinion
from sanwaad.listener import Listener, SeenStore, is_tagged, mentions_brand
from sanwaad.models import AuthorMeta, Channel, Complaint
from sanwaad.pattern import (
    Fingerprint,
    PatternStore,
    assess_cluster,
    coordinated_authors,
)

NOW = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def _complaint(text, author="u/someone", **meta):
    return Complaint(external_id="x1", channel=Channel.MOCK, author=author,
                     text=text, author_meta=AuthorMeta(**meta))


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

TROLL_TEXT = ("This app is a SCAM, they hang the site on purpose just to make "
              "hype. Fake company, total loot. BOYCOTT everyone.")
CUSTOMER_TEXT = ("₹4,500 debited 6 days ago for my Swiggy order and never "
                 "reached the merchant. Transaction id 88AB4471X.")


def test_a_fresh_account_shouting_scam_is_not_worth_a_drafted_reply():
    """The case the pipeline used to answer with a careful grounded reply."""
    v = assess(_complaint(TROLL_TEXT, account_age_days=0, karma=1, post_count=1))
    assert v.author_class == "troll"
    assert not v.reply_worthy


def test_the_same_accusation_from_a_large_account_is_still_answered():
    """Reach outranks suspicion. An unanswered post read by sixty thousand
    people becomes the fact of the matter, whoever wrote it."""
    v = assess(_complaint(TROLL_TEXT, account_age_days=2400, followers=61000,
                          verified=True))
    assert v.reach == 61000
    assert v.reply_worthy


def test_concrete_detail_outweighs_a_young_account():
    """A new account describing a transaction it could only know about if it
    happened is a customer, not a throwaway."""
    v = assess(_complaint(CUSTOMER_TEXT, account_age_days=9, karma=2))
    assert v.author_class == "customer"
    assert v.reply_worthy


def test_anger_alone_is_never_held_against_a_complainant():
    """Indian customers are direct, and a person who has lost money is
    entitled to be furious. Only anger with nothing behind it counts."""
    furious = assess(_complaint("This is ABSOLUTELY UNACCEPTABLE. " + CUSTOMER_TEXT,
                                account_age_days=800, karma=900))
    polite = assess(_complaint(CUSTOMER_TEXT, account_age_days=800, karma=900))
    assert furious.reply_worthy and polite.reply_worthy
    assert furious.author_class == polite.author_class == "customer"


def test_detail_cannot_be_stacked_to_out_score_a_terse_complaint():
    """A verbose account naming five things must not beat a customer who
    named one, or the score becomes a writing-length contest."""
    verbose = assess(_complaint(
        "₹500 ₹600 ₹700 order id ABCD1234 for 3 days on Swiggy and Zomato, "
        "screenshot and receipt attached", account_age_days=400))
    terse = assess(_complaint("₹500 gone", account_age_days=400))
    assert verbose.authenticity - terse.authenticity <= 0.4


def test_copy_paste_from_several_accounts_reads_as_a_campaign():
    v = assess(_complaint(TROLL_TEXT, account_age_days=30, karma=40),
               duplicate_authors=4)
    assert v.author_class == "bot"
    assert not v.reply_worthy


def test_a_bot_is_never_answered_however_large_its_following():
    """The one case where reach does not buy a reply: replying to a botnet
    is what a botnet is for."""
    v = assess(_complaint(TROLL_TEXT, account_age_days=30, followers=90000),
               duplicate_authors=5)
    assert v.author_class == "bot"
    assert not v.reply_worthy


def test_only_the_ambiguous_middle_is_worth_a_model_call():
    confident_troll = assess(_complaint(TROLL_TEXT, account_age_days=0, karma=0,
                                        post_count=1))
    confident_customer = assess(_complaint(CUSTOMER_TEXT, account_age_days=900,
                                           karma=3000))
    assert not needs_a_second_opinion(confident_troll)
    assert not needs_a_second_opinion(confident_customer)


def test_missing_platform_metadata_is_not_evidence_of_anything():
    """Most platforms tell us almost nothing. A silent connector must not
    look like a suspicious account."""
    bare = assess(_complaint("The refund never arrived."))
    assert bare.author_class != "troll"
    assert bare.reply_worthy


# ---------------------------------------------------------------------------
# The pattern agent
# ---------------------------------------------------------------------------

# Hand-built unit vectors: cos([1,0],[0.8,0.6]) == 0.8, above the 0.72 floor;
# cos([1,0],[0,1]) == 0. Keeping the fixtures arithmetic rather than embedded
# means these tests pin the thresholds, not the embedding model.
SAME = [0.8, 0.6]
ALSO_SAME = [0.9, 0.435889894]
DIFFERENT = [0.0, 1.0]
QUERY = [1.0, 0.0]


def _fp(author, minutes_ago, vector=SAME, case=None, text=""):
    return Fingerprint(
        case_id=case or f"case_{author}_{minutes_ago}",
        author=author,
        category="service_outage",
        summary="payments failing",
        at=(NOW - timedelta(minutes=minutes_ago)).isoformat(),
        vector=vector,
        text=text,
    )


def test_one_angry_person_posting_six_times_is_one_angry_person():
    """The failure mode of every naive volume alert: it fires on whoever is
    loudest, not on whatever is broken."""
    neighbours = [_fp("u/same_guy", m, case=f"c{m}") for m in (1, 2, 3, 4, 5, 6)]
    signal = assess_cluster(vector=QUERY, author="u/same_guy", at=NOW,
                            neighbours=neighbours)
    assert signal.cluster_size == 1
    assert signal.level == "none"


def test_six_different_people_in_twelve_minutes_is_a_crisis():
    neighbours = [_fp(f"u/person{i}", i * 2) for i in range(1, 6)]
    signal = assess_cluster(vector=QUERY, author="u/person6", at=NOW,
                            neighbours=neighbours)
    assert signal.cluster_size == 6
    assert signal.level == "crisis"


def test_the_same_six_spread_across_the_day_is_not():
    """Volume without velocity is a backlog. Only the window is different."""
    neighbours = [_fp(f"u/person{i}", i * 400) for i in range(1, 6)]
    signal = assess_cluster(vector=QUERY, author="u/person6", at=NOW,
                            neighbours=neighbours, crisis_cluster=99)
    assert signal.level == "watch"


def test_three_people_inside_five_minutes_already_trips_the_alarm():
    """The whole point is to fire before the sixth person arrives."""
    neighbours = [_fp("u/a", 2), _fp("u/b", 4)]
    signal = assess_cluster(vector=QUERY, author="u/c", at=NOW, neighbours=neighbours)
    assert signal.cluster_size == 3
    assert signal.level == "crisis"


def test_unrelated_complaints_do_not_form_a_cluster():
    neighbours = [_fp(f"u/person{i}", i, vector=DIFFERENT) for i in range(1, 8)]
    signal = assess_cluster(vector=QUERY, author="u/me", at=NOW, neighbours=neighbours)
    assert signal.cluster_size == 1
    assert signal.level == "none"


def test_near_misses_on_the_similarity_floor_are_excluded():
    """0.8 is in, 0.0 is out, and the boundary is the configured number rather
    than whatever the embedding happened to produce."""
    signal = assess_cluster(vector=QUERY, author="u/me", at=NOW,
                            neighbours=[_fp("u/a", 1, vector=SAME)],
                            similarity=0.9)
    assert signal.cluster_size == 1


def test_a_lone_complaint_has_no_velocity():
    signal = assess_cluster(vector=QUERY, author="u/me", at=NOW, neighbours=[])
    assert signal.velocity_per_hour == 0.0
    assert signal.level == "none"


def test_a_fingerprint_with_no_vector_never_matches():
    """A write that failed mid-embedding must degrade to 'not similar', not
    take the next live case down with it."""
    signal = assess_cluster(vector=QUERY, author="u/me", at=NOW,
                            neighbours=[_fp("u/a", 1, vector=[])])
    assert signal.cluster_size == 1


def test_coordination_counts_accounts_not_posts():
    line = "This app is a scam, boycott it everyone"
    neighbours = [
        _fp("u/a", 1, text=line), _fp("u/a", 2, text=line),
        _fp("u/b", 3, text=line + " !!"),
    ]
    assert coordinated_authors(line, "u/me", neighbours) == 2


def test_differently_worded_reports_are_not_coordination():
    """Nine people describing one outage in nine ways is an incident. The
    lexical check must not confuse the two."""
    neighbours = [
        _fp("u/a", 1, text="Payment failed, ₹890 stuck in pending"),
        _fp("u/b", 2, text="Is the app down? Nothing is going through"),
        _fp("u/c", 3, text="निंबस से पेमेंट फेल हो रहा है"),
    ]
    assert coordinated_authors("UPI transfer keeps failing at the last step",
                               "u/me", neighbours) == 0


# ---------------------------------------------------------------------------
# The pattern store
# ---------------------------------------------------------------------------

def test_the_case_asking_the_question_is_not_its_own_history(tmp_path):
    """The pattern agent writes before the judge reads, so a naive count finds
    the current case and reports every first-time poster as a returning
    customer. This is the bug the first end-to-end run surfaced."""
    store = PatternStore(path=tmp_path / "memory.json")
    store.add(_fp("u/first_timer", 0, case="case_now"))
    assert store.author_history("u/first_timer", exclude_case_id="case_now") == 0
    assert store.author_history("u/first_timer") == 1


def test_the_window_excludes_what_fell_out_of_it(tmp_path):
    store = PatternStore(path=tmp_path / "memory.json")
    store.add(_fp("u/recent", 10, case="c1"))
    store.add(_fp("u/old", 300, case="c2"))
    recent = store.recent(within_minutes=90, now=NOW)
    assert [fp.author for fp in recent] == ["u/recent"]


def test_a_corrupt_window_does_not_fail_a_live_case(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert PatternStore(path=path).load() == []


def test_the_store_stays_bounded(tmp_path):
    store = PatternStore(path=tmp_path / "memory.json", max_history=10)
    for i in range(25):
        store.add(_fp(f"u/p{i}", 1, case=f"c{i}"))
    assert len(store.load()) == 10


# ---------------------------------------------------------------------------
# Priority — the three readings folded together
# ---------------------------------------------------------------------------

CREDIBLE = {"authenticity": 0.9, "reach": 50, "reply_worthy": True,
            "author_class": "customer", "history_with_brand": 0}
QUIET = {"level": "none", "cluster_size": 1}
CRISIS = {"level": "crisis", "cluster_size": 7}


def test_a_crisis_outranks_a_low_individual_severity():
    """Each comment in an outage is individually a mid-severity ticket. The
    queue must not order them that way."""
    p = prioritise({"is_complaint": True, "severity": 2}, CREDIBLE, CRISIS)
    assert p.tier == "crisis"
    assert p.drafting


def test_an_unworthy_author_is_logged_rather_than_answered():
    verdict = {**CREDIBLE, "reply_worthy": False, "author_class": "troll",
               "authenticity": 0.1, "reach": 0}
    p = prioritise({"is_complaint": True, "severity": 2}, verdict, QUIET)
    assert p.tier == "ignore"
    assert not p.drafting


def test_a_large_audience_promotes_an_otherwise_routine_complaint():
    p = prioritise({"is_complaint": True, "severity": 2},
                   {**CREDIBLE, "reach": 61000}, QUIET)
    assert p.tier == "priority"


def test_a_watch_promotes_a_routine_complaint():
    p = prioritise({"is_complaint": True, "severity": 2}, CREDIBLE,
                   {"level": "watch", "cluster_size": 4})
    assert p.tier == "priority"


def test_praise_costs_one_triage_call_and_stops():
    p = prioritise({"is_complaint": False, "severity": 1}, CREDIBLE, QUIET)
    assert not p.drafting


def test_a_severity_five_troll_post_is_still_read_by_a_human():
    """Severity 5 means regulatory or safety exposure. Being unable to verify
    the author is not a reason to let that pass unseen."""
    verdict = {**CREDIBLE, "reply_worthy": True, "author_class": "unknown",
               "authenticity": 0.4}
    p = prioritise({"is_complaint": True, "severity": 5}, verdict, QUIET)
    assert p.tier == "priority"
    assert p.drafting


# ---------------------------------------------------------------------------
# The crisis gate
# ---------------------------------------------------------------------------

def test_an_incident_reply_never_goes_out_unattended():
    """Forty individually-correct auto-replies with slightly different wording
    IS the screenshot. One human sets the line during an incident."""
    state = {
        "triage": {"severity": 1, "needs_private_data": False},
        "draft": {"promises_compensation": False},
        "grounding": {"grounded": True},
        "pattern": {"level": "crisis", "cluster_size": 8},
    }
    allowed, reason = auto_post_allowed(state)
    assert not allowed
    assert "crisis" in reason


def test_a_quiet_day_still_allows_an_unattended_reply():
    state = {
        "triage": {"severity": 1, "needs_private_data": False},
        "draft": {"promises_compensation": False},
        "grounding": {"grounded": True},
        "pattern": {"level": "none", "cluster_size": 1},
    }
    assert auto_post_allowed(state)[0]


# ---------------------------------------------------------------------------
# The listener
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "NimbusPay ate my money",
    "nimbus is down again",
    "निंबस से पेमेंट फेल हो रहा है",
    "Nimbus Pay support never replies",
])
def test_the_brand_is_recognised_when_it_is_only_named(text):
    """The complaints that hurt are the ones that were never addressed to us."""
    assert mentions_brand(text)
    assert not is_tagged(text)


def test_being_tagged_is_recorded_separately_from_being_mentioned():
    assert is_tagged("@NimbusPay please help") and mentions_brand("@NimbusPay please help")


def test_an_unrelated_comment_is_not_picked_up():
    assert not mentions_brand("My PhonePe transfer failed today")


def test_an_owned_feed_needs_no_brand_name():
    """On our own Play Store listing, "this app" is us. Requiring the name
    would discard most of the reviews."""
    listener = Listener(channels=["mock"])
    assert not listener.needs_mention("mock")
    assert listener.needs_mention("reddit")


def test_the_same_comment_is_never_handed_over_twice(tmp_path):
    """Connectors return overlapping pages on every poll, and a double reply
    is the one public mistake a support account cannot take back."""
    seen = SeenStore(path=tmp_path / "seen.json")
    items = [_complaint("refund missing", author="u/a")]
    assert seen.filter_new(items) == items
    seen.mark(items)
    assert seen.filter_new(items) == []


@pytest.mark.asyncio
async def test_polling_twice_yields_nothing_the_second_time(tmp_path):
    listener = Listener(channels=["mock"], seen=SeenStore(path=tmp_path / "seen.json"))
    first = await listener.poll()
    assert first.complaints
    listener.seen.mark(first.complaints)
    second = await listener.poll()
    assert second.complaints == []
    assert second.skipped_seen == len(first.complaints)


@pytest.mark.asyncio
async def test_a_dead_channel_does_not_silence_the_others(tmp_path):
    """A listener that goes quiet because one API rate-limited us is a
    listener that misses the review that mattered."""
    listener = Listener(channels=["nonexistent_channel", "mock"],
                        seen=SeenStore(path=tmp_path / "seen.json"))
    heard = await listener.poll()
    assert heard.complaints
