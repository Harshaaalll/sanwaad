"""The judge: read *who* is speaking before deciding what it costs us.

Triage reads the complaint. It cannot tell you that the account posting it was
created ninety minutes ago and has said the same sentence under four different
launches, or that this particular person has bought from us twice and is
describing a UTR they could only have if the transaction is real. The words are
identical; what we owe them is not.

The scoring is deliberately a function rather than a model call:

- The signals are enumerable and already in hand — account age, karma, whether
  the text contains an identifier only a real customer would have. A model
  asked to weigh them adds a call, a failure mode and a latency budget to
  reach the same answer less reproducibly.
- "Why was this ignored?" is the question a brand asks when it goes wrong, and
  a list of named signals answers it. A softmax does not.

A model *is* used, once, for the genuinely ambiguous band in the middle — see
`needs_a_second_opinion`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import JUDGE
from .models import AuthorVerdict, Complaint

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    name: str
    delta: float          # pushes authenticity up (+) or down (-)
    detail: str


# A real grievance tends to carry something only the aggrieved would have.
# These are the cheapest possible proxy for "this person actually transacted".
_SPECIFICITY = (
    (re.compile(r"\b(?:utr|rrn|txn|transaction|order|ref(?:erence)?)\s*(?:id|no\.?|number)?\s*[:#]?\s*[a-z0-9]{6,}",
                re.I), "quotes a transaction or order identifier"),
    (re.compile(r"(?:₹|rs\.?|inr)\s*[\d,]+", re.I), "names a specific amount"),
    (re.compile(r"\b(\d+)\s*(?:din|days?|hafte|weeks?|months?|mahine)\b", re.I),
     "gives a concrete timeframe"),
    (re.compile(r"\b(?:swiggy|zomato|uber|ola|amazon|flipkart|irctc|paytm|phonepe|gpay)\b",
                re.I), "names the merchant involved"),
    (re.compile(r"\b(?:screenshot|proof|receipt|statement|chat\s*log)\b", re.I),
     "offers evidence"),
)

# Outrage vocabulary that costs nothing to type. On its own this proves
# nothing — plenty of genuinely wronged people are furious and unspecific — so
# it only counts against an account when nothing concrete accompanies it.
_GENERIC_OUTRAGE = re.compile(
    r"\b(scam|fraud\s*company|loot|chor|thug|fake|hype|paid\s*promo|boycott|"
    r"worst\s*(?:app|company|brand)|band\s*kar\s*do|bakwaas|ghatiya)\b",
    re.I,
)

# Claims about the brand's motives rather than about anything that happened.
_MOTIVE_CLAIM = re.compile(
    r"\b(?:purposely|deliberately|jaan\s*bujh\s*kar|intentionally|"
    r"on\s*purpose|just\s*(?:to|for)\s*(?:make\s*)?hype|marketing\s*stunt)\b",
    re.I,
)


def specificity_signals(text: str) -> list[Signal]:
    """Concrete detail is the strongest single indicator of a real grievance."""
    found = [
        Signal("specificity", JUDGE.specificity_bonus, detail)
        for pattern, detail in _SPECIFICITY
        if pattern.search(text)
    ]
    # Cap it: naming an amount and a merchant and a date is one real complaint,
    # not three, and letting detail stack lets a verbose troll out-score a
    # terse customer.
    return found[: JUDGE.max_specificity_signals]


def account_signals(complaint: Complaint) -> list[Signal]:
    meta = complaint.author_meta
    out: list[Signal] = []

    age = meta.account_age_days
    if age is not None:
        if age < JUDGE.new_account_days:
            out.append(Signal("account_age", -0.30,
                              f"account is {age} day(s) old"))
        elif age >= JUDGE.established_account_days:
            out.append(Signal("account_age", 0.15,
                              f"account is {age // 365 or 1}+ year(s) old"))

    karma = meta.karma
    if karma is not None:
        if karma <= JUDGE.throwaway_karma:
            out.append(Signal("karma", -0.20, f"karma {karma}"))
        elif karma >= JUDGE.established_karma:
            out.append(Signal("karma", 0.10, f"karma {karma}"))

    if meta.verified:
        out.append(Signal("verified", 0.15, "platform-verified account"))

    posts = meta.post_count
    if posts is not None and posts <= JUDGE.throwaway_posts:
        out.append(Signal("post_count", -0.10, f"only {posts} post(s) ever"))

    return out


def history_signal(history_with_brand: int) -> list[Signal]:
    """Someone who has raised a case with us before is, at minimum, a user.

    Capped low on purpose. A repeat complainer is a real customer we have
    already failed, which is a reason to prioritise them, not a licence to
    treat frequency as credibility.
    """
    if history_with_brand <= 0:
        return []
    return [Signal("history", min(0.10 * history_with_brand, JUDGE.max_history_bonus),
                   f"{history_with_brand} prior case(s) from this author")]


def duplication_signal(duplicate_authors: int) -> list[Signal]:
    """The same sentence from many accounts at once is a campaign, not a wave.

    Distinct from the pattern agent's job: that one counts *complaints* to spot
    an outage, this one counts *near-identical wording across accounts* to spot
    coordination. Nine people describing a failed payment in nine ways is an
    incident; nine accounts posting one sentence is astroturf.
    """
    if duplicate_authors < JUDGE.coordinated_duplicate_authors:
        return []
    return [Signal("duplication", -0.35,
                   f"near-identical text from {duplicate_authors} other accounts")]


def rhetoric_signals(text: str, has_specificity: bool) -> list[Signal]:
    out: list[Signal] = []
    if _GENERIC_OUTRAGE.search(text) and not has_specificity:
        out.append(Signal("generic_outrage", -0.25,
                          "brand-level insult with nothing concrete behind it"))
    if _MOTIVE_CLAIM.search(text) and not has_specificity:
        out.append(Signal("motive_claim", -0.15,
                          "asserts intent rather than describing an event"))
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        out.append(Signal("shouting", -0.10, "mostly capitals"))
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def reach_of(complaint: Complaint) -> int:
    """Audience size, using whatever the platform gives us.

    Followers when we have them; karma is the only proxy Reddit offers and it
    is a poor one, so it is discounted hard rather than trusted.
    """
    meta = complaint.author_meta
    if meta.followers is not None:
        return int(meta.followers)
    if meta.karma is not None and meta.karma > 0:
        return int(meta.karma // JUDGE.karma_to_reach_divisor)
    return 0


def assess(complaint: Complaint, *, history_with_brand: int = 0,
           duplicate_authors: int = 0) -> AuthorVerdict:
    """Score the author from signals alone. No network, no tokens, no model."""
    text = complaint.text
    spec = specificity_signals(text)

    signals = (
        spec
        + account_signals(complaint)
        + history_signal(history_with_brand)
        + duplication_signal(duplicate_authors)
        + rhetoric_signals(text, has_specificity=bool(spec))
    )

    authenticity = JUDGE.base_authenticity + sum(s.delta for s in signals)
    authenticity = max(0.0, min(1.0, authenticity))
    reach = reach_of(complaint)

    author_class = _classify(
        authenticity=authenticity,
        has_specificity=bool(spec),
        history_with_brand=history_with_brand,
        duplicate_authors=duplicate_authors,
        text=text,
    )

    # Reach overrides suspicion for the reply decision, and this is deliberate.
    # A troll with forty thousand followers still shapes what everyone else
    # believes, and silence reads as confirmation. Judge the person to decide
    # how much we *spend*; judge the audience to decide whether we *speak*.
    reply_worthy = (
        author_class != "bot"
        and (authenticity >= JUDGE.reply_worthy_authenticity
             or reach >= JUDGE.reply_worthy_reach)
    )

    return AuthorVerdict(
        author_class=author_class,
        authenticity=round(authenticity, 3),
        reach=reach,
        history_with_brand=history_with_brand,
        evidence=[f"{s.name}: {s.detail}" for s in signals],
        reply_worthy=reply_worthy,
    )


def _classify(*, authenticity: float, has_specificity: bool, history_with_brand: int,
              duplicate_authors: int, text: str) -> str:
    if duplicate_authors >= JUDGE.coordinated_duplicate_authors and not has_specificity:
        return "bot"
    if has_specificity or history_with_brand > 0:
        return "customer"
    if authenticity <= JUDGE.troll_ceiling:
        return "troll"
    if authenticity >= JUDGE.audience_floor:
        return "audience"
    return "unknown"


def needs_a_second_opinion(verdict: AuthorVerdict) -> bool:
    """Is this close enough to the line to be worth one cheap model call?

    Only the middle band. Confidently real and confidently manufactured are
    both already decided, and paying a model to agree with a settled answer is
    the most common way an "agentic" system quietly doubles its bill.
    """
    return (
        verdict.author_class in ("unknown", "troll")
        and JUDGE.ambiguous_low <= verdict.authenticity <= JUDGE.ambiguous_high
    )
