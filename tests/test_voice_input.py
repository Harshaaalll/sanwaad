"""Tests for what the call hears before the model sees it.

Two pieces of the voice leg run on every turn and neither of them needs a
model, a key or audio: the spoken-number normaliser and the per-call hotword
list. They are pure functions over text, which means the interesting cases —
Hindi scales, a number word that is also an English verb, a bias list that
must never carry an identifier — can all be pinned down offline.

The bar throughout is the one the normaliser sets for itself: when it is not
sure, it must leave the text alone. A wrong amount is worse than an
unconverted one, because a wrong amount looks like a fact.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.rag.store import get_store
from sanwaad.voice.brief import MAX_HOTWORDS, build_hotwords, build_voice_prompt
from sanwaad.voice.numbers import normalise, spoken_amount

# --- Numbers: the conversions that must happen ----------------------------

@pytest.mark.parametrize("spoken,value", [
    ("four thousand five hundred", 4500),
    ("chaar hazaar paanch sau", 4500),
    ("4 hazaar 500", 4500),                  # ASR mixes digits and words
    ("ek lakh bees hazaar", 120_000),
    ("चार हज़ार पाँच सौ", 4500),
    ("साढ़े चार हज़ार", 4500),                # "four and a half thousand"
    ("dhai lakh", 250_000),
    ("twenty two", 22),
    ("do hazaar", 2000),
    ("ek crore", 10_000_000),
])
def test_spoken_amounts_become_numbers(spoken, value):
    assert normalise(spoken).values == [value]


def test_devanagari_digits_are_transliterated():
    assert "4500" in normalise("मेरे ४५०० रुपये वापस चाहिए").text


def test_a_hindi_scale_is_reachable_whichever_way_the_nukta_arrives():
    """"ज़" is one code point or two, and which one an ASR emits is not our
    choice. Both must find the same table entry."""
    precomposed = normalise("do हज़ार")     # ज़ as U+095B
    decomposed = normalise("do हज़ार")  # ज + nukta
    assert precomposed.values == decomposed.values == [2000]


def test_a_currency_word_makes_it_an_amount_from_either_side():
    before = normalise("rupees two thousand")
    after = normalise("two thousand rupees")
    assert before.amounts_inr == after.amounts_inr == [2000]
    # And the word it came from is absorbed, not left doubled up.
    assert before.text == after.text == "₹2,000"


def test_an_amount_is_written_so_the_existing_triage_regex_finds_it():
    """The point of normalising is that the rest of the system can read it.
    `_largest_rupee_amount` is what decides severity and the ESC-02
    escalation, and it only matches a symbol-prefixed figure."""
    from sanwaad.graph.nodes import _largest_rupee_amount

    said = "mera chaar hazaar paanch sau rupaye ka transfer fail ho gaya"
    assert _largest_rupee_amount(said) is None
    assert _largest_rupee_amount(normalise(said).text) == 4500


# --- Numbers: the conversions that must NOT happen ------------------------

@pytest.mark.parametrize("text", [
    "do you have my refund",          # `do` is 2 in Hindi and a verb in English
    "one moment please",              # a lone unit is not a figure
    "bayalis hazaar rupaye",          # 42 in Hindi: not in the tables, so declined
    "bayalis hazaar paanch sau rupaye",   # ...and still declined mid-sentence
    "ikkyavan hazaar do sau rupees",      # 51,200: same, in another spelling
    "hazaar rupaye",                  # a scale with nothing in front of it
    "saath mein bhejo",               # 60, or "with" — never guessed
    "do hazaar hazaar",               # a stutter on the scale word
    "paanch sau sau",
    "char lakh crore",                # two scales, ascending
    "do do hazaar",                   # a stutter on the unit
    "twenty twenty",
])
def test_ambiguous_text_is_left_exactly_as_spoken(text):
    found = normalise(text)
    assert found.text == text
    assert not found.changed
    assert found.values == []


@pytest.mark.parametrize("said,never", [
    ("bayalis hazaar", 1000),                  # 42,000
    ("bayalis hazaar paanch sau", 1500),       # 42,500 — the mid-sentence form
    ("ikkyavan hazaar do sau", 1200),          # 51,200
])
def test_an_unknown_hindi_number_never_becomes_the_bare_scale(said, never):
    """The failure this guard exists for: the tables do not know `bayalis`, so
    `hazaar` starts a run with nothing in front of it, and reading it as one
    thousand hands the ledger a confident wrong figure.

    The earlier version of this test only passed "bayalis hazaar", which is the
    one case the first guard happened to catch. Anything the customer said
    afterwards joined the run and the guard stopped firing — 42,500 came out as
    ₹1,500. A test weaker than its own docstring is how that survived.
    """
    found = normalise(said)
    assert never not in found.values
    assert found.text == said            # left exactly as spoken
    assert spoken_amount(said) is None


def test_a_grouped_figure_is_read_as_one_number():
    """"₹4,500" is four thousand five hundred rupees, not four. The tokeniser
    split on the comma and only the group beside the currency symbol counted."""
    assert normalise("₹4,500").amounts_inr == [4500]
    assert normalise("Rs. 4,500 debited").amounts_inr == [4500]
    assert normalise("I paid 1,20,000 rupees").amounts_inr == [120_000]


def test_normalising_twice_gives_the_same_answer():
    """The module renders amounts with grouping, so its own output has to be
    readable to it. It was not: a second pass turned ₹4,500 into ₹4. The voice
    processor mutates the frame in place, so any frame that reaches it twice
    took that path."""
    once = normalise("chaar hazaar paanch sau rupaye")
    twice = normalise(once.text)
    assert once.amounts_inr == twice.amounts_inr == [4500]
    assert twice.text == once.text


def test_a_sentence_boundary_ends_a_number():
    """"refund of 640. Two orders" became "refund of 642 orders": the period
    was swallowed into the number token, so the run continued into the next
    sentence, ate its first word and invented 642."""
    found = normalise("refund of 640. Two orders")
    assert found.text == "refund of 640. Two orders"
    assert 642 not in found.values

    # The number before the stop is still read, and the word after it survives.
    after = normalise("It was paanch sau. Do you have it?")
    assert after.values == [500]
    assert "Do you have it?" in after.text


def test_digits_already_in_the_text_are_not_rewritten():
    assert normalise("₹640 double debit").text == "₹640 double debit"


# --- Numbers: read-out sequences ------------------------------------------

@pytest.mark.parametrize("spoken,digits", [
    ("nine eight seven six", "9876"),
    ("double four two", "442"),
    ("ek do teen", "123"),
])
def test_a_run_of_single_digits_is_a_sequence_not_a_sum(spoken, digits):
    found = normalise(spoken)
    assert found.references == [digits]
    assert found.values == []


# --- spoken_amount: refuses to choose -------------------------------------

def test_spoken_amount_returns_one_figure_or_nothing():
    assert spoken_amount("chaar hazaar paanch sau rupaye") == 4500
    assert spoken_amount("hello, are you there") is None
    # Two different figures in one breath: the caller must ask, not guess.
    assert spoken_amount("do hazaar ya teen hazaar") is None


# --- Hotwords -------------------------------------------------------------

_COMPLAINT = (
    "@NimbusPay double debit for one Swiggy order — ₹640 taken twice. "
    "My UPI payment failed but the money is gone. "
    "Call me on 9876543210, ref TXN-8842113."
)


def _citations():
    store = get_store()
    return [store.get("RFD-01"), store.get("RFD-02")]


def test_hotwords_carry_the_brand_and_the_names_the_customer_used():
    words = build_hotwords(complaint=_COMPLAINT, category="refund",
                           citations=_citations())
    assert "NimbusPay" in words
    assert "Swiggy" in words
    assert "UPI" in words


def test_hotwords_never_carry_an_identifier():
    """The list is sent to a third-party ASR. A reference number would be a
    useful bias term and is not worth the exposure."""
    words = build_hotwords(complaint=_COMPLAINT, citations=_citations())
    assert not any(any(ch.isdigit() for ch in w) for w in words)
    assert not any("9876543210" in w or "8842113" in w for w in words)


def test_a_capital_letter_that_only_means_a_sentence_started_is_not_a_name():
    words = build_hotwords(complaint="Refund not received. Call me back.")
    assert "Call" not in words
    assert "Refund" not in words


def test_hotwords_come_from_this_case_not_a_fixed_list():
    """Two different complaints must produce two different vocabularies, or
    the list is just a static prompt with extra steps."""
    refund = build_hotwords(complaint=_COMPLAINT, citations=_citations())
    mandate = build_hotwords(
        complaint="Mera NACH mandate cancel nahi ho raha, PhonePe se try kiya")
    assert "NACH" in mandate and "NACH" not in refund
    assert "Swiggy" in refund and "Swiggy" not in mandate


def test_the_list_stays_short_enough_to_be_worth_boosting():
    huge = " ".join(f"Merchant{i}" for i in range(80))
    assert len(build_hotwords(complaint=huge)) <= MAX_HOTWORDS


def test_the_call_prompt_carries_the_hotwords_it_was_given():
    prompt = build_voice_prompt(
        complaint=_COMPLAINT, public_reply="We are looking into it.",
        summary="Duplicate debit of ₹640", category="refund",
        language="en-IN", citations=_citations(),
    )
    assert "Swiggy" in prompt
    assert "Never guess an amount or a reference." in prompt
