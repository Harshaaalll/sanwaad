"""Spoken numbers, turned back into numbers.

Every deterministic check in this system runs on a number. The ledger lookup
matches an amount. The guardrails compare what the agent said against what the
clause allows. The consistency receipt asserts the voice leg and the public
reply named the same figure. All of that assumes the amount arrives as `4500`.

Speech does not arrive that way. An Indic ASR transcribing a real call returns
"chaar hazaar paanch sau", "4 hazaar 500", "साढ़े चार हज़ार" or "rupees four
thousand five hundred", and a downstream `float(...)` sees none of them. The
failure is silent and it is the worst shape of failure: the customer stated
their amount clearly, the model heard it correctly, and the lookup still missed
because nothing in the pipeline converts words to digits.

So this module sits between the transcript and everything that reasons about
it. Three rules govern it:

1.  IT NEVER REWRITES THE RECORD. `normalise` returns a new string for the
    machinery. What the customer actually said stays in the transcript, because
    a normaliser that edits the evidence cannot be audited when it is wrong.
2.  AMBIGUITY IS LEFT ALONE. Hindi number words collide with ordinary English
    ones — `do` is 2 and also "do", `char` is 4 and also "char", `so` is
    neither. Those tokens only count as numbers next to another number or a
    scale word, so "do hazaar" becomes 2000 while "do you have" is untouched.
    A normaliser that is right 95% of the time and silently wrong the rest is
    worse than one that declines.
3.  IT IS PURE AND OFFLINE. No provider, no key, no network. It is tested the
    way arithmetic should be tested.

## What it handles

    four thousand five hundred      → 4500
    chaar hazaar paanch sau         → 4500
    4 hazaar 500                    → 4500       (ASR mixes forms mid-phrase)
    ek lakh bees hazaar             → 120000
    चार हज़ार पाँच सौ                 → 4500
    ४५००                            → 4500       (Devanagari digits)
    rupees two thousand             → ₹2000      (tagged as an amount)
    nine eight seven six            → "9876"     (a reference, not 30)
    double four two                 → "442"

## What it declines, on purpose

Hindi has an irregular, non-compositional word for every number from 21 to 99
(`ikkyavan` is 51, not "fifty one"), spelled a dozen ways by different ASR
models. Encoding a guessed table would produce confident wrong amounts, which
is precisely the failure this module exists to prevent. Round tens are covered;
`bayalis hazaar` is not, and is left as text for a person to read — including
when the sentence carries on afterwards, which is the case an earlier version
of this module got wrong.

It also declines what is malformed rather than repairing it. A scale needs
something in front of it, scales must strictly decrease (crore, lakh, thousand,
hundred, each at most once) and so must the additive words, so an ASR stutter
like "do hazaar hazaar" or "do do hazaar" is declined instead of being read as
3,000 or 4,000. Repetition and disorder are what a garbled number looks like,
and neither is worth guessing at.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Devanagari digits are a pure transliteration — never ambiguous, always safe.
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_UNITS: dict[str, int] = {
    "zero": 0, "shunya": 0, "sifar": 0, "शून्य": 0,
    "one": 1, "ek": 1, "एक": 1,
    "two": 2, "do": 2, "दो": 2,
    "three": 3, "teen": 3, "तीन": 3,
    "four": 4, "chaar": 4, "char": 4, "चार": 4,
    "five": 5, "paanch": 5, "panch": 5, "पांच": 5, "पाँच": 5,
    "six": 6, "chhe": 6, "che": 6, "chah": 6, "छह": 6, "छे": 6,
    "seven": 7, "saat": 7, "sat": 7, "सात": 7,
    "eight": 8, "aath": 8, "ath": 8, "आठ": 8,
    "nine": 9, "nau": 9, "नौ": 9,
}

_TEENS: dict[str, int] = {
    "ten": 10, "das": 10, "दस": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

# Round tens only. See the module docstring for why 21–99 in Hindi are absent,
# and why 60 (`saath`, also "with") is absent even though it is round.
_TENS: dict[str, int] = {
    "twenty": 20, "bees": 20, "बीस": 20,
    "thirty": 30, "tees": 30, "तीस": 30,
    "forty": 40, "chalis": 40, "chaalis": 40, "चालीस": 40,
    "fifty": 50, "pachas": 50, "pachaas": 50, "पचास": 50,
    "sixty": 60,
    "seventy": 70, "sattar": 70, "सत्तर": 70,
    "eighty": 80, "assi": 80, "अस्सी": 80,
    "ninety": 90, "nabbe": 90, "नब्बे": 90,
}

_SCALES: dict[str, int] = {
    "hundred": 100, "sau": 100, "सौ": 100,
    "thousand": 1_000, "hazaar": 1_000, "hazar": 1_000, "hajar": 1_000,
    "हजार": 1_000, "हज़ार": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "लाख": 100_000,
    "million": 1_000_000,
    "crore": 10_000_000, "crores": 10_000_000, "karod": 10_000_000,
    "करोड़": 10_000_000, "करोड": 10_000_000,
}

_REPEATERS: dict[str, int] = {"double": 2, "triple": 3, "treble": 3}

# Words that mean "and a half" after a scale: saadhe chaar hazaar = 4500.
_HALF_BEFORE = {"saadhe", "sadhe", "साढ़े", "साढे"}     # X and a half
_QUARTER_MORE = {"sava", "sawa", "सवा"}                 # X and a quarter
_QUARTER_LESS = {"paune", "पौने"}                       # a quarter less than X
_FIXED_FRACTIONS: dict[str, float] = {
    "dedh": 1.5, "डेढ़": 1.5, "देढ़": 1.5,      # one and a half
    "dhai": 2.5, "ढाई": 2.5,                   # two and a half
    "adha": 0.5, "aadha": 0.5, "आधा": 0.5,
}

_CURRENCY = {
    "rupees", "rupee", "rupaye", "rupaiye", "rupaya", "rs", "rs.", "inr", "₹",
    "रुपये", "रुपए", "रुपया",
}

# Hindi number words that are also ordinary English words. They only count
# inside a run that some unambiguous token already anchors, so "do hazaar" is
# 2000 while "do you have" keeps its verb.
_AMBIGUOUS = {"do", "char", "sat"}


def _canon(word: str) -> str:
    """Devanagari has two spellings for the same letter — "ज़" is either one
    code point or "ज" plus a nukta, and which one arrives depends on the ASR.
    Both sides of every lookup go through NFC so they cannot miss each other.
    """
    return unicodedata.normalize("NFC", word)


_UNITS = {_canon(k): v for k, v in _UNITS.items()}
_TEENS = {_canon(k): v for k, v in _TEENS.items()}
_TENS = {_canon(k): v for k, v in _TENS.items()}
_SCALES = {_canon(k): v for k, v in _SCALES.items()}
_FIXED_FRACTIONS = {_canon(k): v for k, v in _FIXED_FRACTIONS.items()}
_HALF_BEFORE = {_canon(w) for w in _HALF_BEFORE}
_QUARTER_MORE = {_canon(w) for w in _QUARTER_MORE}
_QUARTER_LESS = {_canon(w) for w in _QUARTER_LESS}
_CURRENCY = {_canon(w) for w in _CURRENCY}

_NUMBER_WORDS = {**_UNITS, **_TEENS, **_TENS}
# The Devanagari range is spelled out because Python's `\w` excludes combining
# marks: without it, "हज़ार" tokenises as "हज", "़", "ा", "र" and every Hindi
# number word in the tables above becomes unreachable.
# Digits first, and as a whole figure: a grouped "4,500" and a decimal "4.50"
# are each ONE token. Tokenising them as "4" / "," / "500" made `₹4,500` read
# as ₹4 — the first group only, because that is the group the currency symbol
# was adjacent to. It also made the module non-idempotent: it renders amounts
# with grouping, so its own output was unreadable to it.
#
# The trailing period is deliberately NOT part of a token. It used to be, and
# "refund of 640. Two orders" became "refund of 642 orders": the period was
# stripped for the lookup, so the run continued across the sentence boundary,
# ate the next sentence's first word and invented 642.
_TOKEN = re.compile(r"[₹]|\d[\d,]*(?:\.\d+)?|[\wऀ-ॿ]+|\s+|[^\s\w]", re.UNICODE)
_NUMERIC = re.compile(r"^\d[\d,]*(?:\.\d+)?$")


def _numeric(token: str) -> float:
    """A digit token's value. Grouping separators are ours, not the speaker's."""
    return float(token.replace(",", ""))


@dataclass
class Change:
    """One rewritten span, kept so a reviewer can see what was changed."""

    spoken: str
    written: str
    value: float
    is_amount: bool


@dataclass
class Normalised:
    text: str
    amounts_inr: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.changes)


# ---------------------------------------------------------------------------
# Parsing one run of number tokens
# ---------------------------------------------------------------------------

def _is_number_token(word: str) -> bool:
    return (word in _NUMBER_WORDS or word in _SCALES or word in _REPEATERS
            or word in _FIXED_FRACTIONS or word in _HALF_BEFORE
            or word in _QUARTER_MORE or word in _QUARTER_LESS
            or bool(_NUMERIC.match(word)))


def _anchored(words: list[str]) -> bool:
    """Is there at least one token here that can only be a number?

    This is rule 2 of the module docstring, in one function. "do hazaar" is
    anchored by `hazaar`; "do you" is not anchored at all, so nothing in it is
    treated as a number.
    """
    return any(w not in _AMBIGUOUS and _is_number_token(w) and not _NUMERIC.match(w)
               for w in words) or any(w in _SCALES and w not in _AMBIGUOUS for w in words)


def _as_digit_string(words: list[str]) -> str | None:
    """A read-out reference or phone number, rather than an arithmetic phrase.

    Nobody says "one two three" meaning six. A run made only of single digits
    (with `double`/`triple` expanding the next one) is a sequence being read
    aloud, and joining the digits is the only reading that is ever intended.
    """
    if len(words) < 2:
        return None
    out: list[str] = []
    repeat = 1
    for w in words:
        if w in _REPEATERS:
            repeat = _REPEATERS[w]
            continue
        if w in _UNITS:
            digit = str(_UNITS[w])
        elif _NUMERIC.match(w) and len(w) == 1:
            digit = w
        else:
            return None
        out.append(digit * repeat)
        repeat = 1
    return "".join(out) if repeat == 1 and out else None


def _arithmetic(words: list[str]) -> float | None:
    """Standard place-value accumulation, with the Indian scales included.

    Two rules keep a malformed phrase from becoming a confident number, and
    both exist because the naive version invented figures nobody said:

    A SCALE MUST HAVE SOMETHING IN FRONT OF IT. `base = current or 1.0` reads
    a bare scale as one of them, so "bayalis hazaar paanch sau" — 42,500, whose
    `bayalis` this module openly does not know — came out as ₹1,500. The
    unknown word ends the previous run, `hazaar` starts a new one with nothing
    before it, and the `or 1.0` supplies the missing 42.

    SCALES MUST STRICTLY DECREASE. A well-formed number says crore, then lakh,
    then thousand, then hundred, each at most once. Without that, an ASR
    stutter on a scale word multiplies: "do hazaar hazaar" was 3,000 and
    "paanch sau sau" was 50,000. Repetition and disorder are the two shapes a
    garbled scale takes, and neither is a number worth guessing at.
    """
    total, current = 0.0, 0.0
    seen, pending_multiplier = False, 1.0
    last_scale: float = float("inf")
    last_add: float = float("inf")

    for w in words:
        if w in _REPEATERS:            # only meaningful in a digit string
            return None
        if w in _FIXED_FRACTIONS:
            current += _FIXED_FRACTIONS[w]
            seen = True
        elif w in _HALF_BEFORE:
            pending_multiplier = 0.5   # applied as "+ half of the next scale"
            seen = True
        elif w in _QUARTER_MORE:
            pending_multiplier = 0.25
            seen = True
        elif w in _QUARTER_LESS:
            pending_multiplier = -0.25
            seen = True
        elif w in _NUMBER_WORDS or _NUMERIC.match(w):
            value = _NUMBER_WORDS[w] if w in _NUMBER_WORDS else _numeric(w)
            band = 10 if value >= 10 else 1
            if band >= last_add:
                # The same decreasing rule the scales follow. Only tens-then-
                # unit composes ("twenty two"); unit-then-unit and
                # tens-then-tens do not, and reading them as a sum turned the
                # stutter "do do hazaar" into 4,000 and "twenty twenty" into 40.
                return None
            last_add = band
            current += value
            seen = True
        elif w in _SCALES:
            scale = _SCALES[w]
            if scale >= last_scale:
                return None            # repeated or out-of-order: not a number
            last_scale = scale
            last_add = 100             # a scale starts a fresh group
            if current == 0.0 and pending_multiplier == 1.0:
                return None            # a scale with nothing to multiply
            base = current
            if pending_multiplier != 1.0:
                base = (base or 1.0) + pending_multiplier
                pending_multiplier = 1.0
            if scale >= 1_000:
                total += base * scale
                current = 0.0
            else:
                current = base * scale
            seen = True
        else:
            return None

    if not seen or pending_multiplier != 1.0:
        return None                     # "saadhe" with nothing to scale
    return total + current


def _format(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"


# ---------------------------------------------------------------------------
# The pass over a transcript
# ---------------------------------------------------------------------------

def normalise(text: str, *, currency_symbol: str = "₹") -> Normalised:
    """Rewrite spoken numbers as digits. The input is never mutated.

    Returns the rewritten text plus what was found: amounts tagged as rupees,
    bare values, and digit sequences that were read out rather than counted.
    """
    text = _canon(text.translate(_DEVANAGARI_DIGITS))
    tokens = _TOKEN.findall(text)
    result = Normalised(text=text)
    out: list[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        word = token.lower() if token.strip() else token

        if not token.strip() or not _is_number_token(word):
            out.append(token)
            i += 1
            continue

        # Collect the run, keeping the whitespace so we can put it back
        # untouched if we decide not to rewrite.
        run_tokens, run_words, j = [], [], i
        while j < len(tokens):
            t = tokens[j]
            w = t.lower()
            if not t.strip():
                # Whitespace only continues a run if a number follows it.
                k = j + 1
                if k < len(tokens) and _is_number_token(tokens[k].lower()):
                    run_tokens.append(t)
                    j += 1
                    continue
                break
            if not _is_number_token(w):
                break
            run_tokens.append(t)
            run_words.append(w)
            j += 1

        spoken = "".join(run_tokens)
        amounts_before = len(result.amounts_inr)
        rewritten = _rewrite(run_words, result, spoken, currency_symbol,
                             before=_word_before(out), after=_word_after(tokens, j))
        if rewritten is None:
            out.append(spoken)
            i = j
            continue

        # "rupees two thousand" would otherwise become "rupees ₹2,000". The
        # symbol says it already, so the word it came from is absorbed.
        if len(result.amounts_inr) > amounts_before and rewritten.startswith(currency_symbol):
            _drop_currency_before(out)
            j = _skip_currency_after(tokens, j)
        out.append(rewritten)
        i = j

    result.text = "".join(out)
    return result


def _drop_currency_before(out: list[str]) -> None:
    k = len(out) - 1
    while k >= 0 and not out[k].strip():
        k -= 1
    if k >= 0 and out[k].lower() in _CURRENCY:
        del out[k:]


def _skip_currency_after(tokens: list[str], j: int) -> int:
    k = j
    while k < len(tokens) and not tokens[k].strip():
        k += 1
    return k + 1 if k < len(tokens) and tokens[k].lower() in _CURRENCY else j


def _word_before(out: list[str]) -> str:
    """The word to the left, looking through an abbreviation's full stop.

    "Rs. 4,500" puts a lone "." between the currency word and the figure now
    that a period is its own token, and skipping it is what keeps that figure
    recognised as money. Only one, and only a period: anything else between
    them means they are not adjacent.
    """
    skipped_dot = False
    for token in reversed(out):
        if not token.strip():
            continue
        if token == "." and not skipped_dot:
            skipped_dot = True
            continue
        return token.lower()
    return ""


def _word_after(tokens: list[str], j: int) -> str:
    for token in tokens[j:]:
        if token.strip():
            return token.lower()
    return ""


def _rewrite(words: list[str], result: Normalised, spoken: str,
             symbol: str, *, before: str, after: str) -> str | None:
    """Decide what one run of number tokens becomes, or None to leave it."""
    if not words:
        return None

    # Already digits, nothing spoken to convert. Still worth recording as an
    # amount if a currency word sits beside it, so callers get one list.
    if all(_NUMERIC.match(w) for w in words):
        # One figure only. A run of two digit tokens is two numbers that happen
        # to be adjacent ("640 500"), and concatenating them would invent a
        # third — which is what `float("".join(...))` used to do.
        if len(words) == 1 and (before in _CURRENCY or after in _CURRENCY or before == symbol):
            value = _numeric(words[0])
            result.amounts_inr.append(value)
            result.values.append(value)
        return None

    if not _anchored(words):
        return None

    # A scale with nothing in front of it. This is the signature of a Hindi
    # number the tables do not know: "bayalis hazaar" reaches here as just
    # ["hazaar"], and reading it as 1,000 would put a confidently wrong amount
    # into the ledger lookup. Declining leaves it for a person.
    if not any(w in _NUMBER_WORDS or w in _FIXED_FRACTIONS or _NUMERIC.match(w)
               for w in words):
        return None

    digits = _as_digit_string(words)
    if digits is not None:
        result.references.append(digits)
        result.changes.append(Change(spoken, digits, float(digits), False))
        return digits

    value = _arithmetic(words)
    if value is None:
        return None

    is_amount = before in _CURRENCY or after in _CURRENCY or before == symbol

    # A lone "one" or "chaar" is far more often an article or a count than a
    # figure anyone will act on. Rewriting it buys nothing and makes the
    # transcript read like a receipt, so it stays as spoken unless a currency
    # word says it is money.
    if len(words) == 1 and words[0] in _UNITS and not is_amount:
        return None
    written = f"{symbol}{_format(value)}" if is_amount and before != symbol else _format(value)
    result.values.append(value)
    if is_amount:
        result.amounts_inr.append(value)
    result.changes.append(Change(spoken, written, value, is_amount))
    return written


def spoken_amount(text: str) -> float | None:
    """The single rupee amount in a phrase, or None if it is not exactly one.

    Used where the caller needs a number and an ambiguous answer is worse than
    no answer: the ledger lookup would rather ask again than search for the
    wrong figure.
    """
    found = normalise(text)
    amounts = found.amounts_inr or found.values
    unique = sorted(set(amounts))
    return unique[0] if len(unique) == 1 else None
