"""The bridge that makes the two channels agree.

This module is the whole thesis in one file. The voice agent's system prompt
is not authored separately — it is *generated* from the same case state and
the same retrieved clauses that produced the public reply. There is no second
knowledge source for the two channels to drift apart on.

It also avoids a common failure of prompt-driven voice agents: one large static
prompt that relies on the model to remember which phase of the call it is in.
Here the graph owns the phase and the prompt carries only what this call
actually needs.
"""

from __future__ import annotations

import re

from ..guardrails import find_pii
from ..models import Citation

_VOICE_TEMPLATE = """You are Adhik from NimbusPay support, calling a customer back about a complaint they posted publicly.

# What already happened
They posted: "{complaint}"
We replied publicly: "{public_reply}"

You must not contradict that public reply. If you now believe it was wrong,
say you will re-check and escalate — never announce a different answer on the
call than the one standing in public.

# What this call is about
{summary}
Category: {category}. Speak in: {language}.

# The ONLY facts you may state
{clauses}

Every timeline, amount, entitlement and process you state must come from the
clauses above. If the customer asks something they do not cover, say you will
find out and have someone confirm in writing. Do not estimate. Do not round.
Do not invent a reference number.

# Words the transcription may get wrong
{hotwords}

# How to talk
- This is a phone call. One or two sentences per turn, then stop.
- Open by naming the complaint, so they know you actually read it.
- Never ask for an OTP, PIN, password or CVV, and say so if they offer one.
- Do not read account numbers or transaction ids aloud unless the customer
  says them first.
- When you state something that comes from a clause, you may reference the
  policy in plain language, never by its id. The customer must never hear
  "clause RFD-01".

# Ending
When the issue is resolved or the next step is agreed and the customer has
nothing further, thank them and end. Do not prolong the call to fill silence.
"""


_NO_HOTWORDS = "Nothing specific. Ask them to repeat anything you did not catch."


def build_voice_prompt(
    *,
    complaint: str,
    public_reply: str,
    summary: str,
    category: str,
    language: str,
    citations: list[Citation],
    hotwords: list[str] | None = None,
) -> str:
    """Render the call's system prompt from the case's own citations."""
    clauses = "\n\n".join(f"[{c.clause_id}] {c.heading}\n{c.text}" for c in citations)
    if hotwords is None:
        hotwords = build_hotwords(complaint=complaint, category=category,
                                  citations=citations)
    return _VOICE_TEMPLATE.format(
        complaint=complaint.strip(),
        public_reply=public_reply.strip(),
        summary=summary,
        category=category,
        language=language,
        clauses=clauses,
        hotwords=_hotword_block(hotwords),
    )


def _hotword_block(hotwords: list[str]) -> str:
    if not hotwords:
        return _NO_HOTWORDS
    return (
        "Expect to hear these, and read a near-miss as the intended word:\n"
        + ", ".join(hotwords)
        + "\nIf a word you need is not in that list and you did not catch it, "
          "ask them to repeat it. Never guess an amount or a reference."
    )


# ---------------------------------------------------------------------------
# Hotwords: the vocabulary this particular call is about
# ---------------------------------------------------------------------------

# A general-purpose ASR is trained on general speech, and a support call is not
# general speech. It is dense with exactly the words a general model is worst at:
# a brand it has never seen, a merchant's name, an acronym said fast, a piece of
# payments jargon. "NimbusPay" comes back as "nimbus pay" or "nimble pay";
# "NACH mandate" as "nach mandate"; "chargeback" as "charge back".
#
# Every serious ASR takes a bias list for this, under one name or another —
# hotwords, keyterms, a boost vocabulary, an initial prompt. The interesting part
# is not the API. It is that the list has to be *small* and *specific*: a hundred
# boosted terms bias the model toward hearing them everywhere, so the list must
# be the vocabulary of this call and nothing else.
#
# Sanwaad already knows what this call is about, from the same two sources that
# produced the public reply: the complaint text and the retrieved clauses. So the
# list is derived, not authored, and it cannot drift from the case.
#
# Two rules:
#
# - NOTHING IDENTIFYING LEAVES. A bias list is sent to a third party, so anything
#   the redactor would mask, and anything containing a digit, is dropped. A
#   reference number would be a useful hotword and is not worth the exposure.
# - IT IS PROVIDER-AGNOSTIC. `build_hotwords` returns plain strings. Whichever
#   STT is configured maps them to its own parameter, and one that has no such
#   parameter loses nothing: the same list goes into the call prompt, so the
#   model reads a near-miss as the word that was meant.

MAX_HOTWORDS = 24

# Payments and support vocabulary that a general ASR mishears, with the casing
# we want back. Only the entries this case actually touches are ever sent.
_DOMAIN_TERMS: tuple[str, ...] = (
    "UPI", "NEFT", "IMPS", "RTGS", "NACH", "KYC", "OTP", "UTR", "RRN", "IFSC",
    "EMI", "VPA", "SLA", "TAT", "QR code", "autopay", "e-mandate", "mandate",
    "chargeback", "reversal", "provisional credit", "double debit",
    "duplicate debit", "beneficiary", "remitter", "settlement", "payout",
    "standing instruction", "collect request", "nodal officer", "ombudsman",
    "grievance", "arbitration", "passbook", "cashback",
)
# Deliberately absent: debit, credit, merchant, dispute, wallet. They are
# domain words, but they are also everyday ones that a general model already
# hears correctly, and every slot spent on them biases the list for nothing.

# Capitalised words that are only capitalised because a sentence started, or
# because people shout. Never worth boosting.
_NOT_A_NAME = frozenset("""
    i my me we you your our the a an this that these those there here it its
    hi hey hello please thanks thank sorry why when what where how who which
    still again since after before now today yesterday tomorrow money amount
    refund payment transaction account bank support help team service issue
    problem app site website customer care number rupees rs inr no yes ok okay
    dear sir madam guys urgent scam fraud worst pathetic shame kindly
""".split())

_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}(?:[A-Z][a-z]+)*\b")


def build_hotwords(
    *,
    complaint: str,
    category: str = "",
    citations: list[Citation] | None = None,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """The bias vocabulary for one call, derived from that call's own case.

    Ordered by how much boosting them is worth: the brand first, then names
    this customer used, then the jargon this case's clauses are written in.
    Truncation therefore drops the least valuable terms, not arbitrary ones.
    """
    citations = citations or []
    clause_text = " ".join(f"{c.heading} {c.text}" for c in citations)
    identifying = " ".join(match for _, match in find_pii(complaint))

    ordered: list[str] = [*_brand_terms(), *extra]

    ordered.extend(_names_in(complaint))
    ordered.extend(_acronyms_in(complaint))

    # Jargon, but only what this case is actually about. `category` is included
    # because the word itself is often spoken ("this is a billing issue").
    haystack = f"{complaint} {clause_text} {category}".lower()
    ordered.extend(t for t in _DOMAIN_TERMS
                   if re.search(rf"\b{re.escape(t.lower())}\b", haystack))

    seen: set[str] = set()
    out: list[str] = []
    for term in ordered:
        term = term.strip()
        key = term.lower()
        if not term or key in seen or len(term) < 2:
            continue
        if any(ch.isdigit() for ch in term) or (identifying and term in identifying):
            continue
        seen.add(key)
        out.append(term)
        if len(out) == MAX_HOTWORDS:
            break
    return out


def _acronyms_in(complaint: str) -> list[str]:
    """All-caps tokens, but only where all-caps still means something.

    Capitals are how an acronym announces itself — and also how an angry
    customer types. In a shouted complaint nothing stands out, so every word
    matched and the bias list filled with FAILED, NOBODY and EVER. A list of
    stop words is worse than no list: it biases the ASR toward hearing them
    everywhere while crowding out the terms worth boosting.

    A stop-word list cannot fix that, because the words are ordinary ones. What
    can is noticing that the signal is absent: when the complaint is mostly
    capitals, only vocabulary we already recognise counts.
    """
    letters = [c for c in complaint if c.isalpha()]
    shouting = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.6
    known = {t.lower() for t in _DOMAIN_TERMS}
    return [a for a in _ACRONYM.findall(complaint)
            if a.lower() not in _NOT_A_NAME and (not shouting or a.lower() in known)]


def _names_in(complaint: str) -> list[str]:
    """Merchant and product names the customer typed.

    A name the ASR has never heard is the most mis-transcribed word on a
    payments call, so these rank just below the brand. The trap is the capital
    letter that only means a sentence started: "Call me back" would otherwise
    boost "Call". A word is taken as a name if it appears capitalised somewhere
    other than the start of a sentence, or carries an inner capital the way
    NimbusPay does.
    """
    mid: list[str] = []
    seen_mid: set[str] = set()
    sentence_initial: list[str] = []

    for match in _PROPER.finditer(complaint):
        term = match.group()
        if term.lower() in _NOT_A_NAME:
            continue
        before = complaint[:match.start()].rstrip()
        if not before or before[-1] in ".!?…\n" or any(c.isupper() for c in term[1:]):
            sentence_initial.append(term)
        else:
            mid.append(term)
            seen_mid.add(term.lower())

    return mid + [t for t in sentence_initial
                  if t.lower() in seen_mid or any(c.isupper() for c in t[1:])]


def _brand_terms() -> list[str]:
    """The brand and the ways it is written, from the listener's own config, so
    the two ends of the system cannot disagree about what the brand is called."""
    from ..listener import BRAND, BRAND_ALIASES

    return [BRAND, *BRAND_ALIASES]


class CitationTracker:
    """Works out which clauses the call actually relied on.

    We cannot ask the model to self-report reliably mid-call, and we do not
    want a second LLM pass per turn. Instead we match distinctive terms from
    each clause against the assistant's transcript. It over-reports slightly,
    which is the right direction to err: a clause wrongly credited shows up as
    harmless divergence, while a missed one could hide a real contradiction.
    """

    # Words too common to identify a clause by.
    _STOP = frozenset("""
        the a an and or of to in for is are was were be been we you your our it
        that this with on at by from as not no if then than so but do does did
        can may will shall must never always any all each per within under over
        customer agent nimbuspay account they them their he she his her its
    """.split())

    def __init__(self, citations: list[Citation], min_terms: int = 2):
        self.citations = citations
        self.min_terms = min_terms
        self._terms: dict[str, set[str]] = {}
        for c in citations:
            self._terms[c.clause_id] = self._distinctive(f"{c.heading} {c.text}")
        self._spoken: list[str] = []

    def _distinctive(self, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9+₹%.-]{3,}", text.lower())
        return {w for w in words if w not in self._STOP}

    def observe(self, assistant_text: str) -> None:
        self._spoken.append(assistant_text)

    def used(self) -> list[str]:
        spoken = self._distinctive(" ".join(self._spoken))
        if not spoken:
            return []
        return sorted(
            cid for cid, terms in self._terms.items()
            if len(terms & spoken) >= self.min_terms
        )
