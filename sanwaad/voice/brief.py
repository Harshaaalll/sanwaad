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


def build_voice_prompt(
    *,
    complaint: str,
    public_reply: str,
    summary: str,
    category: str,
    language: str,
    citations: list[Citation],
) -> str:
    """Render the call's system prompt from the case's own citations."""
    clauses = "\n\n".join(f"[{c.clause_id}] {c.heading}\n{c.text}" for c in citations)
    return _VOICE_TEMPLATE.format(
        complaint=complaint.strip(),
        public_reply=public_reply.strip(),
        summary=summary,
        category=category,
        language=language,
        clauses=clauses,
    )


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
