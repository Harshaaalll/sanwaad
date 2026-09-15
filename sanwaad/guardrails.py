"""Guardrails: the checks that run whether or not the model cooperates.

A guardrail is not a prompt instruction. Prompt instructions are requests; a
guardrail is code that runs after the model has spoken and can still refuse.
The distinction matters because everything here defends against the case where
the model does the wrong thing, which is exactly the case where asking it
nicely has already failed.

Three positions, and they are not interchangeable:

  INPUT   before the text reaches any model — untrusted customer content
  OUTPUT  after generation, before anything is published
  ACTION  before an irreversible effect (posting, calling, moving money)

Sanwaad's existing gates are ACTION guardrails (`auto_post_allowed`) and a
model-based OUTPUT guardrail (the grounding check). This module adds the
deterministic ones, which are cheaper, faster and cannot themselves be talked
out of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# PII — enforces clause PRV-02, which forbids repeating identifiers in public
# ---------------------------------------------------------------------------

# Indian mobile numbers: 10 digits starting 6-9, optionally +91 / 0 prefixed,
# tolerant of the spaces and dashes people actually type.
# 10 digits starting 6-9, with separators allowed ANYWHERE between them.
# Indians write the same number as 9876543210, 98765 43210 and 98765-43210;
# a pattern that assumes one grouping leaks the other two. The digit-boundary
# lookarounds stop it from biting a chunk out of a longer run such as a UTR.
_PHONE = re.compile(r"(?:(?:\+?91|0)[\s-]?)?(?<!\d)[6-9](?:[\s-]?\d){9}(?!\d)")
_UTR = re.compile(r"\b\d{12,22}\b")                       # UTR / RRN / account
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
# Aadhaar is 12 digits, never starts 0 or 1, and is conventionally written in
# 4-4-4 groups. A *bare* 12-digit run is far more often a UTR, so we require the
# separators here and let UTR claim the ungrouped case. Both get redacted either
# way; this only decides which label lands in the audit log.
_AADHAAR = re.compile(r"\b[2-9]\d{3}[\s-]\d{4}[\s-]\d{4}\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_CARD = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")

# Order matters: the most specific pattern must claim its digits first, or a
# card number gets partially eaten by the phone pattern and the remainder
# leaks. Card and Aadhaar are both 16/12 digits, so they precede the rest.
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("card", _CARD),
    ("aadhaar", _AADHAAR),   # grouped 4-4-4 only
    ("utr", _UTR),           # bare 12-22 digit runs

    ("email", _EMAIL),
    ("pan", _PAN),
    ("phone", _PHONE),
]

_PLACEHOLDER = {
    "phone": "[phone]", "utr": "[reference]", "email": "[email]",
    "aadhaar": "[aadhaar]", "pan": "[pan]", "card": "[card]",
}


def find_pii(text: str) -> list[tuple[str, str]]:
    """Return [(kind, matched_text)] without mutating anything."""
    found, claimed = [], []

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for kind, pattern in _PII_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            if any(overlaps(span, c) for c in claimed):
                continue
            claimed.append(span)
            found.append((kind, m.group()))
    return found


def redact(text: str) -> tuple[str, list[str]]:
    """Replace identifiers with placeholders. Returns (clean_text, kinds)."""
    kinds, claimed = [], []
    spans = []
    for kind, pattern in _PII_PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.span()
            if any(not (e <= cs or ce <= s) for cs, ce in claimed):
                continue
            claimed.append((s, e))
            spans.append((s, e, kind))
            kinds.append(kind)
    for s, e, kind in sorted(spans, reverse=True):
        text = text[:s] + _PLACEHOLDER[kind] + text[e:]
    return text, sorted(set(kinds))


# ---------------------------------------------------------------------------
# Prompt injection — the complaint text is untrusted input
# ---------------------------------------------------------------------------

_INJECTION = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)"
    r"|disregard\s+(the\s+)?(system|previous|above)"
    r"|you\s+are\s+now\s+(a|an)\b"
    r"|new\s+(system\s+)?(instructions?|prompt)\s*:"
    r"|forget\s+(everything|your\s+(rules|instructions))"
    r"|pretend\s+(to\s+be|you\s+are)"
    r"|reveal\s+(your\s+)?(system\s+)?prompt"
    r"|act\s+as\s+(if|though)\s+you",
    re.IGNORECASE,
)


def detect_injection(text: str) -> list[str]:
    return [m.group().strip() for m in _INJECTION.finditer(text)]


# ---------------------------------------------------------------------------
# Output guardrail
# ---------------------------------------------------------------------------

Severity = Literal["block", "warn"]


@dataclass
class Violation:
    rule: str
    severity: Severity
    detail: str


@dataclass
class GuardResult:
    passed: bool
    text: str                                    # possibly repaired
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(v.severity == "block" for v in self.violations)


# Clause BV-06 bans these outright; they measurably raise escalation rates.
_BANNED_PHRASES = [
    "as per policy", "we regret the inconvenience", "kindly do the needful",
    "your call is important to us", "please be informed",
]

# Money commitments. Matched on the reply, not the complaint.
_MONEY_PROMISE = re.compile(
    r"\b(we\s+(will|shall|have)\s+(refund|credit|reverse|waive|compensate)"
    r"|refund(ed|ing)?\s+(of|will|has been)"
    r"|(has|have)\s+been\s+(approved|credited|refunded|reversed)"
    r"|you\s+will\s+receive\s+(₹|rs\.?|inr)\s*[\d,]+)",
    re.IGNORECASE,
)

_WORD_CEILING = 60


def check_reply(text: str, *, allowed_clause_ids: list[str] | None = None) -> GuardResult:
    """Run every deterministic check on a drafted public reply.

    PII is *repaired* rather than blocked — a good reply that happens to echo
    a UTR should be fixed and posted, not thrown away. Everything else blocks,
    because those are cases where the reply is wrong rather than untidy.
    """
    violations: list[Violation] = []
    out = text

    clean, kinds = redact(out)
    if kinds:
        out = clean
        violations.append(Violation(
            "PRV-02", "warn", f"redacted {', '.join(kinds)} from the public reply"))

    for phrase in _BANNED_PHRASES:
        if phrase in out.lower():
            violations.append(Violation("BV-06", "block", f"banned phrase: {phrase!r}"))

    if _MONEY_PROMISE.search(out):
        violations.append(Violation(
            "RFD-05", "block", "reply commits money; requires approval"))

    words = len(out.split())
    if words > _WORD_CEILING:
        violations.append(Violation(
            "BV-01", "warn", f"{words} words, over the {_WORD_CEILING}-word ceiling"))

    # A clause id must never reach the customer — they are internal handles.
    leaked = re.findall(r"\b(?:BV|RFD|BIL|KYC|ESC|PRV)-\d{2}\b", out)
    if leaked:
        violations.append(Violation(
            "BV-01", "block", f"internal clause id(s) in customer text: {leaked}"))

    return GuardResult(
        passed=not any(v.severity == "block" for v in violations),
        text=out, violations=violations,
    )


def check_complaint(text: str) -> tuple[str, list[Violation]]:
    """INPUT guardrail. Returns (text_safe_to_prompt_with, violations).

    This flags instruction-shaped text so an injection attempt lands in the
    human queue instead of the auto-post path. It does not redact: that happens
    at the model boundary (`context.minimal_text`), which strips identifiers but
    keeps amounts — severity genuinely depends on "₹18,000", and nothing a
    model decides here depends on a phone number.
    """
    violations = []
    hits = detect_injection(text)
    if hits:
        violations.append(Violation(
            "injection", "warn", f"instruction-like text in complaint: {hits[:3]}"))
    return text, violations
