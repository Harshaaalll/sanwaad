"""Graph nodes. Each is a pure-ish async function: state in, state delta out."""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field

from ..caching import TRIAGE_CACHE
from ..config import JUDGE, MAX_CASE_COST_INR, REVIEW
from ..context import minimal_text, untrusted, with_trust_rules
from ..guardrails import check_complaint, check_reply
from ..llm import NON_CALL_MODELS, format_citations, structured
from ..models import (
    Category,
    Citation,
    Draft,
    GroundingVerdict,
    Priority,
    Review,
    Triage,
    VoiceOutcome,
)
from ..obs import TRACER
from ..rag.store import CATEGORY_CLAUSES, get_store
from ..router import route
from .state import GrievanceState, event

# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = """You triage inbound public comments for NimbusPay, an Indian UPI wallet.

Comments arrive in English, Hindi (Devanagari), Hinglish written in Latin script,
and other Indian languages. Read them as an Indian support lead would.

Set `severity` on customer impact, not on tone:
  1  mild annoyance, no money at stake
  2  routine issue, small amount, no deadline pressure
  3  real money blocked or lost, customer inconvenienced
  4  large amount, repeated contact, or account fully inaccessible
  5  regulatory, legal, fraud, or safety exposure

`summary` MUST be written in English regardless of the comment's language, in
one sentence, naming the amount and timeframe if present. This summary is used
verbatim as a retrieval query against an English policy index, so write it with
the vocabulary a policy document would use.

`language` is the BCP-47 tag of the language to REPLY in, mirroring the
customer. Hinglish in Latin script is "hi-Latn"."""

# Clause ESC-05: these never get auto-answered, whatever the model computes.
_REGULATORY_PATTERNS = re.compile(
    r"\b(rbi|ombudsman|consumer\s*court|legal\s*notice|lawyer|advocate|police|"
    r"fir|cyber\s*cell|media|journalist|defraud|fraud|scam|stolen)\b",
    re.IGNORECASE,
)


async def triage_node(state: GrievanceState) -> dict:
    complaint = state["complaint"]
    text = complaint["text"]

    with TRACER.span("triage", trace_id=state["case_id"]) as span:
        # INPUT guardrail. The complaint is untrusted text that is about to be
        # placed inside a prompt; anything instruction-shaped is flagged now so
        # the downstream router and the auto-post gate can both see it.
        text, input_violations = check_complaint(text)
        injection = [v.detail for v in input_violations if v.rule == "injection"]

        r = route("triage")
        span.set(model=r.model, injection=bool(injection))

        cached = TRIAGE_CACHE.get(r.model, text, namespace="triage")
        if cached is not None:
            span.set(cache="hit")
            result = Triage(**cached)
            cost = {"stage": "triage", "model": "cache", "usd": 0.0, "inr": 0.0,
                    "prompt_tokens": 0, "output_tokens": 0, "attempts": 0}
        else:
            span.set(cache="miss")
            fallback = _offline_triage(text)
            # Minimal, isolated context: no author handle, no identifiers, and
            # the comment fenced off as data rather than pasted in as prose.
            result, cost = await structured(
                model=r.model,
                fallback_model=r.fallback,
                system=with_trust_rules(_TRIAGE_SYSTEM),
                user=(f"A public comment on {complaint['channel']}:\n\n"
                      + untrusted("customer_comment", minimal_text(text))),
                schema=Triage,
                stage="triage",
                offline_fallback=fallback,
                max_output_tokens=r.max_output_tokens,
                timeout_s=r.timeout_s,
                max_latency_ms=r.max_latency_ms,
                spent_inr=_spent(state),
                budget_inr=MAX_CASE_COST_INR,
                trace_id=state["case_id"],
            )
            TRIAGE_CACHE.put(r.model, text, result.model_dump(mode="json"), namespace="triage")
        span.set(cost_inr=cost.get("inr", 0.0))

    # Deterministic override. A keyword match here outranks the model, because
    # a missed regulatory mention is a compliance incident and a false positive
    # is merely a human reading one extra ticket.
    regulatory = bool(_REGULATORY_PATTERNS.search(text))
    if regulatory and result.severity < 5:
        result.severity = 5

    return {
        "triage": result.model_dump(mode="json"),
        "retrieval_query": result.summary,
        "injection_flagged": bool(injection),
        "costs": [cost],
        "events": [event(
            "triage",
            f"{result.category.value} / severity {result.severity} / {result.sentiment}",
            regulatory_flag=regulatory, injection=bool(injection),
        )],
    }


def _keyword_in(keyword: str, text: str) -> bool:
    """Latin keywords must start at a word boundary.

    Plain substring matching let "fee" fire inside "coffee", so a question
    about filter coffee was triaged as a billing complaint. The trajectory
    eval's out-of-scope scenario caught it; the golden set had carried that
    exact sentence all along without anything checking the triage step.
    Devanagari keeps substring matching, because its vowel signs are not word
    characters to a regex and a boundary would split words mid-letter.
    """
    if keyword.isascii():
        return re.search(r"(?<![a-z])" + re.escape(keyword), text) is not None
    return keyword in text


def _offline_triage(text: str) -> dict:
    """Keyword triage for keyless runs. Crude on purpose — it exists so the
    pipeline is demonstrable, not so it is good."""
    low = text.lower()
    # Keywords are listed in every script the mock feed actually contains.
    # An English-only list silently mis-routes Devanagari complaints to
    # off_topic, which looks like the pipeline is broken when it is only the
    # stub that is.
    pairs = [
        (("double debit", "twice", "duplicate", "do baar", "दो बार",
          "refund", "wapas", "reversal", "reverse", "come back", "not back",
          "money back", "paise nahi", "return", "रिफंड",
          "वापस", "पैसे नहीं", "पैसा नहीं"), Category.REFUND),
        (("charge", "charged", "deduct", "kata", "fee", "extra", "चार्ज",
          "कट गए", "कटे", "शुल्क"), Category.BILLING),
        (("freeze", "frozen", "block", "kyc", "login", "locked", "फ्रीज",
          "ब्लॉक", "बंद", "लॉक"), Category.ACCOUNT_ACCESS),
        # Payment-failure vocabulary. Without these the six-comment outage in
        # the mock feed triages as off_topic severity 1 and the crisis path is
        # unreachable offline — the stub hiding the feature it exists to show.
        (("down", "not working", "outage", "server", "डाउन", "चल नहीं",
          "failing", "failed", "fail ho", "stuck", "pending", "not received",
          "nahi mila", "फेल", "अटक"), Category.SERVICE_OUTAGE),
        (("rude", "behaviour", "misbehav", "बदतमीज़", "बदतमीज"),
         Category.AGENT_BEHAVIOUR),
        (("privacy", "otp", "phishing", "ओटीपी", "धोखा"), Category.DATA_PRIVACY),
        (("thank", "well done", "love", "clean", "शुक्रिया", "धन्यवाद"),
         Category.PRAISE),
    ]
    category = Category.OFF_TOPIC
    for keys, cat in pairs:
        if any(_keyword_in(k, low) for k in keys):
            category = cat
            break

    if _REGULATORY_PATTERNS.search(text):
        severity = 5
    elif category in (Category.OFF_TOPIC, Category.PRAISE):
        severity = 1
    else:
        # Money at stake drives severity, matching the real triage rubric.
        amount = _largest_rupee_amount(text) or 0
        severity = 4 if amount > 10_000 else 3

    devanagari = bool(re.search(r"[ऀ-ॿ]", text))
    return {
        "is_complaint": category not in (Category.OFF_TOPIC, Category.PRAISE),
        "category": category.value,
        "severity": severity,
        "sentiment": "positive" if category is Category.PRAISE else "frustrated",
        "language": "hi-IN" if devanagari else "en-IN",
        "summary": f"Customer reports an issue in the {category.value} category: {text[:160]}",
        "entities": {},
        "needs_private_data": category is not Category.PRAISE,
    }


# ---------------------------------------------------------------------------
# Pattern — what this looks like next to the others
# ---------------------------------------------------------------------------

async def pattern_node(state: GrievanceState) -> dict:
    """Score the complaint against the recent window.

    Runs before the judge because the judge needs a cross-case fact — how many
    other accounts posted near-identical wording — and only the window knows
    it. Neither agent can answer the other's question from one comment.
    """
    from ..pattern import detect

    complaint = state["complaint"]
    triage = state["triage"]

    posted_at = None
    try:
        from datetime import datetime, timezone

        posted_at = datetime.fromisoformat(complaint["created_at"])
        if posted_at.tzinfo is None:
            posted_at = posted_at.replace(tzinfo=timezone.utc)
    except (KeyError, ValueError):
        pass  # detect() falls back to now, which is right for a live comment

    with TRACER.span("pattern", trace_id=state["case_id"]) as span:
        signal, duplicates = detect(
            case_id=state["case_id"],
            author=complaint["author"],
            category=triage["category"],
            summary=triage["summary"],
            text=complaint["text"],
            at=posted_at,
        )
        span.set(level=signal.level, cluster=signal.cluster_size,
                 velocity=signal.velocity_per_hour, duplicates=duplicates)

    message = (
        f"{signal.level}: {signal.cluster_size} distinct author(s) in "
        f"{signal.window_minutes}m ({signal.velocity_per_hour}/hr)"
        if signal.level != "none"
        else "first of its kind in the window"
    )
    return {
        "pattern": signal.model_dump(mode="json"),
        "coordination": duplicates,
        "events": [event("pattern", message, theme=signal.theme,
                         related=signal.related_case_ids[:5])],
    }


# ---------------------------------------------------------------------------
# Judge — who is speaking
# ---------------------------------------------------------------------------

class AuthorSecondOpinion(BaseModel):
    """Only used for the ambiguous band; see judge.needs_a_second_opinion."""

    author_class: str = Field(description="customer, audience, troll, bot, competitor or unknown")
    authenticity: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


_JUDGE_SYSTEM = """You are deciding whether a public comment about NimbusPay, an
Indian UPI wallet, comes from someone with a real problem or from someone
performing outrage.

You are given the comment and the signals already computed from the account.
The signals were inconclusive, which is why you were called.

Judge the COMMENT, not its tone. Indian customers are direct, and anger is not
evidence of bad faith — a person who has lost ₹18,000 is entitled to be
furious. What separates a real grievance is whether anything in it could only
be known by someone it actually happened to.

Treat as REAL: a described sequence of events, a named merchant, an amount, a
timeframe, a prior support interaction, a specific broken screen.
Treat as PERFORMED: claims about the company's motives with no event attached,
insults that would fit under any brand's post, demands to boycott, and
copy-paste phrasing.

Set `authenticity` 0.0-1.0. Do not round to the middle to be safe; an
uncommitted score is the same as not being called."""


def triage_wants_a_reply(state: GrievanceState) -> bool:
    """Is there anything here a reply would address?"""
    triage = state.get("triage") or {}
    return bool(triage.get("is_complaint")) or int(triage.get("severity", 1)) >= 4


async def judge_node(state: GrievanceState) -> dict:
    """Read the author, then spend a model call only if the answer is close."""
    from ..judge import assess, needs_a_second_opinion
    from ..models import Complaint
    from ..pattern import get_store

    complaint = Complaint(**state["complaint"])
    history = get_store().author_history(complaint.author,
                                         exclude_case_id=state["case_id"])

    verdict = assess(
        complaint,
        history_with_brand=history,
        duplicate_authors=int(state.get("coordination") or 0),
    )

    cost = {"stage": "judge", "model": "rules", "usd": 0.0, "inr": 0.0,
            "prompt_tokens": 0, "output_tokens": 0, "attempts": 0}

    # Never buy an opinion we cannot act on: a compliment's author does not
    # need adjudicating, whatever the account looks like.
    worth_asking = triage_wants_a_reply(state) and needs_a_second_opinion(verdict)

    if worth_asking:
        r = route("judge")
        opinion, cost = await structured(
            model=r.model,
            fallback_model=r.fallback,
            system=with_trust_rules(_JUDGE_SYSTEM),
            user=(f"A comment on {complaint.channel.value}:\n"
                  + untrusted("customer_comment", minimal_text(complaint.text))
                  + "\n\nSignals already computed from the account (trusted):\n- "
                  + "\n- ".join(verdict.evidence or ["none"])),
            schema=AuthorSecondOpinion,
            temperature=0.0,
            stage="judge",
            max_output_tokens=r.max_output_tokens,
            timeout_s=r.timeout_s,
            max_latency_ms=r.max_latency_ms,
            spent_inr=_spent(state),
            budget_inr=MAX_CASE_COST_INR,
            trace_id=state["case_id"],
            offline_fallback={"author_class": verdict.author_class,
                              "authenticity": verdict.authenticity,
                              "reasoning": "offline mode: rules verdict kept"},
        )
        # The model may only move the verdict within the ambiguous band it was
        # called for. Letting one call overturn hard account evidence — a
        # ninety-minute-old throwaway — is how a persuasive troll gets promoted
        # to a priority customer.
        if opinion.author_class in ("customer", "audience", "troll", "bot",
                                    "competitor", "unknown"):
            verdict.author_class = opinion.author_class
        verdict.authenticity = round(
            max(JUDGE.ambiguous_low, min(JUDGE.ambiguous_high, opinion.authenticity)), 3)
        verdict.evidence.append(f"model: {opinion.reasoning[:160]}")
        verdict.reply_worthy = (
            verdict.author_class != "bot"
            and (verdict.authenticity >= JUDGE.reply_worthy_authenticity
                 or verdict.reach >= JUDGE.reply_worthy_reach)
        )

    with TRACER.span("judge", trace_id=state["case_id"]) as span:
        span.set(author_class=verdict.author_class, authenticity=verdict.authenticity,
                 reach=verdict.reach, history=history,
                 second_opinion=cost.get("model") != "rules")

    return {
        "verdict": verdict.model_dump(mode="json"),
        "costs": [cost],
        "events": [event(
            "judge",
            f"{verdict.author_class} / authenticity {verdict.authenticity} / "
            f"reach {verdict.reach}"
            + ("" if verdict.reply_worthy else " — not worth a drafted reply"),
            evidence=verdict.evidence,
        )],
    }


# ---------------------------------------------------------------------------
# Prioritise — fold three readings into one decision
# ---------------------------------------------------------------------------

_PATTERN_WEIGHT = {"none": 0.0, "watch": 15.0, "crisis": 40.0}


def prioritise(triage: dict, verdict: dict, pattern: dict) -> Priority:
    """How urgent is this, and is it worth drafting for at all?

    Severity says how bad it is for one person, the verdict says whose voice it
    is, the pattern says whether it is one of many. A queue ordered by any one
    of them alone is wrong in a way somebody notices: by severity alone the
    outage arrives as forty separate tickets; by reach alone the loudest
    account outranks the person who actually lost money.
    """
    reasons: list[str] = []
    level = pattern.get("level", "none")
    severity = int(triage.get("severity", 1))
    authenticity = float(verdict.get("authenticity", 0.5))
    reach = int(verdict.get("reach", 0))

    score = severity * 10.0 + authenticity * 8.0
    score += min(reach / 1000.0, 20.0)
    score += _PATTERN_WEIGHT.get(level, 0.0)
    if verdict.get("history_with_brand"):
        score += 5.0
        reasons.append("known customer with prior cases")

    if level == "crisis":
        reasons.append(f"crisis: {pattern.get('cluster_size')} authors reporting the same thing")
        return Priority(tier="crisis", score=round(score, 1), reasons=reasons, drafting=True)

    # Not worth drafting for. Note what this does NOT do: it does not delete
    # the case or mark it handled. It is recorded, counted by the pattern
    # agent, and visible in the console — because "we ignored it" and "we never
    # saw it" are different failures, and only one of them is defensible.
    if not verdict.get("reply_worthy", True):
        reasons.append(f"{verdict.get('author_class')} with reach {reach}: logged, not answered")
        return Priority(tier="ignore", score=round(score, 1), reasons=reasons, drafting=False)

    if not triage.get("is_complaint") and severity < 4:
        reasons.append("not a complaint")
        return Priority(tier="ignore", score=round(score, 1), reasons=reasons, drafting=False)

    if severity >= 4 or level == "watch" or reach >= JUDGE.reply_worthy_reach:
        if severity >= 4:
            reasons.append(f"severity {severity}")
        if level == "watch":
            reasons.append(f"{pattern.get('cluster_size')} similar in the last hour")
        if reach >= JUDGE.reply_worthy_reach:
            reasons.append(f"audience of {reach:,}")
        return Priority(tier="priority", score=round(score, 1), reasons=reasons, drafting=True)

    reasons.append("routine complaint")
    return Priority(tier="routine", score=round(score, 1), reasons=reasons, drafting=True)


async def prioritise_node(state: GrievanceState) -> dict:
    p = prioritise(state["triage"], state.get("verdict") or {}, state.get("pattern") or {})
    return {
        "priority": p.model_dump(mode="json"),
        "events": [event("prioritise", f"{p.tier} (score {p.score}): " + "; ".join(p.reasons))],
    }


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

async def retrieve_node(state: GrievanceState) -> dict:
    triage = state["triage"]
    store = get_store()

    query = state.get("retrieval_query") or state["complaint"]["text"]
    boost = CATEGORY_CLAUSES.get(triage["category"], ())
    span_cm = TRACER.span("retrieve", trace_id=state["case_id"], strategy="rrf")
    # RRF over dense + BM25. Measured at strict@5 1.00 on the golden set vs
    # 0.89 for the old score blend — see `python -m sanwaad.evals.retrieval`.
    with span_cm as span:
        retrieved = store.search_fused(query, k=5, boost_prefixes=boost)
        span.set(hits=[c.clause_id for c in retrieved])

    # Brand-voice and privacy clauses are constitutional, not retrieved.
    constitutional = store.always_on(("BV-", "PRV-"))

    seen, merged = set(), []
    for c in retrieved + constitutional:
        if c.clause_id not in seen:
            seen.add(c.clause_id)
            merged.append(c)

    return {
        "citations": [c.model_dump() for c in merged],
        "events": [event(
            "retrieve",
            f"{len(retrieved)} retrieved + {len(constitutional)} constitutional",
            retrieved=[c.clause_id for c in retrieved],
        )],
    }


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM = """You write public replies for NimbusPay on social media.

You are given the customer's comment and the exact policy clauses that govern
it. Follow these rules absolutely:

- Every factual claim, timeline, amount, or entitlement you state MUST come
  from a supplied clause. If the clauses do not answer something, say what you
  will check instead of guessing. Inventing a refund timeline is the single
  worst thing you can do here.
- List in `citations` the id of every clause you actually relied on. Do not
  list clauses you did not use.
- Reply in the language given by `reply_language`, mirroring the customer.
  For "hi-Latn", write Hinglish in Latin script the way Indians actually type.
- Under 60 words. This is a public thread, not an email.
- Never repeat back any phone number, account number, transaction id or UTR.
- End with exactly one concrete next step.
- Set `promises_compensation` true if your text commits to any money moving:
  a refund, a credit, a waiver, or a compensation amount."""


async def draft_node(state: GrievanceState) -> dict:
    triage = state["triage"]
    citations = [Citation(**c) for c in state["citations"]]
    revision = state.get("revision_count", 0)

    correction = ""
    grounding = state.get("grounding")
    if grounding and grounding.get("unsupported_claims"):
        correction = (
            "\n\nYour previous draft was REJECTED as ungrounded. These claims are "
            "not supported by any clause and must be removed or replaced with a "
            "commitment to check:\n- "
            + "\n- ".join(grounding["unsupported_claims"])
        )

    # Tell the writer what the pattern agent knows. Without this the reply says
    # "let me look into your case" to the sixtieth person reporting one outage,
    # which reads as a brand that has not noticed its own incident.
    pattern = state.get("pattern") or {}
    incident = ""
    if pattern.get("level") in ("watch", "crisis"):
        incident = (
            f"\n\nKNOWN INCIDENT: {pattern.get('cluster_size')} other people have "
            f"reported the same thing in the last {pattern.get('window_minutes')} minutes. "
            "Acknowledge it as something we are already on, in one clause. Do not "
            "give a fix time unless a clause states one, and do not imply this "
            "person is the only one affected."
        )

    # The comment AND the triage summary are untrusted: the summary was written
    # by a model reading the comment, so anything the comment smuggled in can
    # survive into it. Only the clauses are ours.
    comment_block = untrusted("customer_comment", minimal_text(state["complaint"]["text"]))
    summary_block = untrusted("triage_summary", triage["summary"])
    user = f"""Customer comment ({triage['sentiment']}, severity {triage['severity']}):
{comment_block}

What they are reporting:
{summary_block}
reply_language: {triage['language']}

Governing clauses (trusted policy):
{format_citations(citations)}{incident}{correction}"""

    r = route("draft", severity=triage["severity"], revision=revision,
              injection_flagged=bool(state.get("injection_flagged")),
              language=triage["language"])

    result, cost = await structured(
        model=r.model,
        fallback_model=r.fallback,
        system=with_trust_rules(_DRAFT_SYSTEM),
        user=user,
        schema=Draft,
        temperature=0.3,
        stage=f"draft{'_revision' if revision else ''}",
        max_output_tokens=r.max_output_tokens,
        timeout_s=r.timeout_s,
        max_latency_ms=r.max_latency_ms,
        spent_inr=_spent(state),
        budget_inr=MAX_CASE_COST_INR,
        trace_id=state["case_id"],
        offline_fallback={
            "text": (
                "Sorry about this — that is a genuinely frustrating position to be in. "
                "I'm pulling up the details on the transaction you mentioned now. "
                "Sending you a DM so we can check it properly without your details "
                "being public. — Adhik, ref " + state["case_id"][-6:]
            ),
            "citations": [c.clause_id for c in citations[:3]],
            "promises_compensation": False,
        },
    )

    # OUTPUT guardrail, before the model-based grounding check. Deterministic,
    # free, and it repairs PII in place rather than discarding a good reply.
    guard = check_reply(result.text)
    result.text = guard.text
    if guard.blocked:
        result.promises_compensation = True  # forces the human gate below

    with TRACER.span("draft", trace_id=state["case_id"]) as span:
        span.set(model=r.model, route_reason=r.reason, revision=revision,
                 cost_inr=cost.get("inr", 0.0), words=len(result.text.split()),
                 guardrail_blocked=guard.blocked,
                 violations=[v.rule for v in guard.violations])

    return {
        "draft": result.model_dump(mode="json"),
        "revision_count": revision + 1,
        "guardrails": [v.__dict__ for v in guard.violations],
        "costs": [cost],
        "events": [event(
            "draft",
            f"{'revision ' + str(revision) if revision else 'first draft'} via {r.model}, "
            f"{len(result.text.split())} words, cites {result.citations}"
            + (f" — GUARDRAIL {[v.rule for v in guard.violations]}" if guard.violations else ""),
        )],
    }


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

_GROUND_SYSTEM = """You are a compliance checker. You are given a proposed public
reply and the policy clauses available to its author.

For each factual claim in the reply — timelines, amounts, entitlements,
processes, guarantees — decide whether a supplied clause supports it.

List in `unsupported_claims` every claim with no clause behind it. Be strict:
"within 3 working days" is unsupported unless a clause states that window.

Do NOT flag: apologies, empathy, offers to check, offers to DM, or requests
for information. Those commit us to nothing.

`grounded` is true only when `unsupported_claims` is empty."""


async def ground_check_node(state: GrievanceState) -> dict:
    draft = state["draft"]
    citations = [Citation(**c) for c in state["citations"]]

    r = route("ground_check")
    result, cost = await structured(
        model=r.model,
        fallback_model=r.fallback,
        system=with_trust_rules(_GROUND_SYSTEM),
        user=("Proposed reply:\n" + untrusted("model_draft", draft["text"])
              + f"\n\nAvailable clauses (trusted policy):\n{format_citations(citations)}"),
        schema=GroundingVerdict,
        temperature=0.0,
        stage="ground_check",
        max_output_tokens=r.max_output_tokens,
        timeout_s=r.timeout_s,
        max_latency_ms=r.max_latency_ms,
        spent_inr=_spent(state),
        budget_inr=MAX_CASE_COST_INR,
        trace_id=state["case_id"],
        offline_fallback={"grounded": True, "unsupported_claims": [],
                          "reasoning": "offline mode: not verified"},
    )

    if cost.get("degraded"):
        # The offline default says "grounded" so keyless demos flow. A checker
        # that could not run in production has verified nothing, and
        # "unverified" must never be read as "grounded". Send it to a person
        # rather than looping the drafter against a checker that is down.
        result = GroundingVerdict(grounded=False, unavailable=True,
                                  unsupported_claims=["grounding check could not run"],
                                  reasoning="every model attempt failed")

    return {
        "grounding": result.model_dump(mode="json"),
        "costs": [cost],
        "events": [event(
            "ground_check",
            "grounded" if result.grounded
            else f"{len(result.unsupported_claims)} unsupported claim(s)",
            unsupported=result.unsupported_claims,
        )],
    }


# ---------------------------------------------------------------------------
# Plan — the ghostwriter's private half: propose the fix, validate it in code
# ---------------------------------------------------------------------------

class PlannedAction(BaseModel):
    kind: str = Field(description="reversal or ticket")
    reference: Optional[str] = Field(default=None, description="exact ledger reference, reversals only")
    amount_inr: Optional[float] = Field(default=None, description="exact ledger amount, reversals only")
    reason: str = Field(description="one plain sentence a reviewer can check")


class ActionPlan(BaseModel):
    actions: list[PlannedAction] = Field(default_factory=list)
    reasoning: str = ""


_PLAN_SYSTEM = """You propose the internal actions that would actually resolve a
customer's complaint for NimbusPay, an Indian UPI wallet. You do not execute
anything. Every proposal is checked by code against the ledger and approved by
a person before anything happens, so propose what the facts support and nothing
more.

Available actions:
- reversal: return ONE specific debit. Only for a transaction that appears in
  the ledger results you are given, using its exact reference and amount.
- ticket: open an internal ticket so a person follows up — when the fix is not
  a reversal, or when the ledger could not be checked.

Rules:
- Never invent a reference or an amount. No ledger match, no reversal.
- An ordinary settled payment is not reversible. A duplicate debit, or a failed
  transfer whose money has not come back, may be.
- `reason` is one plain sentence."""


def _offline_plan(matches: list[dict], tool_errors: list[str], amounts: list[float]) -> dict:
    """What a careful planner proposes, written as rules for keyless runs.

    It proposes on any problem-shaped transaction — including ones the
    validator will then refuse. That is deliberate: an offline stub that only
    proposed valid actions would never exercise the validator, and the
    validator is the part that has to hold when a real model gets it wrong.
    """
    actions = []
    for m in matches:
        if m["kind"] in ("duplicate_debit", "failed_transfer", "failed_payment"):
            amount = next((a for a in amounts if abs(a - m["amount_inr"]) < 0.01), m["amount_inr"])
            actions.append({
                "kind": "reversal", "reference": m["reference"], "amount_inr": amount,
                "reason": f"{m['kind'].replace('_', ' ')} of ₹{m['amount_inr']:,.0f} at {m['merchant']}",
            })
    if tool_errors:
        actions.append({"kind": "ticket",
                        "reason": "Ledger unavailable while planning; check the transaction manually"})
    elif amounts and not matches:
        actions.append({"kind": "ticket",
                        "reason": "No matching transaction for this customer; ask for the reference privately"})
    return {"actions": actions, "reasoning": "offline mode: rule-based plan"}


async def plan_node(state: GrievanceState) -> dict:
    """Gather facts, let the model propose, then validate every proposal in code.

    The ledger is searched BEFORE the model is asked, with keys the system
    controls — the author's handle, and amounts and references parsed from the
    comment. The model reasons over facts it was handed; it is not given a tool
    and trusted to look things up well.
    """
    from ..actions import make_action, references_in, validate_action
    from ..config import ACTIONS
    from ..tools import REGISTRY

    complaint, triage, case_id = state["complaint"], state["triage"], state["case_id"]
    author, text = complaint["author"], complaint["text"]

    # 1. Context, through the READ tool.
    references = references_in(text)
    amounts = _rupee_amounts(text)
    queries = [{"handle": author, "reference": ref} for ref in references]
    queries += [{"handle": author, "amount_inr": a} for a in amounts]

    matches: dict[str, dict] = {}
    tool_errors: list[str] = []
    for q in queries:
        result = await REGISTRY.call("lookup_transaction", q, agent="plan", trace_id=case_id)
        if result.ok:
            for m in result.output["matches"]:
                matches[m["reference"]] = m
        else:
            tool_errors.append(f"{result.error.code.value}: {result.error.message}")

    # 2. The model proposes.
    r = route("plan")
    ledger_view = json.dumps(list(matches.values()), ensure_ascii=False)
    plan, cost = await structured(
        model=r.model,
        fallback_model=r.fallback,
        system=with_trust_rules(_PLAN_SYSTEM),
        user=(f"Complaint category: {triage['category']}, severity {triage['severity']}.\n\n"
              + untrusted("customer_comment", minimal_text(text))
              + "\n\nLedger results for this customer:\n"
              + untrusted("tool_lookup_transaction", ledger_view)
              + (f"\n\nThe ledger could not be checked: {'; '.join(tool_errors)}"
                 if tool_errors else "")),
        schema=ActionPlan,
        temperature=0.0,
        stage="plan",
        offline_fallback=_offline_plan(list(matches.values()), tool_errors, amounts),
        max_output_tokens=r.max_output_tokens,
        timeout_s=r.timeout_s,
        max_latency_ms=r.max_latency_ms,
        spent_inr=_spent(state),
        budget_inr=MAX_CASE_COST_INR,
        trace_id=case_id,
    )

    # 3. Code types, deduplicates and validates every proposal.
    proposals, seen = [], set()

    def add(kind: str, reason: str, reference=None, amount=None, proposed_by="plan") -> None:
        action = make_action(kind, case_id=case_id, reason=reason, reference=reference,
                             amount_inr=amount, proposed_by=proposed_by)
        if action.id in seen:
            return
        seen.add(action.id)
        validation = validate_action(action, author=author)
        proposals.append({"proposal": action.model_dump(mode="json"),
                          "validation": validation.model_dump(mode="json")})

    for item in plan.actions:
        if item.kind in ("reversal", "ticket"):
            add(item.kind, item.reason, item.reference, item.amount_inr)

    # A rule that must always hold does not depend on the model remembering it.
    if (triage["severity"] >= ACTIONS.ticket_min_severity
            and not any(p["proposal"]["kind"] == "ticket" for p in proposals)):
        add("ticket", f"Severity {triage['severity']} {triage['category']} complaint: follow up",
            proposed_by="rules")

    summary = [
        f"{p['proposal']['kind']}"
        + (f" {p['proposal']['reference']} ₹{p['proposal']['amount_inr']:,.0f}"
           if p["proposal"]["reference"] and p["proposal"]["amount_inr"] else "")
        + (" valid" if p["validation"]["ok"] else
           " BLOCKED: " + ", ".join(c["name"] for c in p["validation"]["checks"] if not c["passed"]))
        for p in proposals
    ]
    message = "; ".join(summary) if summary else "no action proposed"
    if tool_errors:
        message += f" (ledger lookup failed: {tool_errors[0]})"
    return {
        "actions": proposals,
        "costs": [cost],
        "events": [event("plan", message, tool_errors=tool_errors)],
    }


# ---------------------------------------------------------------------------
# Review gate
# ---------------------------------------------------------------------------

def auto_post_allowed(state: GrievanceState) -> tuple[bool, str]:
    """Decide whether this reply may go out without a human.

    Returns (allowed, reason). The reason is logged either way, because
    "why did this need a human" is the question a pilot customer asks most.
    """
    triage = state["triage"]
    draft = state["draft"]
    grounding = state.get("grounding") or {}

    # Ordered by consequence, so the reason a reviewer reads first is the one
    # that matters most when several apply at once.
    if any(a["proposal"]["risk"] == "write_high" for a in (state.get("actions") or [])):
        # Including proposals that failed validation: a refused attempt to move
        # someone else's money is exactly what a person should see.
        return False, "a money-moving action is proposed; a human approves the exact action"
    if state.get("injection_flagged"):
        return False, "complaint contains instruction-like text; never auto-post"
    if (state.get("pattern") or {}).get("level") == "crisis":
        # During an incident the individually-correct reply is the dangerous
        # one: forty auto-posted apologies with slightly different wording IS
        # the screenshot. One human decides the line, then everything uses it.
        return False, "crisis pattern detected; incident replies go out under one human line"
    blocking = [g for g in (state.get("guardrails") or []) if g.get("severity") == "block"]
    if blocking:
        return False, f"guardrail block: {[g['rule'] for g in blocking]}"
    if REVIEW.forbid_auto_compensation and draft.get("promises_compensation"):
        return False, "draft commits money; clause RFD-05 requires approval"
    if triage["severity"] > REVIEW.auto_post_max_severity:
        return False, f"severity {triage['severity']} above auto-post ceiling {REVIEW.auto_post_max_severity}"
    if REVIEW.require_grounded and not grounding.get("grounded"):
        return False, "draft contains unsupported claims"
    if triage.get("needs_private_data"):
        return False, "resolution needs account data not available publicly"
    return True, "low severity, fully grounded, commits nothing"


async def review_gate_node(state: GrievanceState) -> dict:
    allowed, reason = auto_post_allowed(state)
    if allowed:
        review = Review(decision="approve", reviewer="auto", note=reason, auto=True,
                        final_text=state["draft"]["text"])
        return {
            "review": review.model_dump(mode="json"),
            "events": [event("review_gate", f"auto-approved: {reason}")],
        }

    # Human needed. LangGraph's interrupt suspends the run durably; the case
    # sits in the review queue until the API resumes it with a decision, which
    # may be minutes or days later, in a different process.
    from langgraph.types import interrupt

    decision = interrupt({
        "case_id": state["case_id"],
        "reason": reason,
        "complaint": state["complaint"]["text"],
        "draft": state["draft"]["text"],
        "citations": [c["clause_id"] for c in state["citations"]],
        "triage": state["triage"],
        "grounding": state.get("grounding"),
        "actions": state.get("actions") or [],
    })

    review = Review(**decision) if isinstance(decision, dict) else Review(decision="reject")
    if review.decision == "approve" and not review.final_text:
        review.final_text = state["draft"]["text"]

    # Capture the correction while it exists. This is the only moment the pair
    # (what the model wrote, what a human actually sends) is available.
    try:
        from ..feedback import FeedbackRecord
        from ..feedback import record as record_feedback

        record_feedback(FeedbackRecord(
            case_id=state["case_id"], decision=review.decision,
            complaint=state["complaint"]["text"], draft=state["draft"]["text"],
            final=review.final_text or "", category=state["triage"]["category"],
            severity=state["triage"]["severity"], language=state["triage"]["language"],
            citations=state["draft"].get("citations", []),
            reviewer=review.reviewer, note=review.note,
        ))
    except Exception as exc:  # feedback capture must never block a review
        from loguru import logger

        logger.warning(f"feedback capture failed for {state['case_id']}: {exc}")

    return {
        "review": review.model_dump(mode="json"),
        "events": [event("review_gate", f"human {review.decision} ({reason})")],
    }


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

async def publish_node(state: GrievanceState) -> dict:
    from ..tools import REGISTRY

    review = state["review"]
    complaint = state["complaint"]

    # ACTION guardrail: last check before an irreversible public post. A human
    # can paste anything into the edit box, so the reviewer's text is checked
    # exactly like the model's.
    final_guard = check_reply(review["final_text"] or "")
    if final_guard.blocked:
        return {
            "published": {"blocked": True,
                          "reasons": [v.detail for v in final_guard.violations]},
            "events": [event("publish", "BLOCKED at the action guardrail: "
                             + "; ".join(v.detail for v in final_guard.violations))],
        }

    # Posting is a high-risk write. The approval is whoever cleared the review
    # gate — the auto-post policy or a person — bound to this exact text.
    args = {"channel": complaint["channel"], "external_id": complaint["external_id"],
            "text": final_guard.text}
    approval = REGISTRY.approval_for(
        "post_reply", args, by=review.get("reviewer", "auto"),
        human=not review.get("auto", False), note=review.get("note", ""))
    result = await REGISTRY.call("post_reply", args, agent="publish", approval=approval,
                                 trace_id=state["case_id"])
    if not result.ok:
        reason = f"{result.error.code.value}: {result.error.message}"
        return {
            "published": {"blocked": True, "reasons": [reason]},
            "events": [event("publish", f"NOT posted — {reason}")],
        }

    receipt = result.output
    return {
        "published": receipt,
        "events": [event("publish", f"posted to {complaint['channel']}", **receipt)],
    }


# ---------------------------------------------------------------------------
# Act — the executor, which trusts neither the planner nor the reviewer
# ---------------------------------------------------------------------------

async def act_node(state: GrievanceState) -> dict:
    """Carry out approved actions, re-validating each one first.

    Validation already ran at planning time, and a human has looked since.
    It runs again anyway: the ledger may have changed in the hours a case sat
    in the queue, another case may have reversed the same debit, and an
    executor that trusts an earlier layer's conclusion is only as safe as the
    most stale thing it trusts.
    """
    from ..actions import ProposedAction, tool_args, validate_action
    from ..tools import REGISTRY

    review = state.get("review") or {}
    decisions = review.get("actions") or {}
    author = state["complaint"]["author"]
    results, events = [], []

    for item in state.get("actions") or []:
        action = ProposedAction(**item["proposal"])
        outcome = {"action_id": action.id, "kind": action.kind, "tool": action.tool,
                   "risk": action.risk, "reference": action.reference,
                   "amount_inr": action.amount_inr}

        if action.risk == "write_high" and (review.get("auto") or decisions.get(action.id) != "approve"):
            outcome.update(status="not_approved", detail="no human approval for this action")
        else:
            validation = validate_action(action, author=author)
            if not validation.ok:
                outcome.update(status="blocked",
                               detail="; ".join(c.detail for c in validation.failed()))
            else:
                args = tool_args(action, triage=state["triage"])
                approval = None
                if action.risk == "write_high":
                    approval = REGISTRY.approval_for(
                        action.tool, args, by=review.get("reviewer", "human"), human=True,
                        note=review.get("note", ""))
                result = await REGISTRY.call(action.tool, args, agent="act",
                                             approval=approval, trace_id=state["case_id"])
                if result.ok:
                    outcome.update(status="executed", output=result.output,
                                   attempts=result.attempts)
                else:
                    outcome.update(status="failed",
                                   detail=f"{result.error.code.value}: {result.error.message}")

        results.append(outcome)
        label = f"{action.kind}{' ' + action.reference if action.reference else ''}"
        events.append(event("act", f"{label} — {outcome['status']}"
                            + (f": {outcome['detail']}" if outcome.get("detail") else "")))

    if not events:
        events.append(event("act", "nothing to do"))
    return {"action_results": results, "events": events}


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class EscalationDecision(BaseModel):
    needed: bool
    reason: str
    suggested_channel: str = Field(default="webrtc", description="webrtc or exotel")


async def escalation_node(state: GrievanceState) -> dict:
    """Apply clause ESC-02 rather than asking a model to remember it.

    The escalation triggers are written down and enumerable, so they are code.
    A model is not more accurate here, only more expensive and less auditable.
    """
    triage = state["triage"]
    text = state["complaint"]["text"]
    reasons = []

    if triage["severity"] >= REVIEW.escalate_to_voice_min_severity:
        reasons.append(f"severity {triage['severity']} (ESC-02)")
    if triage["category"] in ("account_access", "data_privacy"):
        reasons.append(f"{triage['category']} always escalates (ESC-02)")
    amount = _largest_rupee_amount(text)
    if amount and amount > 2000:
        reasons.append(f"disputed amount ₹{amount:,.0f} exceeds ₹2,000 (ESC-02)")
    if re.search(r"\b(call me|baat kar|phone|speak to|talk to someone)\b", text, re.I):
        reasons.append("customer explicitly asked to speak to someone (ESC-02)")

    # A verified customer caught in a live incident gets the call, whatever
    # their individual severity says. Their problem is not small; it is early.
    pattern = state.get("pattern") or {}
    verdict = state.get("verdict") or {}
    if pattern.get("level") == "crisis" and verdict.get("author_class") == "customer":
        reasons.append(f"crisis cluster of {pattern.get('cluster_size')} and a confirmed customer")

    needed = bool(reasons)
    return {
        "escalation": {
            "needed": needed,
            "reasons": reasons,
            "channel": "webrtc" if needed else "none",
        },
        "events": [event(
            "escalation",
            "voice offered: " + "; ".join(reasons) if needed else "no voice escalation needed",
        )],
    }


_RUPEE_RE = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def _rupee_amounts(text: str) -> list[float]:
    values = []
    for m in _RUPEE_RE.finditer(text or ""):
        try:
            values.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    return sorted(set(values))


def _spent(state: GrievanceState) -> float:
    """What this case has cost in models so far.

    Read from the accumulated cost entries rather than tracked separately, so
    the ceiling is enforced against the same number the closure reports. Two
    counters would eventually disagree, and the one the ceiling used would be
    the one nobody was looking at.
    """
    return sum(float(c.get("inr", 0) or 0) for c in state.get("costs") or [])



def _largest_rupee_amount(text: str) -> Optional[float]:
    values = []
    for m in _RUPEE_RE.finditer(text):
        try:
            values.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    return max(values) if values else None


# ---------------------------------------------------------------------------
# Voice handoff
# ---------------------------------------------------------------------------

async def voice_node(state: GrievanceState) -> dict:
    """Suspend until a voice call completes.

    The graph does not place the call. It publishes everything the voice agent
    needs — citations included — and waits. Whether the call arrives over
    WebRTC from a browser or over Exotel from a phone is the transport's
    problem, not the state machine's.
    """
    from langgraph.types import interrupt

    outcome = interrupt({
        "await": "voice_call",
        "case_id": state["case_id"],
        "channel": state["escalation"]["channel"],
        "brief": {
            "summary": state["triage"]["summary"],
            "language": state["triage"]["language"],
            "category": state["triage"]["category"],
            "public_reply": state["review"]["final_text"],
            "clause_ids": [c["clause_id"] for c in state["citations"]],
            # The call must not contradict what the system already did, any
            # more than it may contradict what it already said.
            "actions_taken": [
                f"{r['kind']}{' ' + r['reference'] if r.get('reference') else ''}: {r['status']}"
                for r in (state.get("action_results") or [])
            ],
        },
    })

    result = VoiceOutcome(**outcome) if isinstance(outcome, dict) else VoiceOutcome()
    return {
        "voice": result.model_dump(mode="json"),
        "events": [event(
            "voice",
            f"{result.channel} call, {result.duration_s:.0f}s, "
            f"{'resolved' if result.resolved else 'unresolved'}",
            citations_used=result.citations_used,
        )],
    }


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------

async def close_node(state: GrievanceState) -> dict:
    """Roll up cost and emit the consistency receipt."""
    from ..consistency import build_receipt

    costs = state.get("costs", [])
    total_usd = sum(c.get("usd", 0.0) for c in costs)
    total_inr = sum(c.get("inr", 0.0) for c in costs)

    voice = state.get("voice") or {}
    receipt = build_receipt(
        text_clauses=(state.get("draft") or {}).get("citations", []),
        voice_clauses=voice.get("citations_used", []),
    )

    resolved = bool(voice.get("resolved")) or not (state.get("escalation") or {}).get("needed")

    # Count attempts, not cost entries. A call that needed a schema retry and a
    # fallback is three calls on the invoice; the rules-based judge is none.
    llm_calls = sum(
        int(c["attempts"]) if "attempts" in c
        else (0 if c.get("model") in NON_CALL_MODELS else 1)
        for c in costs
    )

    results = state.get("action_results") or []
    actions = {
        status: [r["action_id"] for r in results if r.get("status") == status]
        for status in ("executed", "blocked", "not_approved", "failed")
    }

    return {
        "closure": {
            "resolved": resolved,
            "total_cost_usd": round(total_usd, 6),
            "total_cost_inr": round(total_inr, 4),
            "llm_calls": llm_calls,
            "degraded_steps": [c["stage"] for c in costs if c.get("degraded")],
            "actions": actions,
            "consistency": receipt.model_dump(),
        },
        "events": [event(
            "close",
            f"closed {'resolved' if resolved else 'unresolved'}, ₹{total_inr:.4f}, "
            f"channels {'consistent' if receipt.consistent else 'DIVERGED'}",
        )],
    }
