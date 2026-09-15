"""Loop policies: what decides the next action on each pass.

In the loop-engineering era the model writes its own next step and pulls its
own context through tool calls. The input is no longer the scarce skill; the
loop around it is. So a policy is deliberately a small interface — given the
rendered context, return one structured decision — and everything that makes
the loop safe lives in the kernel, not here.

Two policies implement it:

- `ModelPolicy` asks Gemini for the decision, with the customer's text and
  every tool result inside an untrusted block.
- `ScriptedSupportPolicy` is the offline stand-in, so the loop, its evals and
  the MINT ladder all run without a key. It reads ONLY the rendered view, the
  same text a model would see — so when compaction drops a step, it really
  loses it, and memory has to earn its place. It is also deliberately *eager*,
  like a real model: it drafts as soon as it has transaction facts, including
  the timeline it "remembers", and relies on the in-loop verifier to send it
  to the policy first. Its remembered timeline for a failed transfer (five to
  seven days) is the common folk answer, and wrong: the policy says three
  working days. That is exactly the mistake cheap in-loop verification is for.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from ..config import TIER_DRAFT, TIER_REASONING
from ..context import untrusted, with_trust_rules
from ..llm import is_offline, structured
from .kernel import CONSULT_POLICY, FINAL, HANDOFF, PROPOSE_REVERSAL, Decision, LoopContext
from .verify import FinalAnswer

# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------

INTENTS = ("refund_status", "refund_request", "payment_status", "policy_question",
           "manipulation", "other")

_MANIPULATION = re.compile(
    r"ignore (?:all )?(?:previous|prior) instructions|you are now|system prompt|developer mode", re.I)
_POLICY_Q = re.compile(
    r"\b(how long|how many days|kitne din|what is (?:the|your) policy|policy (?:on|for)|rules? for)\b", re.I)
_STATUS = re.compile(
    r"\b(where is|status|when will|kab|yet|still|not (?:received|credited|back)|come back|came back)\b", re.I)
_REFUND = re.compile(
    r"\b(refund|reverse|reversal|double debit|debited twice|charged twice|taken twice|"
    r"deducted twice|money back|wapas|come back|not back)\b", re.I)
_PAYMENT = re.compile(r"\b(pending|stuck|went through|payment status|failed|failing)\b", re.I)
_AMOUNT = re.compile(r"(?:₹|rs\.?\s?|inr\s?)([\d,]+)", re.I)
_REFERENCE = re.compile(r"\bNP-TXN-[A-Z0-9-]+\b")


def classify_intent(message: str) -> str:
    text = message or ""
    if _MANIPULATION.search(text):
        return "manipulation"
    has_money = bool(_AMOUNT.search(text) or _REFERENCE.search(text))
    if _POLICY_Q.search(text) and not has_money:
        return "policy_question"
    if _PAYMENT.search(text) and has_money and not _REFUND.search(text):
        return "payment_status"
    if _REFUND.search(text):
        return "refund_status" if _STATUS.search(text) else "refund_request"
    if _POLICY_Q.search(text):
        return "policy_question"
    return "other"


def message_amount(message: str) -> Optional[float]:
    m = _AMOUNT.search(message or "")
    return float(m.group(1).replace(",", "")) if m else None


def message_reference(message: str) -> Optional[str]:
    m = _REFERENCE.search(message or "")
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Reading the view the way a model would have to
# ---------------------------------------------------------------------------

_TXN = re.compile(
    r"(NP-TXN-[A-Z0-9-]+): (payment|duplicate debit|failed transfer|failed payment) ₹([\d,]+)"
    r"(?: at ([^,\n]+))?(?:, (settled|failed not reversed|pending|reversed))?(?:, ([\d.]+)d old)?")
_REV = re.compile(r"(NP-TXN-[A-Z0-9-]+): reversal (REV-[A-F0-9]{8}) (initiated|credited)")
_NO_REV = re.compile(r"(NP-TXN-[A-Z0-9-]+): no reversal on record")
_CLAUSE = re.compile(r"\[((?:BV|RFD|BIL|KYC|ESC|PRV)-\d{2})\] ([A-Z][^:\n]{2,40}): ([^\n]*)")
_SUBAGENT = re.compile(r"policy sub-agent ((?:\[[A-Z]+-\d{2}\])+): ([^\n]+)")
_TICKET = re.compile(r"ticket (TKT-[A-F0-9]{8})")
_REFUSED = re.compile(r"'([a-z_]+)' is not an available action")
_CLAUSE_TIMELINE = re.compile(r"\*\*(\d+\s+(?:working\s+)?(?:hours|days))\*\*")


@dataclass
class Seen:
    txns: dict[str, dict] = field(default_factory=dict)
    reversals: dict[str, tuple[str, str]] = field(default_factory=dict)
    no_reversal: set[str] = field(default_factory=set)
    clauses: dict[str, str] = field(default_factory=dict)       # id -> snippet
    ticket: Optional[str] = None
    refused: set[str] = field(default_factory=set)
    looked_up: bool = False
    lookup_failed: bool = False
    searched: bool = False
    consulted: bool = False
    timeline_rejected: bool = False
    proposal_refused: Optional[str] = None
    proposal_without_handoff: bool = False


def read_view(view: str) -> Seen:
    s = Seen()
    for m in _TXN.finditer(view):
        s.txns[m.group(1)] = {
            "reference": m.group(1), "kind": m.group(2), "amount": float(m.group(3).replace(",", "")),
            "merchant": (m.group(4) or "").strip(), "status": m.group(5), "age": float(m.group(6)) if m.group(6) else None,
        }
    s.reversals = {m.group(1): (m.group(2), m.group(3)) for m in _REV.finditer(view)}
    s.no_reversal = set(_NO_REV.findall(view))
    s.clauses = {m.group(1): m.group(3) for m in _CLAUSE.finditer(view)}
    for m in _SUBAGENT.finditer(view):
        for cid in re.findall(r"[A-Z]+-\d{2}", m.group(1)):
            s.clauses.setdefault(cid, m.group(2))
    ticket = _TICKET.search(view)
    s.ticket = ticket.group(1) if ticket else None
    s.refused = set(_REFUSED.findall(view))
    # A remembered transaction counts as looked up even after the step that
    # found it has been compacted away — otherwise memory would buy a re-lookup.
    s.looked_up = "lookup_transaction" in view or "no matching transaction" in view or bool(s.txns)
    s.lookup_failed = "lookup_transaction failed" in view
    s.searched = "search_policy" in view
    s.consulted = "policy sub-agent" in view
    s.timeline_rejected = "stated a timeline" in view
    refused = re.search(r"Reversal proposal refused: ([^\n]+)", view)
    s.proposal_refused = refused.group(1) if refused else None
    s.proposal_without_handoff = "no human hand-over" in view
    return s


# ---------------------------------------------------------------------------
# The offline policy
# ---------------------------------------------------------------------------

_POLICY_QUERY = {
    "duplicate debit": ("RFD-06", "duplicate debit reversal time", "within 24 hours"),
    "failed transfer": ("RFD-01", "failed transfer auto reversal window", "within 5 to 7 days"),
}


def _final(text: str, cited: Optional[list[str]] = None, thought: str = "answer the customer") -> tuple[Decision, dict]:
    return Decision(thought=thought, action=FINAL,
                    answer=FinalAnswer(text=text, cited_clauses=cited or [])), {}


def _act(action: str, args: dict, thought: str) -> tuple[Decision, dict]:
    return Decision(thought=thought, action=action, args=args), {}


def _money(amount: float) -> str:
    return f"₹{amount:,.0f}"


class ScriptedSupportPolicy:
    name = "scripted"

    async def decide(self, view: str, ctx: LoopContext) -> tuple[Decision, dict]:
        intent = classify_intent(ctx.message)
        tools = set(ctx.config.tools)
        s = read_view(view)
        amount = message_amount(ctx.message)
        reference = message_reference(ctx.message)

        if intent == "manipulation":
            return _final("I can only help with your own NimbusPay payments, refunds and reversals. "
                          "Tell me which transaction you mean and I'll check it.")
        if intent == "other":
            return _final("I can help with NimbusPay payments, refunds and reversals. "
                          "Tell me the amount or the transaction you mean.")
        if intent == "policy_question":
            return self._policy_answer(ctx, s, tools)

        # --- anything about money starts with the ledger ----------------------
        if s.lookup_failed and not s.txns:
            if "open_ticket" in tools and not s.ticket and "open_ticket" not in s.refused:
                return _act("open_ticket", {
                    "category": "refund", "severity": 3,
                    "summary": "Ledger lookup failed during a support conversation; check the transaction manually",
                }, "the ledger is down: hand the check to a person instead of guessing")
            follow = f" I've opened ticket {s.ticket} so a colleague follows up." if s.ticket else " A colleague will follow up."
            return _final("I couldn't check your transactions just now." + follow)

        if not s.looked_up:
            if "lookup_transaction" in tools:
                args = {"reference": reference} if reference else ({"amount_inr": amount} if amount else {})
                return _act("lookup_transaction", args, "find the transaction before saying anything about it")
            return _final("I can't see transaction details from here. A colleague will check "
                          "and get back to you.")

        target = self._target(s, amount, reference)
        if target is None:
            if "open_ticket" in tools and not s.ticket and "open_ticket" not in s.refused:
                return _act("open_ticket", {
                    "category": "refund", "severity": 3,
                    "summary": "No matching transaction found; ask the customer for the reference privately",
                }, "nothing matches: record it rather than guess which debit they mean")
            what = f"a {_money(amount)} transaction" if amount else "that transaction"
            ticket = f" I've opened ticket {s.ticket} to follow up." if s.ticket else ""
            return _final(f"I couldn't find {what} on your account. Could you share the transaction "
                          f"reference from the app?{ticket}")

        ref, kind, status = target["reference"], target["kind"], target["status"]
        amt = _money(target["amount"])

        if kind == "failed payment" or status == "pending":
            return _final(f"Your {amt} payment is still pending. It hasn't failed, so there is "
                          f"nothing to reverse yet.")
        if kind == "payment":
            return _final(f"Your {amt} payment went through normally. If the order didn't arrive, "
                          f"the merchant handles that dispute, and I can share the reference for it.")

        if kind == "duplicate debit":
            if ref not in s.reversals and ref not in s.no_reversal and "reversal_status" in tools:
                return _act("reversal_status", {"reference": ref}, "has a reversal already started?")
            if ref in s.reversals:
                rev_id, rev_status = s.reversals[ref]
                if rev_status == "credited":
                    return _final(f"The reversal {rev_id} for the duplicate {amt} debit has completed.")
                return self._with_timeline(
                    ctx, s, tools, "duplicate debit",
                    f"A reversal ({rev_id}) for the duplicate {amt} debit is already in progress.",
                    "It is completed {timeline} of being reported.")
            return self._propose(s, ctx, target, "duplicate debit for one order")

        if kind == "failed transfer" and status == "failed not reversed":
            age = target["age"]
            if age is not None and age <= 3:
                return self._with_timeline(
                    ctx, s, tools, "failed transfer",
                    f"Your {amt} transfer failed at the receiving bank, so the money is on its way back.",
                    "A failed UPI transfer returns by itself {timeline}.")
            return self._propose(s, ctx, target, "failed transfer not returned after T+3")

        return _final(f"I can see the {amt} transaction. A colleague will look at it and get back to you.")

    # --- helpers ----------------------------------------------------------------

    @staticmethod
    def _target(s: Seen, amount: Optional[float], reference: Optional[str]) -> Optional[dict]:
        if reference and reference in s.txns:
            return s.txns[reference]
        problem = [t for t in s.txns.values()
                   if t["kind"] in ("duplicate debit", "failed transfer", "failed payment")]
        for pool in (problem, list(s.txns.values())):
            for t in pool:
                if amount is None or abs(t["amount"] - amount) < 0.01:
                    return t
        return None

    def _propose(self, s: Seen, ctx: LoopContext, target: dict, reason: str) -> tuple[Decision, dict]:
        amt = _money(target["amount"])
        if s.proposal_refused:
            if ctx.config.human_handoff and HANDOFF not in s.refused:
                return _act(HANDOFF, {"reason": "reversal refused by validation: " + s.proposal_refused[:120]},
                            "out of policy for me to fix: a person decides")
            return _final(f"I can't start a reversal for the {amt} transaction myself. "
                          f"A colleague will review it and get back to you.")
        if s.proposal_without_handoff:
            return _final(f"The {amt} transaction qualifies for review. A colleague has to approve "
                          f"any reversal and will pick it up.")
        return _act(PROPOSE_REVERSAL, {"reference": target["reference"], "amount_inr": target["amount"],
                                       "reason": reason},
                    "propose the fix; code validates it and a person approves")

    def _with_timeline(self, ctx: LoopContext, s: Seen, tools: set[str], kind: str,
                       base: str, sentence: str) -> tuple[Decision, dict]:
        clause_id, query, remembered = _POLICY_QUERY[kind]
        if clause_id in s.clauses:
            found = _CLAUSE_TIMELINE.search(s.clauses[clause_id])
            timeline = f"within {found.group(1)}" if found else None
            if timeline:
                return _final(f"{base} {sentence.format(timeline=timeline)}", [clause_id],
                              "answer with the timeline the policy states")
            return _final(f"{base} I'll confirm the exact timeline for you.")
        if s.timeline_rejected:
            if ctx.config.subagents and not s.consulted and CONSULT_POLICY not in s.refused:
                # Delegate in policy terms, not the customer's words. A vague question
                # retrieves the wrong clause and costs the loop an extra pass.
                return _act(CONSULT_POLICY, {"question": query},
                            "ask the policy sub-agent instead of pulling clauses into this context")
            if "search_policy" in tools and not s.searched:
                return _act("search_policy", {"query": query}, "find the clause that states the timeline")
            return _final(f"{base} I'll confirm the exact timeline for you.")
        # The eager first draft: the timeline it remembers, cited to nothing.
        return _final(f"{base} {sentence.format(timeline=remembered)}", [],
                      "I know how long this usually takes")

    def _policy_answer(self, ctx: LoopContext, s: Seen, tools: set[str]) -> tuple[Decision, dict]:
        if s.clauses:
            clause_id, snippet = next(iter(s.clauses.items()))
            text = snippet.replace("**", "").split(". ")[0].rstrip(".…") + "."
            return _final(text[:300], [clause_id], "answer from the retrieved clause")
        if ctx.config.subagents and not s.consulted and CONSULT_POLICY not in s.refused:
            return _act(CONSULT_POLICY, {"question": ctx.message}, "delegate the policy question")
        if "search_policy" in tools and not s.searched:
            return _act("search_policy", {"query": ctx.message[:300]}, "look up the policy")
        return _final("I'll check the policy and come back to you with the exact answer.")


# ---------------------------------------------------------------------------
# The model policy
# ---------------------------------------------------------------------------

_SYSTEM = """You are NimbusPay's support agent, working inside a loop. On each pass
you choose exactly ONE action.

Actions (arguments go in args_json, as a JSON object):
- lookup_transaction {"reference"?: "NP-TXN-...", "amount_inr"?: number}
- reversal_status {"reference": "NP-TXN-..."}
- search_policy {"query": "specific policy terms, not the customer's words"}
- open_ticket {"category": str, "severity": 1-5, "summary": str}
- consult_policy {"question": str}  — a policy specialist answers in its own context
- propose_reversal {"reference": str, "amount_inr": number, "reason": str}  — code validates, a person approves
- handoff {"reason": str}
- final — reply to the customer: answer_text (under 60 words, no clause ids) and
  cited_clauses (ids of clauses you retrieved)

Only the actions listed as available will work.

Rules:
- Never state an amount, reference or timeline you have not observed. A timeline
  needs a retrieved clause in cited_clauses.
- If a tool fails, do not guess: open a ticket or hand off.
- Never promise money. Propose a reversal; a person decides.
- Stop as soon as you can answer. Every pass costs time and money."""


class ModelDecision(BaseModel):
    thought: str = ""
    action: str
    args_json: str = "{}"
    answer_text: str = ""
    cited_clauses: list[str] = Field(default_factory=list)


class ModelPolicy:
    name = "gemini"

    def __init__(self, model: str = TIER_DRAFT, fallback: str = TIER_REASONING):
        self.model, self.fallback = model, fallback

    async def decide(self, view: str, ctx: LoopContext) -> tuple[Decision, dict]:
        cfg = ctx.config
        available = list(cfg.tools) + [
            name for name, on in ((CONSULT_POLICY, cfg.subagents), (PROPOSE_REVERSAL, cfg.human_handoff),
                                  (HANDOFF, cfg.human_handoff)) if on] + [FINAL]
        raw, cost = await structured(
            model=self.model, fallback_model=self.fallback,
            system=with_trust_rules(_SYSTEM + "\n\nAvailable actions now: " + ", ".join(available)),
            user="Everything so far:\n" + untrusted("loop_context", view),
            schema=ModelDecision, temperature=0.0, stage="loop",
            max_output_tokens=300, timeout_s=15.0, trace_id=ctx.run_id,
        )
        try:
            args = json.loads(raw.args_json or "{}")
            args = args if isinstance(args, dict) else {}
        except ValueError:
            args = {}
        answer = FinalAnswer(text=raw.answer_text, cited_clauses=raw.cited_clauses) if raw.action == FINAL else None
        usage = {"prompt_tokens": cost.get("prompt_tokens"), "output_tokens": cost.get("output_tokens"),
                 "cost_inr": cost.get("inr")}
        return Decision(thought=raw.thought, action=raw.action, args=args, answer=answer), usage


def default_policy():
    """Real model when a key is configured, the scripted stand-in otherwise."""
    return ScriptedSupportPolicy() if is_offline() else ModelPolicy()
