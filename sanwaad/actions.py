"""Approvals: the model suggests, code validates, a human approves, a tool executes.

Moving money is the one thing in Sanwaad that must never happen because a model
inferred it should. So the path from "this customer was double-debited" to a
reversal is split into four hands, and no hand trusts the one before it:

    plan        (model)  proposes a reversal: reference, amount, reason
    validate    (code)   checks it against the system of record — exists? owned
                         by this author? verified? amount right? eligible under
                         policy? under the ceiling? not already reversed?
    approve     (human)  sees the proposal AND every check, approves the exact
                         arguments; that approval is bound to them by digest
    execute     (tool)   the executor re-validates — the ledger may have changed
                         since planning — and the registry refuses any
                         high-risk call whose approval does not match

The validator is the source of truth for business rules. Not the prompt, not
the planner, not the reviewer's memory. That is why it takes the author from
the complaint (the system) and the transaction from the ledger (the system),
and takes nothing but the proposal itself from the model.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from .config import ACTIONS
from .tools.ledger import BACKEND, MockBackend, Transaction

ActionKind = Literal["reversal", "ticket"]

_TOOL_FOR = {"reversal": "initiate_reversal", "ticket": "open_ticket"}
_RISK_FOR = {"reversal": "write_high", "ticket": "write_low"}

_REFERENCE_RE = re.compile(r"\bNP-TXN-[A-Z0-9-]{1,20}\b")


class ProposedAction(BaseModel):
    id: str
    kind: ActionKind
    tool: str
    risk: str
    case_id: str
    reason: str
    reference: Optional[str] = None
    amount_inr: Optional[float] = None
    clause_id: Optional[str] = None
    proposed_by: str = "plan"


class Check(BaseModel):
    name: str
    passed: bool
    detail: str


class Validation(BaseModel):
    ok: bool
    checks: list[Check] = Field(default_factory=list)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def make_action(kind: ActionKind, *, case_id: str, reason: str,
                reference: Optional[str] = None, amount_inr: Optional[float] = None,
                proposed_by: str = "plan") -> ProposedAction:
    """A stable id, so the same proposal made twice is recognisably the same."""
    digest = hashlib.sha256(f"{kind}|{case_id}|{reference or ''}".encode()).hexdigest()[:8]
    return ProposedAction(
        id=f"act_{digest}", kind=kind, tool=_TOOL_FOR[kind], risk=_RISK_FOR[kind],
        case_id=case_id, reason=reason[:200], reference=reference,
        amount_inr=amount_inr, proposed_by=proposed_by,
    )


def references_in(text: str) -> list[str]:
    return sorted(set(_REFERENCE_RE.findall(text or "")))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def reversal_eligibility(txn: Transaction) -> tuple[bool, Optional[str], str]:
    """Is this debit reversible under written policy, and which clause says so?

    Encoded as code because the rules are enumerable. A model asked "is this
    reversible?" would be right most of the time; this is right every time,
    and when it is wrong the fix is a one-line diff someone can review.
    """
    if txn.status == "reversed":
        return False, None, "already reversed"
    if txn.kind == "duplicate_debit" and txn.status == "settled":
        return True, "RFD-06", "duplicate debit — reversed under the duplicate-debit rule"
    if txn.kind == "failed_transfer" and txn.status == "failed_not_reversed":
        if txn.age_days <= 3:
            return False, "RFD-01", (f"failed {txn.age_days:g} day(s) ago — still inside the "
                                     "T+3 auto-reversal window, it returns on its own")
        return True, "RFD-02", f"failed {txn.age_days:g} days ago — past T+3, reversal is owed"
    if txn.kind == "failed_payment" and txn.status == "pending":
        return False, "RFD-01", "still pending — nothing has failed yet"
    if txn.kind == "payment":
        return False, None, "an ordinary settled payment — nothing to reverse"
    return False, None, f"{txn.kind}/{txn.status} is not a reversible state"


def _default_clause_exists(clause_id: str) -> bool:
    from .rag.store import get_store

    return get_store().get(clause_id) is not None


def validate_action(
    action: ProposedAction,
    *,
    author: str,
    backend: Optional[MockBackend] = None,
    clause_exists: Callable[[str], bool] = _default_clause_exists,
) -> Validation:
    """Every rule, every time, each one named — including the ones that pass.

    All checks run even after one fails. A reviewer looking at a blocked
    proposal needs the whole picture ("wrong owner AND above ceiling"), not
    whichever failure happened to be tested first.
    """
    backend = backend or BACKEND
    checks: list[Check] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name=name, passed=bool(passed), detail=detail))

    if action.kind == "ticket":
        check("required_fields", bool(action.case_id and action.reason),
              "case id and a reason are present" if action.case_id and action.reason
              else "a ticket needs a case id and a reason")
        return Validation(ok=all(c.passed for c in checks), checks=checks)

    # --- reversal ------------------------------------------------------------
    has_fields = bool(action.reference and action.amount_inr)
    check("required_fields", has_fields,
          "reference and amount are present" if has_fields
          else "a reversal needs a transaction reference and an amount")

    txn = backend.get(action.reference) if action.reference else None
    check("transaction_exists", txn is not None,
          f"{action.reference} found in the ledger" if txn
          else f"no transaction {action.reference or '(none given)'}")

    if txn is None:
        for name in ("ownership", "identity_verified", "amount_matches",
                     "eligible_under_policy", "within_ceiling", "not_already_reversed"):
            check(name, False, "cannot check without a transaction")
        return Validation(ok=False, checks=checks)

    # Ownership is checked against the ledger, with the author taken from the
    # complaint — never from anything the model wrote.
    check("ownership", txn.handle == author,
          "transaction belongs to the complaint's author" if txn.handle == author
          else "transaction belongs to a different account")
    check("identity_verified", txn.handle_verified,
          "handle is verified against the account" if txn.handle_verified
          else "handle was claimed, never verified — confirm identity first")

    amount_ok = action.amount_inr is not None and abs(txn.amount_inr - action.amount_inr) < 0.01
    check("amount_matches", amount_ok,
          f"₹{txn.amount_inr:,.0f} matches the ledger" if amount_ok
          else f"proposed ₹{action.amount_inr or 0:,.0f}, ledger says ₹{txn.amount_inr:,.0f}")

    eligible, clause, why = reversal_eligibility(txn)
    check("eligible_under_policy", eligible, f"{why}{f' ({clause})' if clause else ''}")
    if clause:
        action.clause_id = clause

    ceiling = ACTIONS.reversal_ceiling_inr
    check("within_ceiling", txn.amount_inr <= ceiling,
          f"₹{txn.amount_inr:,.0f} is within the ₹{ceiling:,.0f} agent ceiling"
          if txn.amount_inr <= ceiling
          else f"₹{txn.amount_inr:,.0f} exceeds the ₹{ceiling:,.0f} ceiling — nodal officer (ESC-04)")

    prior = backend.reversal_for(txn.reference)
    check("not_already_reversed", prior is None,
          "no earlier reversal" if prior is None
          else f"already reversed as {prior['reversal_id']}")

    if eligible and clause:
        check("clause_exists", clause_exists(clause),
              f"{clause} is in the policy index" if clause_exists(clause)
              else f"{clause} is missing from the policy index")

    return Validation(ok=all(c.passed for c in checks), checks=checks)


# ---------------------------------------------------------------------------
# Tool arguments
# ---------------------------------------------------------------------------

def tool_args(action: ProposedAction, *, triage: dict) -> dict:
    """Build the exact arguments the executor will send.

    The idempotency key is derived from the case and the reference, so the
    same approved action executed twice — a retry, a double click, a resumed
    graph — is one reversal, not two.
    """
    key = f"{action.case_id}:{action.reference or action.kind}"
    if action.kind == "reversal":
        return {
            "reference": action.reference,
            "amount_inr": action.amount_inr,
            "reason": action.reason,
            "case_id": action.case_id,
            "idempotency_key": key,
        }
    return {
        "case_id": action.case_id,
        "category": triage.get("category", "unknown"),
        "severity": int(triage.get("severity", 3)),
        "summary": action.reason,
        "idempotency_key": key,
    }
