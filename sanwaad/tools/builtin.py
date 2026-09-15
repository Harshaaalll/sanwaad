"""The four tools Sanwaad's agents may call, one per rung of the risk ladder.

    lookup_transaction   READ        plan
    open_ticket          WRITE_LOW   act       idempotent, no approval
    post_reply           WRITE_HIGH  publish   approvable by the auto-post policy
    initiate_reversal    WRITE_HIGH  act       approvable only by a human

Two contract details worth copying:

- No tool accepts a free-form instruction. There is no `update_case(request:
  str)`; there is `initiate_reversal(reference, amount_inr, reason, case_id,
  idempotency_key)`, every field bounded. A model cannot ask a backend to do
  something the schema has no field for.
- Outputs are minimal views. `lookup_transaction` returns amount, merchant,
  kind and status — never the account id. The validator reads the full row
  from the backend directly, in code, where no prompt can see it.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..connectors import get_connector
from .contracts import ErrorCode, Risk, ToolFailure, ToolSpec
from .ledger import BACKEND
from .registry import REGISTRY

_REFERENCE = r"^NP-TXN-[A-Z0-9-]{1,20}$"
_CASE_ID = r"^case_[a-z0-9]{6,32}$"


# ---------------------------------------------------------------------------
# lookup_transaction — READ
# ---------------------------------------------------------------------------

class TransactionView(BaseModel):
    reference: str
    amount_inr: float
    merchant: str
    kind: str
    status: str
    age_days: float
    duplicate_of: Optional[str] = None


class LookupTransactionIn(BaseModel):
    handle: str = Field(min_length=2, max_length=64,
                        description="The complaint author's handle. Supplied by the system.")
    reference: Optional[str] = Field(default=None, pattern=_REFERENCE)
    amount_inr: Optional[float] = Field(default=None, gt=0, le=1_000_000)


class LookupTransactionOut(BaseModel):
    matches: list[TransactionView]


async def _lookup_transaction(args: LookupTransactionIn) -> LookupTransactionOut:
    rows = BACKEND.find(handle=args.handle, reference=args.reference,
                        amount_inr=args.amount_inr)
    return LookupTransactionOut(matches=[
        TransactionView(**r.model_dump(exclude={"account_id", "handle", "handle_verified"}))
        for r in rows
    ])


# ---------------------------------------------------------------------------
# open_ticket — WRITE_LOW
# ---------------------------------------------------------------------------

class OpenTicketIn(BaseModel):
    case_id: str = Field(pattern=_CASE_ID)
    category: str = Field(min_length=2, max_length=32)
    severity: int = Field(ge=1, le=5)
    summary: str = Field(min_length=3, max_length=280)
    idempotency_key: str = Field(min_length=6, max_length=64)


class OpenTicketOut(BaseModel):
    ticket_id: str
    duplicate: bool


async def _open_ticket(args: OpenTicketIn) -> OpenTicketOut:
    record, duplicate = BACKEND.open_ticket(
        case_id=args.case_id, category=args.category, severity=args.severity,
        summary=args.summary, idempotency_key=args.idempotency_key)
    return OpenTicketOut(ticket_id=record["ticket_id"], duplicate=duplicate)


# ---------------------------------------------------------------------------
# post_reply — WRITE_HIGH, auto-approvable
# ---------------------------------------------------------------------------

class PostReplyIn(BaseModel):
    channel: Literal["mock", "playstore", "reddit"]
    external_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1200)


class PostReplyOut(BaseModel):
    reply_id: Optional[str] = None
    url: Optional[str] = None
    posted_at: str
    dry_run: bool


async def _post_reply(args: PostReplyIn) -> PostReplyOut:
    receipt = await get_connector(args.channel).reply(args.external_id, args.text)
    return PostReplyOut(**receipt)


# ---------------------------------------------------------------------------
# initiate_reversal — WRITE_HIGH, human only
# ---------------------------------------------------------------------------

class InitiateReversalIn(BaseModel):
    reference: str = Field(pattern=_REFERENCE)
    amount_inr: float = Field(gt=0, le=1_000_000)
    reason: str = Field(min_length=3, max_length=200)
    case_id: str = Field(pattern=_CASE_ID)
    idempotency_key: str = Field(min_length=6, max_length=64)

    @model_validator(mode="after")
    def _whole_paise(self):
        if round(self.amount_inr, 2) != self.amount_inr:
            raise ValueError("amount must be in whole paise")
        return self


class InitiateReversalOut(BaseModel):
    reversal_id: str
    status: Literal["initiated"]
    duplicate: bool
    eta_hours: int


async def _initiate_reversal(args: InitiateReversalIn) -> InitiateReversalOut:
    txn = BACKEND.get(args.reference)
    if txn is None:
        raise ToolFailure(ErrorCode.NOT_FOUND, f"no transaction {args.reference}")
    if abs(txn.amount_inr - args.amount_inr) >= 0.01:
        # The backend checks too. Validation upstream is not a reason for the
        # system of record to accept an amount it does not hold.
        raise ToolFailure(ErrorCode.INVALID_INPUT, "amount does not match the transaction")
    record, duplicate = BACKEND.reverse(
        reference=args.reference, amount_inr=args.amount_inr,
        case_id=args.case_id, idempotency_key=args.idempotency_key)
    return InitiateReversalOut(reversal_id=record["reversal_id"], status="initiated",
                               duplicate=duplicate, eta_hours=24)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

REGISTRY.register(ToolSpec(
    name="lookup_transaction",
    description="Find the complaint author's own transactions by reference and/or amount.",
    input_model=LookupTransactionIn, output_model=LookupTransactionOut,
    handler=_lookup_transaction, risk=Risk.READ,
    timeout_s=3.0, max_retries=2, idempotent=True,
))

REGISTRY.register(ToolSpec(
    name="open_ticket",
    description="Open (or return the existing) internal ticket for a case.",
    input_model=OpenTicketIn, output_model=OpenTicketOut,
    handler=_open_ticket, risk=Risk.WRITE_LOW,
    timeout_s=5.0, max_retries=2, idempotent=True,
))

REGISTRY.register(ToolSpec(
    name="post_reply",
    description="Post an approved public reply on the channel the complaint came from.",
    input_model=PostReplyIn, output_model=PostReplyOut,
    handler=_post_reply, risk=Risk.WRITE_HIGH,
    timeout_s=10.0, max_retries=0, idempotent=False, auto_approvable=True,
))

REGISTRY.register(ToolSpec(
    name="initiate_reversal",
    description="Reverse one eligible debit. Requires a human approval of these exact arguments.",
    input_model=InitiateReversalIn, output_model=InitiateReversalOut,
    handler=_initiate_reversal, risk=Risk.WRITE_HIGH,
    timeout_s=10.0, max_retries=1, idempotent=True, auto_approvable=False,
))
