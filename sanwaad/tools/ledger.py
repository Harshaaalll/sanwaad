"""Mock systems of record: the payments ledger and the ticket desk.

These stand in for backends Sanwaad does not own. They are deliberately NOT
agent memory. Whether a transaction exists, who owns it, and whether it has
already been reversed are business facts; they live in the application's own
store and are reached through tools, and no agent keeps its own copy of them.
An agent that remembers "I reversed that already" instead of asking the ledger
is how the same ₹640 gets returned twice.

State persists to a JSON file so a reversal survives a restart — a case can be
approved in the console hours after it was planned, in another process.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from ..config import DATA_DIR
from .contracts import ErrorCode, ToolFailure

BACKEND_PATH = DATA_DIR / "mock_backend.json"


class Transaction(BaseModel):
    reference: str
    account_id: str          # never leaves the backend; tools expose a view
    handle: str              # the public handle linked to the account
    handle_verified: bool    # was that link verified, or merely claimed?
    amount_inr: float
    merchant: str
    kind: Literal["payment", "duplicate_debit", "failed_transfer", "failed_payment"]
    status: Literal["settled", "failed_not_reversed", "pending", "reversed"]
    age_days: float
    duplicate_of: Optional[str] = None


# Each row exists to exercise one decision the validator has to make.
_SEED: list[dict] = [
    # Karthik's Swiggy double debit: A is the real payment, B the duplicate.
    dict(reference="NP-TXN-640-A", account_id="acct_4417", handle="u/karthik_rn",
         handle_verified=True, amount_inr=640, merchant="Swiggy", kind="payment",
         status="settled", age_days=0.5),
    dict(reference="NP-TXN-640-B", account_id="acct_4417", handle="u/karthik_rn",
         handle_verified=True, amount_inr=640, merchant="Swiggy", kind="duplicate_debit",
         status="settled", age_days=0.5, duplicate_of="NP-TXN-640-A"),
    # Rohit: failed transfer, six days old — past the T+3 window (RFD-02).
    dict(reference="NP-TXN-4500", account_id="acct_2291", handle="u/rohit_mhrs",
         handle_verified=True, amount_inr=4500, merchant="bank transfer",
         kind="failed_transfer", status="failed_not_reversed", age_days=6),
    # Asha: failed two days ago — INSIDE T+3, reverses by itself (RFD-01).
    dict(reference="NP-TXN-2000", account_id="acct_7310", handle="u/asha_v",
         handle_verified=True, amount_inr=2000, merchant="bank transfer",
         kind="failed_transfer", status="failed_not_reversed", age_days=2),
    # An outage payment still pending: nothing has failed yet.
    dict(reference="NP-TXN-1200", account_id="acct_5102", handle="u/nk_bhatia",
         handle_verified=True, amount_inr=1200, merchant="kirana store",
         kind="failed_payment", status="pending", age_days=0.01),
    # Eligible, but above the agent's reversal ceiling.
    dict(reference="NP-TXN-32000", account_id="acct_6620", handle="u/big_ticket",
         handle_verified=True, amount_inr=32000, merchant="bank transfer",
         kind="failed_transfer", status="failed_not_reversed", age_days=5),
    # The handle was claimed in chat but never verified against the account.
    dict(reference="NP-TXN-900", account_id="acct_8841", handle="u/claimed_only",
         handle_verified=False, amount_inr=900, merchant="bank transfer",
         kind="failed_transfer", status="failed_not_reversed", age_days=5),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}-{h}"


class MockBackend:
    def __init__(self, path: Optional[Path] = None):
        self.path = path

    # --- persistence ----------------------------------------------------------

    def _file(self) -> Path:
        return self.path or BACKEND_PATH

    def _load(self) -> dict:
        f = self._file()
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"status": {}, "reversals": {}, "tickets": {}}

    def _save(self, data: dict) -> None:
        f = self._file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self._file().unlink(missing_ok=True)

    # --- reads ------------------------------------------------------------------

    def transactions(self) -> list[Transaction]:
        overrides = self._load()["status"]
        out = []
        for row in _SEED:
            t = Transaction(**row)
            if t.reference in overrides:
                t.status = overrides[t.reference]
            out.append(t)
        return out

    def get(self, reference: str) -> Optional[Transaction]:
        return next((t for t in self.transactions() if t.reference == reference), None)

    def find(self, *, handle: str, reference: Optional[str] = None,
             amount_inr: Optional[float] = None) -> list[Transaction]:
        """Scoped to one handle, always.

        A lookup by reference alone would let anyone who has seen a reference
        number learn the amount and merchant behind it. The scope key comes
        from the system (the complaint's author), never from the model.
        """
        rows = [t for t in self.transactions() if t.handle == handle]
        if reference:
            rows = [t for t in rows if t.reference == reference]
        if amount_inr is not None:
            rows = [t for t in rows if abs(t.amount_inr - amount_inr) < 0.01]
        return rows

    def reversal_for(self, reference: str) -> Optional[dict]:
        return next((r for r in self._load()["reversals"].values()
                     if r["reference"] == reference), None)

    def ticket(self, idempotency_key: str) -> Optional[dict]:
        return self._load()["tickets"].get(idempotency_key)

    # --- writes -----------------------------------------------------------------

    def reverse(self, *, reference: str, amount_inr: float, case_id: str,
                idempotency_key: str) -> tuple[dict, bool]:
        """Returns (record, duplicate). Same key twice returns the first result."""
        data = self._load()
        if idempotency_key in data["reversals"]:
            return data["reversals"][idempotency_key], True
        if any(r["reference"] == reference for r in data["reversals"].values()):
            raise ToolFailure(ErrorCode.CONFLICT,
                              f"{reference} was already reversed under another request")
        if self.get(reference) is None:
            raise ToolFailure(ErrorCode.NOT_FOUND, f"no transaction {reference}")
        record = {
            "reversal_id": _short_id("REV", reference, idempotency_key),
            "reference": reference, "amount_inr": amount_inr,
            "case_id": case_id, "at": _now(),
        }
        data["reversals"][idempotency_key] = record
        data["status"][reference] = "reversed"
        self._save(data)
        return record, False

    def open_ticket(self, *, case_id: str, category: str, severity: int, summary: str,
                    idempotency_key: str) -> tuple[dict, bool]:
        data = self._load()
        if idempotency_key in data["tickets"]:
            return data["tickets"][idempotency_key], True
        record = {
            "ticket_id": _short_id("TKT", idempotency_key),
            "case_id": case_id, "category": category, "severity": severity,
            "summary": summary, "at": _now(),
        }
        data["tickets"][idempotency_key] = record
        self._save(data)
        return record, False


BACKEND = MockBackend()
