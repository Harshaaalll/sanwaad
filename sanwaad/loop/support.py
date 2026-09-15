"""The running example: a customer support conversation, built as a loop.

A customer writes in privately — "where's my refund for the ₹640 double
debit?" — and the agent works it out: classify, look up the transaction, check
whether a reversal already started, check the policy, then answer, open a
ticket, or hand a validated reversal to a person. Each of those is one pass
through the same loop.

This module is where the MINT layers plug in:

- WORKFLOWS (M4). Each intent gets its own flow: a narrower tool set and a pass
  budget sized to the job. A policy question needs one tool and three passes,
  not four tools and eight. A narrower action space is also a smaller surface
  for the agent to wander into.
- CONVERSATION MEMORY (M3). What was *decided* in earlier turns — a ticket
  opened, a hand-over made — is recalled at the start of the next turn, so a
  returning customer isn't asked to start again and doesn't get a second
  ticket. Live facts (a reversal's status, a balance) are deliberately NOT
  remembered: they go stale, and the ledger is always one tool call away.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ..config import DATA_DIR
from ..guardrails import redact
from .kernel import LoopConfig, LoopPolicy, LoopRun, run_loop
from .mint import check_layering, config_for
from .policy import classify_intent, default_policy

CONVERSATIONS_PATH = DATA_DIR / "conversations.json"


@dataclass(frozen=True)
class Workflow:
    name: str
    tools: tuple[str, ...]
    max_iterations: int
    purpose: str


WORKFLOWS: dict[str, Workflow] = {
    "refund_status": Workflow(
        "refund_status", ("lookup_transaction", "reversal_status", "search_policy", "open_ticket"), 7,
        "find the debit, check any reversal, state only the timeline policy supports"),
    "refund_request": Workflow(
        "refund_request", ("lookup_transaction", "search_policy", "open_ticket"), 6,
        "find the debit, propose a reversal for a person to approve"),
    "payment_status": Workflow(
        "payment_status", ("lookup_transaction", "open_ticket"), 4,
        "find the payment and say where it stands"),
    "policy_question": Workflow(
        "policy_question", ("search_policy",), 3,
        "answer from a retrieved clause"),
    "manipulation": Workflow("manipulation", (), 2, "decline, without tools"),
    "other": Workflow("other", (), 2, "ask what they need, without tools"),
}


def workflow_for(message: str) -> Workflow:
    return WORKFLOWS.get(classify_intent(message), WORKFLOWS["other"])


# ---------------------------------------------------------------------------
# Memory across turns
# ---------------------------------------------------------------------------

_TICKET = re.compile(r"\bTKT-[A-F0-9]{8}\b")


class ConversationMemory:
    """Per-customer decisions from earlier turns. Bounded, dated, redacted."""

    def __init__(self, path: Optional[Path] = None, max_turns: int = 5, retention_days: int = 30):
        self.path = path
        self.max_turns = max_turns
        self.retention_days = retention_days

    def _file(self) -> Path:
        return self.path or CONVERSATIONS_PATH

    def _load(self) -> dict:
        f = self._file()
        if not f.exists():
            return {}
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def recall(self, handle: str, now: Optional[datetime] = None) -> list[str]:
        now = now or datetime.now(timezone.utc)
        floor = now - timedelta(days=self.retention_days)
        lines = []
        for turn in self._load().get(handle, [])[-self.max_turns:]:
            try:
                if datetime.fromisoformat(turn["at"]) < floor:
                    continue
            except (KeyError, ValueError):
                continue
            line = f"earlier ({turn['at'][:10]}, {turn['intent']}): {turn['outcome']}"
            if turn.get("ticket"):
                line += f"; ticket {turn['ticket']} is open"
            lines.append(line)
        return lines

    def remember(self, handle: str, run: LoopRun, intent: str) -> None:
        data = self._load()
        text = (run.answer.text if run.answer else "") + " " + json.dumps(run.steps, default=str)
        ticket = _TICKET.search(text)
        outcome = {
            "done": "answered",
            "needs_human": f"handed to a person ({(run.handoff or {}).get('reason', '')[:80]})",
        }.get(run.stop_reason.value, f"loop stopped: {run.stop_reason.value}")
        data.setdefault(handle, []).append({
            "at": datetime.now(timezone.utc).isoformat(),
            "intent": intent,
            "outcome": redact(outcome)[0],
            "ticket": ticket.group(0) if ticket else None,
        })
        data[handle] = data[handle][-self.max_turns:]
        f = self._file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# One turn of the conversation
# ---------------------------------------------------------------------------

async def handle_turn(message: str, *, handle: str, rung: int = 5,
                      config: Optional[LoopConfig] = None,
                      policy: Optional[LoopPolicy] = None,
                      memory: Optional[ConversationMemory] = None,
                      variant: Optional[str] = None) -> LoopRun:
    """Run one customer message through the support loop at a MINT rung."""
    config = config or config_for(rung)
    check_layering(config)
    intent = classify_intent(message)

    workflow = None
    if config.workflows:
        flow = WORKFLOWS.get(intent, WORKFLOWS["other"])
        config = replace(config, tools=tuple(t for t in config.tools if t in flow.tools),
                         budget=replace(config.budget, max_iterations=flow.max_iterations))
        workflow = flow.name

    memory = memory or ConversationMemory()
    notes = memory.recall(handle) if config.memory else []

    run = await run_loop(message, handle=handle, policy=policy or default_policy(), config=config,
                         workflow=workflow, variant=variant, memory_notes=notes)
    if config.memory:
        memory.remember(handle, run, intent)
    return run
