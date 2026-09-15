"""The state carried through a grievance, from public comment to closure.

One state object spans both channels. That is the whole point: when the voice
agent picks the case up it inherits the citations the public reply was written
from, so it cannot answer from a different set of facts.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class GrievanceState(TypedDict, total=False):
    case_id: str

    # Inbound
    complaint: dict          # Complaint.model_dump()

    # Stage outputs
    triage: Optional[dict]           # Triage
    pattern: Optional[dict]          # PatternSignal — how this looks beside the others
    coordination: int                # other accounts posting near-identical wording
    verdict: Optional[dict]          # AuthorVerdict — who is speaking
    priority: Optional[dict]         # Priority — the three readings folded together
    retrieval_query: str             # English summary actually used for search
    citations: list[dict]            # list[Citation]
    draft: Optional[dict]            # Draft
    guardrails: list[dict]           # deterministic violations on the draft
    injection_flagged: bool
    grounding: Optional[dict]        # GroundingVerdict
    revision_count: int
    actions: list[dict]              # [{proposal: ProposedAction, validation: Validation}]
    review: Optional[dict]           # Review
    action_results: list[dict]       # what the executor did with each proposal
    published: Optional[dict]
    escalation: Optional[dict]
    voice: Optional[dict]            # VoiceOutcome
    closure: Optional[dict]

    # Accumulated across nodes
    costs: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]


def event(stage: str, message: str, **extra: Any) -> dict:
    """One audit-trail entry. The console renders these as the case timeline."""
    from datetime import datetime, timezone

    return {
        "stage": stage,
        "message": message,
        "at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
