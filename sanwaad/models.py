"""Domain types shared by the graph, the connectors and the API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _case_id() -> str:
    return f"case_{uuid.uuid4().hex[:10]}"


class Channel(str, Enum):
    REDDIT = "reddit"
    MOCK = "mock"
    PLAYSTORE = "playstore"
    VOICE = "voice"


class Category(str, Enum):
    BILLING = "billing"
    REFUND = "refund"
    SERVICE_OUTAGE = "service_outage"
    DELIVERY = "delivery"
    ACCOUNT_ACCESS = "account_access"
    AGENT_BEHAVIOUR = "agent_behaviour"
    DATA_PRIVACY = "data_privacy"
    PRAISE = "praise"
    OFF_TOPIC = "off_topic"


class AuthorMeta(BaseModel):
    """What the platform knows about whoever posted this.

    Every field is optional because every platform exposes a different subset.
    The judge degrades gracefully: a missing signal is not a negative one, it
    just carries no weight.
    """

    account_age_days: Optional[int] = None
    karma: Optional[int] = None
    followers: Optional[int] = None
    verified: bool = False
    post_count: Optional[int] = None


class Complaint(BaseModel):
    """One inbound item, normalised across connectors."""

    external_id: str
    channel: Channel
    author: str
    text: str
    url: Optional[str] = None
    created_at: str = Field(default_factory=_now)
    parent_text: Optional[str] = None  # thread context, when the connector has it

    # Set by the listener when the brand was never @-mentioned. An untagged
    # complaint is the norm, not the exception, and it is the one nobody sees.
    tagged: bool = True
    author_meta: AuthorMeta = Field(default_factory=AuthorMeta)


class Triage(BaseModel):
    is_complaint: bool
    category: Category
    severity: int = Field(ge=1, le=5, description="1 mild annoyance, 5 regulatory or safety risk")
    sentiment: Literal["angry", "frustrated", "neutral", "positive"]
    language: str = "en"
    summary: str
    entities: dict[str, Any] = Field(default_factory=dict)
    needs_private_data: bool = False


class AuthorVerdict(BaseModel):
    """Who is saying this — the signal that decides how much a complaint costs us.

    The same words carry different weight from a two-year-old account that has
    bought from us twice and from an hour-old account posting the same line
    under every launch. Both still get read; only one gets a drafted reply and
    a callback.
    """

    author_class: Literal["customer", "audience", "troll", "bot", "competitor", "unknown"]
    authenticity: float = Field(ge=0.0, le=1.0,
                                description="0 manufactured outrage, 1 a real person with a real problem")
    reach: int = Field(default=0, ge=0, description="audience-size proxy: followers, or karma when that is all we have")
    history_with_brand: int = Field(default=0, ge=0, description="prior cases from this author")
    evidence: list[str] = Field(default_factory=list)
    reply_worthy: bool = True


class PatternSignal(BaseModel):
    """What this complaint looks like *next to the others*.

    One person saying the app is down is a support ticket. Nine people saying
    it within twenty minutes is an outage, and the difference is not visible
    in any single comment — which is why triage cannot see it and this can.
    """

    level: Literal["none", "watch", "crisis"] = "none"
    cluster_size: int = 1
    window_minutes: int = 0
    velocity_per_hour: float = 0.0
    theme: str = ""
    related_case_ids: list[str] = Field(default_factory=list)


class Priority(BaseModel):
    """Triage severity, author weight and pattern level folded into one call."""

    tier: Literal["ignore", "routine", "priority", "crisis"]
    score: float
    reasons: list[str] = Field(default_factory=list)
    drafting: bool = True   # spend a drafting call at all?


class Citation(BaseModel):
    """A retrieved policy clause. The clause_id is what makes the two channels
    provably consistent — the same id must appear behind the public reply and
    behind whatever the voice agent says."""

    clause_id: str
    doc: str
    heading: str
    text: str
    score: float


class Draft(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    promises_compensation: bool = False


class GroundingVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""
    unavailable: bool = False   # the checker could not run: nothing was verified


class Review(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    final_text: Optional[str] = None
    reviewer: str = "human"
    note: str = ""
    decided_at: str = Field(default_factory=_now)
    auto: bool = False
    # action_id -> approve | reject. Separate from the reply decision: a
    # reviewer can post the reply and still refuse the refund, or the reverse.
    actions: dict[str, Literal["approve", "reject"]] = Field(default_factory=dict)


class VoiceOutcome(BaseModel):
    happened: bool = False
    channel: Literal["webrtc", "exotel", "none"] = "none"
    duration_s: float = 0.0
    resolved: bool = False
    summary: str = ""
    citations_used: list[str] = Field(default_factory=list)
    transcript_path: Optional[str] = None


class CostEntry(BaseModel):
    stage: str
    model: str = ""
    usd: float = 0.0
    inr: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


class ConsistencyReceipt(BaseModel):
    """The artefact that makes the product's claim auditable.

    Both channels answered from `shared_clauses`. `text_only` / `voice_only`
    being non-empty means the channels diverged — which is exactly the bug
    this system exists to catch.
    """

    shared_clauses: list[str]
    text_only: list[str]
    voice_only: list[str]
    consistent: bool
