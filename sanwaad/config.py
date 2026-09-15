"""Central configuration: model tiers, pricing, paths.

Model tiering is the core cost lever. Triage runs on every inbound comment,
so it uses the cheapest model available; drafting runs only on real
complaints; the expensive tier is reserved for high-severity escalations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
POLICY_DIR = PACKAGE_ROOT / "policy"
DATA_DIR = PACKAGE_ROOT / "data"
INDEX_PATH = DATA_DIR / "policy_index.json"
CHECKPOINT_PATH = DATA_DIR / "sanwaad.sqlite"
CASES_PATH = DATA_DIR / "cases.json"


# --- Model tiers -----------------------------------------------------------
# Every inbound item hits TRIAGE. Only genuine complaints reach DRAFT. Only
# severity>=4 reaches REASONING. This ordering is what keeps cost per
# resolution in paise rather than rupees.

TIER_TRIAGE = os.getenv("SANWAAD_MODEL_TRIAGE", "gemini-2.5-flash-lite")
TIER_DRAFT = os.getenv("SANWAAD_MODEL_DRAFT", "gemini-2.5-flash")
# The reasoning tier must be a genuinely stronger model than TIER_DRAFT, or the
# router's escalation branches are decorative. Reached only on a grounding
# failure, an adversarial input, or severity >= 4 — a few percent of traffic.
TIER_REASONING = os.getenv("SANWAAD_MODEL_REASONING", "gemini-2.5-pro")

# Embeddings run locally on ONNX — no per-token cost, no network round trip.
# The multilingual model matters: complaints arrive in Hinglish and Devanagari.
EMBED_MODEL = os.getenv(
    "SANWAAD_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --- Pricing (USD per 1M tokens) -------------------------------------------
# Shared with the voice leg's metrics so text and voice costs roll up into
# one comparable number.

LLM_PRICING_USD = {
    "gemini-2.5-flash": {"input": 0.30, "cache_read": 0.03, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "cache_read": 0.01, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "cache_read": 0.13, "output": 10.00},
    "_default": {"input": 0.30, "cache_read": 0.03, "output": 2.50},
}

# What a human agent costs for the same work, for the comparison the console
# renders. Tune to your market; defaults are Indian BPO ballpark.
HUMAN_COST_INR = {
    "social_reply": 18.0,   # ~3 min of an agent's time at ~₹350/hr
    "voice_call": 85.0,     # ~4 min call + wrap-up
}


def inr_per_usd() -> float:
    try:
        return float(os.getenv("INR_PER_USD", "84.0"))
    except ValueError:
        return 84.0


def llm_cost(model: str, prompt_tokens: int, output_tokens: int) -> dict:
    """Cost of one LLM call in both currencies."""
    rates = LLM_PRICING_USD.get(model, LLM_PRICING_USD["_default"])
    usd = (prompt_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000.0
    return {"model": model, "usd": usd, "inr": usd * inr_per_usd(),
            "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}


@dataclass(frozen=True)
class ReviewPolicy:
    """When may the agent post without a human?

    Deliberately conservative. An LLM posting an unsupervised apology about
    someone's money is the failure mode that ends the pilot, so auto-post is
    gated on low severity AND full grounding AND no compensation promise.
    """

    auto_post_max_severity: int = 2
    require_grounded: bool = True
    forbid_auto_compensation: bool = True
    escalate_to_voice_min_severity: int = 4


REVIEW = ReviewPolicy()


@dataclass(frozen=True)
class JudgePolicy:
    """Thresholds for reading the author. Every number here is a product
    decision, so they live together where a pilot customer can argue with them
    rather than scattered through the scoring code."""

    base_authenticity: float = 0.55   # assume good faith, then let evidence move it

    # Specificity — the strongest positive signal, capped so detail cannot stack.
    specificity_bonus: float = 0.18
    max_specificity_signals: int = 3

    # Account shape
    new_account_days: int = 14
    established_account_days: int = 365
    throwaway_karma: int = 5
    established_karma: int = 500
    throwaway_posts: int = 3
    karma_to_reach_divisor: int = 20

    # History with us
    max_history_bonus: float = 0.20

    # Coordination
    coordinated_duplicate_authors: int = 3

    # Classification bands
    troll_ceiling: float = 0.35
    audience_floor: float = 0.55
    ambiguous_low: float = 0.30
    ambiguous_high: float = 0.55

    # Whether to answer at all. Reach can override suspicion: an unfair post
    # with a large audience still has to be answered in public.
    reply_worthy_authenticity: float = 0.35
    reply_worthy_reach: int = 5_000


JUDGE = JudgePolicy()


@dataclass(frozen=True)
class CrisisPolicy:
    """When does a set of complaints stop being tickets and become an incident?

    Tuned to fire *early and cheaply*. A false "watch" costs a dashboard badge;
    a missed crisis costs the trend everyone screenshots. The asymmetry is the
    whole design.
    """

    window_minutes: int = 90          # how far back a cluster may reach
    similarity: float = 0.72          # cosine floor for "the same complaint"
    watch_cluster: int = 3            # cluster size that raises a watch
    crisis_cluster: int = 6           # cluster size that declares a crisis
    crisis_velocity_per_hour: float = 4.0   # ...or this rate, whichever trips first
    max_history: int = 500            # fingerprints retained in the pattern store


CRISIS = CrisisPolicy()


@dataclass(frozen=True)
class ActionPolicy:
    """Limits on what the executor may do, however the approval was obtained."""

    # Above this, a reversal is not an agent-desk decision at all; it goes to
    # the nodal officer (ESC-04) whoever approves it in the console.
    reversal_ceiling_inr: float = 25_000.0
    # Severity at which the planner opens an internal ticket for a complaint.
    ticket_min_severity: int = 3


ACTIONS = ActionPolicy()
