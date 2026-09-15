"""Graph assembly: the orchestration layer.

Why a state machine instead of one large system prompt: prompt-driven agents
end up with tool descriptions full of capitalised FORBIDDEN clauses, which is
what prompt engineering looks like when it is being asked to do a control-flow
job. Phase transitions belong in edges, where
they are deterministic, inspectable, and testable without spending a token.

Autonomy is not the same as lack of structure. Most of this flow is known in
advance, so it is a pipeline with explicit branches — retry loops, approval
gates, fallback paths — rather than an agent loop deciding its own next step.
Models make judgements INSIDE nodes; code decides which node runs next.

Every node is wrapped so its output is checked against its contract in
`agents.AGENTS`. A node writing state it does not own fails immediately,
which is the difference between a multi-agent system and several prompts
overwriting one dictionary.
"""

from __future__ import annotations

import functools

from langgraph.graph import END, START, StateGraph

from ..agents import check_writes
from .nodes import (
    act_node,
    close_node,
    draft_node,
    escalation_node,
    ground_check_node,
    judge_node,
    pattern_node,
    plan_node,
    prioritise_node,
    publish_node,
    retrieve_node,
    review_gate_node,
    triage_node,
    voice_node,
)
from .state import GrievanceState

MAX_REVISIONS = 2


def _contracted(name: str, fn):
    @functools.wraps(fn)
    async def run(state: GrievanceState) -> dict:
        delta = await fn(state)
        check_writes(name, delta)
        return delta
    return run


def _after_prioritise(state: GrievanceState) -> str:
    """The only place a case is dropped without a reply.

    By this point three independent readings agree it is not worth drafting
    for: triage found no grievance, the judge found no credible author, and
    the pattern agent found nothing else like it. Any one of them alone would
    be a bad reason to stay silent.
    """
    return "retrieve" if (state.get("priority") or {}).get("drafting", True) else "close"


def _after_ground_check(state: GrievanceState) -> str:
    grounding = state.get("grounding") or {}
    if grounding.get("grounded") or grounding.get("unavailable"):
        # Unavailable goes on to a person too — rewriting a draft cannot fix a
        # checker that is down. Ungrounded, it cannot be auto-posted.
        return "plan"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        # Two failed attempts to ground a claim means the policy index does not
        # answer this complaint. That is a knowledge gap, not a drafting bug,
        # and a human should see it rather than a third model call burning
        # tokens on the same missing clause. The review gate will hold it.
        return "plan"
    return "draft"


def _after_review(state: GrievanceState) -> str:
    """A rejected reply is not posted — but the internal fix may still proceed.

    A reviewer can refuse the wording and still approve the reversal. Tying the
    private action to the public reply would mean a badly drafted sentence
    blocks a customer's money.
    """
    return "publish" if (state.get("review") or {}).get("decision") != "reject" else "act"


def _after_act(state: GrievanceState) -> str:
    rejected = (state.get("review") or {}).get("decision") == "reject"
    return "close" if rejected else "escalation"


def _after_escalation(state: GrievanceState) -> str:
    return "voice" if (state.get("escalation") or {}).get("needed") else "close"


NODES = {
    "triage": triage_node,
    "pattern": pattern_node,
    "judge": judge_node,
    "prioritise": prioritise_node,
    "retrieve": retrieve_node,
    "draft": draft_node,
    "ground_check": ground_check_node,
    "plan": plan_node,
    "review_gate": review_gate_node,
    "publish": publish_node,
    "act": act_node,
    "escalation": escalation_node,
    "voice": voice_node,
    "close": close_node,
}


def build_graph(checkpointer=None):
    g = StateGraph(GrievanceState)

    for name, fn in NODES.items():
        g.add_node(name, _contracted(name, fn))

    g.add_edge(START, "triage")
    # Everything reaches the pattern agent, including praise and off-topic
    # chatter: "nine people said something odd about the new checkout" is a
    # signal even when not one of them filed a complaint, and fingerprinting
    # is an embedding — local and free.
    g.add_edge("triage", "pattern")
    # pattern -> judge -> prioritise is a real dependency chain: the judge's
    # coordination signal is a fact about the window, which only pattern holds.
    g.add_edge("pattern", "judge")
    g.add_edge("judge", "prioritise")
    g.add_conditional_edges("prioritise", _after_prioritise,
                            {"retrieve": "retrieve", "close": "close"})
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "ground_check")
    g.add_conditional_edges("ground_check", _after_ground_check,
                            {"draft": "draft", "plan": "plan"})
    g.add_edge("plan", "review_gate")
    g.add_conditional_edges("review_gate", _after_review,
                            {"publish": "publish", "act": "act"})
    g.add_edge("publish", "act")
    g.add_conditional_edges("act", _after_act, {"escalation": "escalation", "close": "close"})
    g.add_conditional_edges("escalation", _after_escalation, {"voice": "voice", "close": "close"})
    g.add_edge("voice", "close")
    g.add_edge("close", END)

    return g.compile(checkpointer=checkpointer)
