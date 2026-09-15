"""Agent contracts: who owns each step, what it reads, writes and may call.

A multi-agent system is only debuggable if every agent has a defined role, an
input contract and an output contract. Without them you have several prompts
passing a dictionary around, and when a field comes back wrong nobody can say
which agent wrote it.

Sanwaad is a multi-agent system with CENTRALISED orchestration: the LangGraph
state machine owns the control flow, and each agent is a node that reads a
declared slice of state and writes a declared slice back. No agent calls
another agent directly. That single rule is what keeps coordination overhead —
the main cost of going multi-agent — bounded.

Three things here are enforced, not just documented:

- WRITES   `check_writes` runs on every node's output (see graph.py). A node
           that writes a key outside its contract fails loudly in tests,
           instead of silently overwriting another agent's conclusion.
- TOOLS    the tool registry refuses a call from an agent whose contract does
           not list that tool. Least privilege is a lookup, not a convention.
- OUTPUTS  each agent's `output` names the pydantic model its result must
           validate against; the LLM layer rejects anything that does not.

`reads` is documentation. Enforcing it would mean proxying state access on
every node, which costs more clarity than it buys; it is kept accurate by
review and by the trajectory evals, which fail when a step acts on something
it should not have seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

Kind = Literal["agent", "gate", "step"]

# Every node may append to the audit trail and the cost ledger.
ALWAYS_WRITABLE = frozenset({"costs", "events"})


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    kind: Kind       # agent: makes a judgement · gate: decides a route · step: does work
    model: str       # what does the thinking — a tier, rules, or nothing
    reads: frozenset[str]
    writes: frozenset[str]
    tools: frozenset[str] = frozenset()
    output: str = ""


def _spec(name: str, role: str, kind: Kind, model: str, reads: Iterable[str],
          writes: Iterable[str], tools: Iterable[str] = (), output: str = "") -> AgentSpec:
    return AgentSpec(name=name, role=role, kind=kind, model=model,
                     reads=frozenset(reads), writes=frozenset(writes) | ALWAYS_WRITABLE,
                     tools=frozenset(tools), output=output)


AGENTS: dict[str, AgentSpec] = {s.name: s for s in [
    _spec("triage", "Read one comment: what is it about, how bad, which language",
          "agent", "triage tier (cheapest)",
          reads=["complaint"],
          writes=["triage", "retrieval_query", "injection_flagged"],
          output="Triage"),
    _spec("pattern", "Hold the recent window: how many distinct people said this",
          "agent", "none — embeddings and a threshold",
          reads=["complaint", "triage"],
          writes=["pattern", "coordination"],
          output="PatternSignal"),
    _spec("judge", "Read the account behind the words: customer, audience, troll or bot",
          "agent", "rules; triage tier only for the ambiguous band",
          reads=["complaint", "coordination", "triage"],
          writes=["verdict"],
          output="AuthorVerdict"),
    _spec("prioritise", "Fold severity, author and pattern into one queue decision",
          "gate", "rules",
          reads=["triage", "verdict", "pattern"],
          writes=["priority"],
          output="Priority"),
    _spec("retrieve", "Fetch the policy clauses that govern this complaint",
          "step", "none — local ONNX + BM25",
          reads=["triage", "retrieval_query"],
          writes=["citations"],
          output="list[Citation]"),
    _spec("draft", "Ghostwriter: the public reply, in brand voice, every claim cited",
          "agent", "draft tier; reasoning tier on escalation signals",
          reads=["complaint", "triage", "citations", "grounding", "pattern",
                 "injection_flagged", "revision_count"],
          writes=["draft", "revision_count", "guardrails"],
          output="Draft"),
    _spec("ground_check", "Verify every claim in the draft traces to a clause",
          "gate", "draft tier",
          reads=["draft", "citations"],
          writes=["grounding"],
          output="GroundingVerdict"),
    _spec("plan", "Ghostwriter's private half: propose the actions that fix the issue",
          "agent", "plan tier — validation downstream is code",
          reads=["complaint", "triage", "citations", "verdict"],
          writes=["actions"],
          tools=["lookup_transaction"],
          output="list[ProposedAction]"),
    _spec("review_gate", "Decide who approves: the auto-post policy, or a human",
          "gate", "rules, then a human",
          reads=["triage", "draft", "grounding", "guardrails", "injection_flagged",
                 "pattern", "actions"],
          writes=["review"],
          output="Review"),
    _spec("publish", "Post the approved reply, re-checking it first",
          "step", "none",
          reads=["review", "complaint"],
          writes=["published"],
          tools=["post_reply"]),
    _spec("act", "Executor: carry out approved actions, re-validating before each",
          "step", "none",
          reads=["actions", "review", "complaint", "triage"],
          writes=["action_results"],
          tools=["open_ticket", "initiate_reversal"]),
    _spec("escalation", "Apply the written escalation rules",
          "gate", "rules",
          reads=["triage", "complaint", "pattern", "verdict"],
          writes=["escalation"]),
    _spec("voice", "Hand the case to a phone call that uses the same clauses",
          "agent", "voice tier (latency-bound)",
          reads=["escalation", "triage", "review", "citations", "action_results"],
          writes=["voice"],
          output="VoiceOutcome"),
    _spec("close", "Roll up cost, actions and the consistency receipt",
          "step", "none",
          reads=["costs", "draft", "voice", "escalation", "action_results"],
          writes=["closure"]),

    # --- the support loop (sanwaad/loop) -----------------------------------
    # Not graph nodes: they write no case state. Their contracts still bound
    # what they may call, which is the part of a contract that matters in a loop.
    _spec("resolver", "Support loop: act, observe, verify, retry — until done or handed over",
          "agent", "loop tier, one decision per pass",
          reads=[], writes=[],
          tools=["lookup_transaction", "reversal_status", "search_policy", "open_ticket"],
          output="FinalAnswer"),
    _spec("policy_subagent", "Answer one policy question in a clean, scoped context",
          "agent", "retrieval, then the draft tier; returns only an answer and clause ids",
          reads=[], writes=[],
          tools=["search_policy"],
          output="SubAgentResult"),
]}


class ContractViolation(RuntimeError):
    """A node wrote state it does not own, or an agent has no contract."""


def check_writes(node: str, delta: dict) -> None:
    spec = AGENTS.get(node)
    if spec is None:
        raise ContractViolation(f"node {node!r} has no contract in agents.AGENTS")
    extra = set(delta or {}) - spec.writes
    if extra:
        raise ContractViolation(
            f"node {node!r} wrote {sorted(extra)}, outside its contract "
            f"{sorted(spec.writes - ALWAYS_WRITABLE)}")


def may_call(agent: str, tool: str) -> bool:
    spec = AGENTS.get(agent)
    return bool(spec and tool in spec.tools)
