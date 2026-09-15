"""The policy sub-agent: multi-agent as a context-management strategy.

A long support loop cannot afford to pull three full policy clauses into its
own window every time a customer asks "how long does this take?". Each clause
would sit there for every remaining pass, crowding out the observations the
next decision actually depends on.

So the question is delegated. The sub-agent starts with a clean context that
holds nothing but the question and the clauses it retrieves, answers in one
bounded step, and returns only a short answer and the clause ids behind it. The
main loop pays for one compact observation instead of a page of policy.

That is the practical reason most multi-agent systems exist: isolation of
context for a bounded task, not a display of capability. `consumed_tokens`
against `returned_tokens` is the saving, and the loop eval reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from ..config import TIER_DRAFT, TIER_REASONING
from ..context import untrusted, with_trust_rules
from ..llm import structured
from .budget import estimate_tokens

_SYSTEM = """You answer one question about NimbusPay's written support policy.

You are given the question and the policy clauses retrieved for it. Answer only
from those clauses, in at most two sentences. If they do not answer the
question, say so plainly. List in `clause_ids` only the ids of clauses you
actually relied on. Never put a clause id inside `answer`."""


class PolicyAnswer(BaseModel):
    answer: str = Field(max_length=400)
    clause_ids: list[str] = Field(default_factory=list)


@dataclass
class SubAgentResult:
    question: str
    answer: str
    clause_ids: list[str]
    prompt_tokens: int
    output_tokens: int
    cost_inr: Optional[float]
    consumed_tokens: int      # everything the sub-agent held in its own context
    returned_tokens: int      # what the main loop receives

    def as_observation(self) -> str:
        cites = "".join(f"[{c}]" for c in self.clause_ids) or "[no clause]"
        return f"policy sub-agent {cites}: {self.answer}"


def _extractive(hits: list[dict]) -> dict:
    """Offline answer: the first sentence of the best clause. Never paraphrased,
    so it can never say something the clause does not."""
    if not hits:
        return {"answer": "No policy clause answers this question.", "clause_ids": []}
    snippet = hits[0]["snippet"]
    first = snippet.split(". ")[0].rstrip(".…") + "."
    return {"answer": first[:380], "clause_ids": [hits[0]["clause_id"]]}


async def consult_policy(question: str) -> SubAgentResult:
    from ..tools import REGISTRY

    found = await REGISTRY.call("search_policy", {"query": question[:300], "k": 3},
                                agent="policy_subagent")
    hits = found.output["results"] if found.ok else []
    clauses = "\n".join(f"[{h['clause_id']}] {h['heading']}: {h['snippet']}" for h in hits)

    system = with_trust_rules(_SYSTEM)
    user = (f"Question:\n{untrusted('customer_question', question)}\n\n"
            f"Retrieved clauses (trusted policy):\n{clauses or '(none retrieved)'}")

    result, cost = await structured(
        model=TIER_DRAFT, fallback_model=TIER_REASONING, system=system, user=user,
        schema=PolicyAnswer, temperature=0.0, stage="policy_subagent",
        offline_fallback=_extractive(hits), max_output_tokens=200, timeout_s=15.0,
    )

    retrieved = {h["clause_id"] for h in hits}
    ids = [c for c in result.clause_ids if c in retrieved]   # cannot cite what it never saw
    answer = result.answer.strip()

    offline = cost.get("model") in ("offline", "degraded") and not cost.get("prompt_tokens")
    prompt_tokens = cost.get("prompt_tokens") or estimate_tokens(system + user)
    output_tokens = cost.get("output_tokens") or estimate_tokens(result.model_dump_json())
    out = SubAgentResult(
        question=question, answer=answer, clause_ids=ids,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        cost_inr=None if offline else cost.get("inr"),
        consumed_tokens=prompt_tokens + output_tokens, returned_tokens=0,
    )
    out.returned_tokens = estimate_tokens(out.as_observation())
    return out
