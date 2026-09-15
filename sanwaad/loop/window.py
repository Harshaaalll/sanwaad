"""The context window of a long-running loop, managed as a budget.

A single-step prompt can be designed once. A loop's context grows on its own:
every tool result and every intermediate step is appended, pass after pass.
Long before the hard context limit, quality drops — the window fills with stale,
low-signal tokens and the model loses the thread. That is context rot, and it
is why long-running agents start strong and degrade halfway through.

So on every pass this module decides what stays, what is compressed and what is
dropped, using three techniques:

1.  TOOL RESULT SHAPING. A lookup can return a large record. It is filtered to
    the fields this conversation needs, capped, and redacted BEFORE it enters
    the window. The raw size and the shaped size are both counted, so the saving
    is measurable.
2.  EXTERNAL MEMORY. Facts are written to a scratchpad as they are observed and
    read back by relevance, instead of dragging the full history along. A fact
    survives compaction even when the step that produced it does not.
3.  COMPACTION. When the rendered window exceeds its token budget, the oldest
    steps are folded into one-line summaries. The most recent steps always stay
    verbatim, because that is what the next decision is about.

Verifiers do not read the window. They read everything ever observed
(`all_seen_text`), so compaction can never make a true fact look invented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from ..guardrails import redact
from .budget import estimate_tokens

KEEP_RECENT = 2          # steps always kept verbatim
MAX_SUMMARY_LINES = 6    # the summary is compacted too, or it becomes the new rot
MAX_FACTS_IN_VIEW = 8

CLAUSE_ID = re.compile(r"\b(?:BV|RFD|BIL|KYC|ESC|PRV)-\d{2}\b")
_WORD = re.compile(r"[a-z0-9ऀ-ॿ]+")


@dataclass
class Fact:
    key: str
    text: str
    source: str
    iteration: int


@dataclass
class Observation:
    iteration: int
    source: str
    text: str
    raw_tokens: int
    tokens: int


Shaped = tuple[str, list[tuple[str, str]]]


# ---------------------------------------------------------------------------
# Tool result shaping
# ---------------------------------------------------------------------------

def _shape_lookup(out: dict) -> Shaped:
    matches = out.get("matches") or []
    lines, facts = [], []
    for m in matches[:3]:
        line = (f"{m['reference']}: {m['kind'].replace('_', ' ')} ₹{m['amount_inr']:,.0f} "
                f"at {m['merchant']}, {m['status'].replace('_', ' ')}, {m['age_days']:g}d old")
        lines.append(line)
        facts.append((f"txn:{m['reference']}", line))
    if len(matches) > 3:
        lines.append(f"(+{len(matches) - 3} more matches omitted)")
    if not lines:
        return "no matching transaction for this customer", [("txn:none", "no matching transaction")]
    return "\n".join(lines), facts


def _shape_reversal(out: dict) -> Shaped:
    ref = out.get("reference", "?")
    if not out.get("found"):
        line = f"{ref}: no reversal on record"
    else:
        line = f"{ref}: reversal {out['reversal_id']} {out['status']}"
        if out.get("eta_hours"):
            line += f", about {out['eta_hours']}h until it credits"
    return line, [(f"reversal:{ref}", line)]


def _shape_policy(out: dict) -> Shaped:
    lines, facts = [], []
    for r in (out.get("results") or [])[:3]:
        line = f"[{r['clause_id']}] {r['heading']}: {r['snippet']}"
        lines.append(line)
        facts.append((f"clause:{r['clause_id']}", line))
    return ("\n".join(lines) or "no policy clause matched"), facts


def _shape_ticket(out: dict) -> Shaped:
    line = f"ticket {out['ticket_id']} {'already open' if out.get('duplicate') else 'opened'}"
    return line, [("ticket", line)]


SHAPERS: dict[str, Callable[[dict], Shaped]] = {
    "lookup_transaction": _shape_lookup,
    "reversal_status": _shape_reversal,
    "search_policy": _shape_policy,
    "open_ticket": _shape_ticket,
}


def shape(tool: str, output: Optional[dict], error: Optional[str] = None) -> Shaped:
    if error:
        line = f"{tool} failed — {error}"
        return line, [(f"error:{tool}", line)]
    shaper = SHAPERS.get(tool)
    if shaper:
        return shaper(output or {})
    text = json.dumps(output, ensure_ascii=False, default=str)
    return (text[:400] + ("…" if len(text) > 400 else "")), []


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

class ContextWindow:
    def __init__(self, goal: str, *, budget_tokens: int, memory: bool = True,
                 compaction: bool = True, shaping: bool = True):
        self.goal = goal
        self.budget_tokens = budget_tokens
        self.memory = memory
        self.compaction = compaction
        self.shaping = shaping

        self.observations: list[Observation] = []
        self.summary: list[str] = []
        self.facts: dict[str, Fact] = {}
        self.clauses_seen: set[str] = set()
        self._seen: list[str] = [goal]

        self.compactions = 0
        self.peak_tokens = 0
        self.raw_tokens_in = 0
        self.window_tokens_in = 0

    # --- writing ----------------------------------------------------------------

    def observe(self, iteration: int, tool: str, output: Optional[dict] = None,
                error: Optional[str] = None) -> Observation:
        raw = json.dumps(output if error is None else {"error": error},
                         ensure_ascii=False, default=str)
        if self.shaping:
            text, facts = shape(tool, output, error)
        else:
            text, facts = raw, []
        text = redact(text)[0]

        obs = Observation(iteration, tool, text, estimate_tokens(raw), estimate_tokens(text))
        self.raw_tokens_in += obs.raw_tokens
        self.window_tokens_in += obs.tokens
        self.observations.append(obs)
        self._seen.append(text)
        if tool == "search_policy" and output:
            # Ids from the result fields only. A snippet can MENTION another
            # clause ("follows the refund path in [RFD-01]"), and a mention is
            # not a retrieval — citing it would pass a check it should fail.
            self.clauses_seen.update(r["clause_id"] for r in output.get("results") or [])

        for key, fact_text in facts:
            fact_text = redact(fact_text)[0]
            self._seen.append(fact_text)
            if self.memory:
                self.facts[key] = Fact(key, fact_text, tool, iteration)
        return obs

    def note(self, iteration: int, source: str, text: str,
             facts: Optional[list[tuple[str, str]]] = None) -> None:
        """A non-tool observation: verifier feedback, or a sub-agent's result."""
        text = redact(text)[0]
        tokens = estimate_tokens(text)
        self.observations.append(Observation(iteration, source, text, tokens, tokens))
        self.window_tokens_in += tokens
        self._seen.append(text)
        if source.startswith("subagent"):
            self.clauses_seen.update(key.split(":", 1)[1] for key, _ in facts or []
                                     if key.startswith("clause:"))
        for key, fact_text in facts or []:
            if self.memory:
                self.facts[key] = Fact(key, redact(fact_text)[0], source, iteration)

    # --- reading ----------------------------------------------------------------

    def all_seen_text(self) -> str:
        return "\n".join(self._seen)

    def fact(self, key: str) -> Optional[Fact]:
        return self.facts.get(key)

    def relevant_facts(self) -> list[Fact]:
        """Facts ranked by overlap with the goal, then recency. Relevance is a
        word overlap on purpose: the scratchpad is small, and an embedding call
        per pass would cost more than the facts it ranks."""
        goal_words = set(_WORD.findall(self.goal.lower()))

        def score(f: Fact) -> tuple[int, int]:
            return (len(goal_words & set(_WORD.findall(f.text.lower()))), f.iteration)

        return sorted(self.facts.values(), key=score, reverse=True)[:MAX_FACTS_IN_VIEW]

    def render(self) -> str:
        """The context for the next pass, fitted to its budget.

        In order of what is given up first: old steps are compacted into
        summaries; then the summaries themselves shrink to a count; only then
        are remembered facts shown more sparingly, least relevant first. Facts
        outlast summaries because keeping them is what external memory is for.
        The most recent steps are never cut — they are what the next decision
        is about.
        """
        facts_limit, summary_limit = MAX_FACTS_IN_VIEW, MAX_SUMMARY_LINES
        while estimate_tokens(self._compose(facts_limit, summary_limit)) > self.budget_tokens:
            if self.compaction and len(self.observations) > KEEP_RECENT:
                self._compact_oldest()
            elif self.compaction and len(self.summary) > 1 and summary_limit > 1:
                summary_limit = min(summary_limit, len(self.summary)) - 1
            elif self.compaction and self.memory and facts_limit > 0:
                facts_limit -= 1
            else:
                break
        text = self._compose(facts_limit, summary_limit)
        self.peak_tokens = max(self.peak_tokens, estimate_tokens(text))
        return text

    def tokens(self) -> int:
        return estimate_tokens(self._compose())

    def _compose(self, facts_limit: int = MAX_FACTS_IN_VIEW,
                 summary_limit: int = MAX_SUMMARY_LINES) -> str:
        parts = [f"Goal: {self.goal}"]
        if self.summary:
            limit = max(1, summary_limit)
            if len(self.summary) <= limit:
                lines = self.summary
            else:
                kept = self.summary[-(limit - 1):] if limit > 1 else []
                lines = [f"{len(self.summary) - len(kept)} earlier steps"] + kept
            parts.append("Earlier steps (compacted):\n- " + "\n- ".join(lines))
        if self.memory and self.facts:
            # A fact whose step is still verbatim below would be paid for twice.
            recent = {o.iteration for o in self.observations}
            shown = [f for f in self.relevant_facts() if f.iteration not in recent][:facts_limit]
            if shown:
                parts.append("Known facts:\n- " + "\n- ".join(f.text for f in shown))
        if self.observations:
            parts.append("Recent steps:\n" + "\n".join(
                f"[{o.iteration}] {o.source}: {o.text}" for o in self.observations))
        return "\n\n".join(parts)

    def _compact_oldest(self) -> None:
        obs = self.observations.pop(0)
        first_line = (obs.text.splitlines() or [""])[0]
        self.summary.append(f"step {obs.iteration} {obs.source}: {first_line[:90]}")
        self.compactions += 1

    def stats(self) -> dict:
        return {
            "peak_tokens": self.peak_tokens,
            "compactions": self.compactions,
            "raw_tool_tokens": self.raw_tokens_in,
            "window_tool_tokens": self.window_tokens_in,
            "facts": len(self.facts),
        }
