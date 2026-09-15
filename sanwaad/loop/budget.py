"""Stopping conditions and loop economics.

The single most common failure in a live agent loop is not a wrong answer. It
is a loop that never decides it is finished: it calls the same tool again,
re-reasons over the same observation, and spins on a task it cannot complete,
spending tokens the whole time.

So termination is designed, never left to the model's own sense of "done".
A production loop runs several stopping conditions at once, and whichever
fires first wins:

  done                a final answer passed every in-loop verifier
  needs_human         the remaining judgement has no cheap verifier
  max_iterations      a hard cap on passes through the loop
  budget_exhausted    a token or cost ceiling
  timeout             a wall-clock ceiling
  stalled             the same tool, with the same arguments, again and again
  verifier_exhausted  the answer kept failing a cheap verifier

The economics are why this is engineering rather than a diagram: every
iteration is a full model call, so latency and cost grow linearly with loop
length. An agent that takes ten passes to do a two-pass task costs ten times
as much and fails in ten times as many places. The `Meter` makes that visible
on every run — offline too, as an estimate priced at the loop's model tier.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..config import llm_cost


class StopReason(str, Enum):
    DONE = "done"
    NEEDS_HUMAN = "needs_human"
    MAX_ITERATIONS = "max_iterations"
    BUDGET = "budget_exhausted"
    TIMEOUT = "timeout"
    STALLED = "stalled"
    VERIFIER_EXHAUSTED = "verifier_exhausted"
    ERROR = "error"


# A loop that stops for one of these did its job: it finished, or it correctly
# recognised a decision it should not make alone. Every other stop is a
# failure of the loop's design, whatever the final text says.
CLEAN_STOPS = frozenset({StopReason.DONE, StopReason.NEEDS_HUMAN})


@dataclass(frozen=True)
class LoopBudget:
    max_iterations: int = 8
    max_tokens: int = 20_000        # cumulative, across every pass
    max_cost_inr: float = 2.0
    max_seconds: float = 90.0
    max_verifier_retries: int = 2
    stall_repeats: int = 3          # identical action this many times in a row
    context_tokens: int = 900       # the window budget per pass; compaction above it


def estimate_tokens(text: str) -> int:
    """A deliberately conservative token estimate: UTF-8 bytes / 4.

    Characters / 4 is the usual rule of thumb for English, and it badly
    undercounts Devanagari, where one character is three bytes and tokenises
    accordingly. A budget that is wrong in the cheap direction is a budget that
    does not protect anything.
    """
    return max(1, len((text or "").encode("utf-8")) // 4)


def action_fingerprint(tool: str, args: dict) -> str:
    canonical = json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{tool}\x00{canonical}".encode("utf-8")).hexdigest()[:12]


@dataclass
class Meter:
    """Everything a loop spends, checked before every pass."""

    budget: LoopBudget
    pricing_model: str = "gemini-2.5-flash"
    iterations: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    cost_inr: float = 0.0
    measured_cost: bool = False
    verifier_retries: int = 0
    started: float = field(default_factory=time.perf_counter)
    recent_actions: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self.started

    def charge(self, *, prompt_tokens: int, output_tokens: int,
               cost_inr: Optional[float] = None) -> None:
        """Record one model call. Without a measured cost (offline), price the
        estimated tokens at the loop's tier, so loop length still shows up as
        money in every report."""
        self.model_calls += 1
        self.prompt_tokens += prompt_tokens
        self.output_tokens += output_tokens
        if cost_inr is None:
            cost_inr = llm_cost(self.pricing_model, prompt_tokens, output_tokens)["inr"]
        else:
            self.measured_cost = True
        self.cost_inr += cost_inr

    def record_action(self, tool: str, args: dict) -> None:
        self.recent_actions.append(action_fingerprint(tool, args))
        del self.recent_actions[:-self.budget.stall_repeats]

    def stalled(self) -> bool:
        n = self.budget.stall_repeats
        return len(self.recent_actions) >= n and len(set(self.recent_actions[-n:])) == 1

    def exhausted(self) -> Optional[StopReason]:
        """Checked BEFORE a pass starts, so a budget stops the next call rather
        than noticing after it has been paid for."""
        b = self.budget
        if self.seconds >= b.max_seconds:
            return StopReason.TIMEOUT
        if self.cost_inr >= b.max_cost_inr or self.tokens >= b.max_tokens:
            return StopReason.BUDGET
        if self.iterations >= b.max_iterations:
            return StopReason.MAX_ITERATIONS
        if self.stalled():
            return StopReason.STALLED
        return None

    def snapshot(self) -> dict:
        return {
            "iterations": self.iterations,
            "model_calls": self.model_calls,
            "tokens": self.tokens,
            "cost_inr": round(self.cost_inr, 5),
            "cost_is_estimate": not self.measured_cost,
            "verifier_retries": self.verifier_retries,
            "seconds": round(self.seconds, 3),
        }
