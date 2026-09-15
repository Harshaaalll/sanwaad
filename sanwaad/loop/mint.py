"""MINT: Minimal Intelligence, Necessary Tools.

Build the minimal system end to end before adding a single layer of
complexity, then add layers one at a time, and only when the layer below has
shown a real, measured need for the next. Every layer answers two questions
before it goes in: how does it break, and what does the system do when it does?

The opposite is the familiar failure: five agents, a vector database and
long-term memory wired together before anyone has confirmed the basic loop
works. Two weeks later nobody can say which layer is failing.

For Sanwaad's support loop the ladder is:

    M0  minimal loop              a prompt and a loop: message in, answer out
    M1  + tool use                look things up instead of guessing
    M2  + continuous evaluation   cheap verifiers in the loop, the loop eval on every change
    M3  + state and memory        scratchpad facts, compaction, memory across turns
    M4  + workflows               a dedicated flow per intent: tools and budget
    M5  + human-in-the-loop       hand-over for money and legal language,
          and multi-agent         a policy sub-agent for context isolation

Two things here are executable rather than advice:

- `config_for(rung)` builds the loop configuration for a rung, so the same
  conversation can be run at every rung and compared.
- `check_layering(config)` refuses a configuration that switches on a layer
  without the layers beneath it — multi-agent without evaluation, memory
  without tools. A shortcut up the ladder fails loudly, at start-up.

`python -m sanwaad.evals.loop_eval --ladder` runs the loop scenarios at every
rung, so each layer's value is a number, not a belief.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .budget import LoopBudget
from .kernel import LoopConfig

LOOP_TOOLS = ("lookup_transaction", "reversal_status", "search_policy", "open_ticket")


@dataclass(frozen=True)
class Rung:
    level: int
    name: str
    adds: str
    need: str            # what the rung below showed, that justified this one
    breaks: str          # how this layer fails
    system_does: str     # what the system does when it fails


LADDER: list[Rung] = [
    Rung(0, "Minimal loop",
         "one prompt, one loop: the message comes in, an answer goes out",
         "none — this is where every system starts",
         "it cannot see any transaction, so every money question is unanswerable",
         "says plainly that a colleague will check, instead of inventing details"),
    Rung(1, "Tool use",
         "lookup_transaction, reversal_status, search_policy, open_ticket",
         "at M0 no money question gets a real answer",
         "a tool times out, errors, or returns nothing",
         "the registry retries reads; a failure becomes an observation, and the loop opens a ticket rather than guessing"),
    Rung(2, "Continuous evaluation",
         "cheap verifiers inside the loop, and the loop eval run on every change",
         "at M1 replies ship timelines that no retrieved clause supports",
         "an answer states something nothing it observed supports",
         "the verdict becomes feedback for another pass; bounded retries, then verifier_exhausted"),
    Rung(3, "State and memory",
         "scratchpad facts, compaction under a context budget, memory across turns",
         "at M2 long conversations outgrow the window and repeat customers get duplicate tickets",
         "the window rots on long tasks; memory recalls something stale",
         "compaction holds the window under budget; memory keeps decisions, never live facts"),
    Rung(4, "Workflows",
         "a dedicated flow per intent, each with its own tools and pass budget",
         "at M3 simple questions still get the whole tool set and budget",
         "a message lands in the wrong workflow",
         "unrecognised intents get the no-tool workflow and a safe answer; budgets cap wasted passes"),
    Rung(5, "Human-in-the-loop and multi-agent",
         "hand-over for money and regulatory language; a policy sub-agent with its own context",
         "at M4 refund requests end without anyone able to approve them, and policy text crowds the window",
         "a decision needs judgement no cheap check can give",
         "the loop stops with needs_human and a validated proposal; the sub-agent returns only a short answer"),
]


class LayeringError(ValueError):
    """A layer was switched on without the layers it is built on."""


def config_for(rung: int, *, budget: LoopBudget = LoopBudget(), record: bool = True) -> LoopConfig:
    if not 0 <= rung <= 5:
        raise ValueError(f"MINT rungs are 0-5, not {rung}")
    config = LoopConfig(budget=replace(budget, max_iterations=min(budget.max_iterations, 4))
                        if rung == 0 else budget, record=record)
    if rung >= 1:
        config = replace(config, tools=LOOP_TOOLS)
    if rung >= 2:
        config = replace(config, verifiers=True)
    if rung >= 3:
        config = replace(config, memory=True, compaction=True)
    if rung >= 4:
        config = replace(config, workflows=True)
    if rung >= 5:
        config = replace(config, human_handoff=True, subagents=True)
    return config


def rung_of(config: LoopConfig) -> int:
    """The highest layer a configuration switches on."""
    if config.human_handoff or config.subagents:
        return 5
    if config.workflows:
        return 4
    if config.memory or config.compaction:
        return 3
    if config.verifiers:
        return 2
    if config.tools:
        return 1
    return 0


def check_layering(config: LoopConfig) -> int:
    """Return the configuration's rung, or raise if it skipped a layer."""
    rung = rung_of(config)
    missing = []
    if rung >= 1 and not config.tools:
        missing.append("tool use")
    if rung >= 2 and not config.verifiers:
        missing.append("continuous evaluation")
    if rung >= 3 and not (config.memory and config.compaction):
        missing.append("state and memory")
    if rung >= 4 and not config.workflows:
        missing.append("workflows")
    if missing:
        raise LayeringError(
            f"{LADDER[rung].name} (M{rung}) is switched on without: {', '.join(missing)}. "
            "Add layers one at a time, each only after the one below has shown the need.")
    return rung
