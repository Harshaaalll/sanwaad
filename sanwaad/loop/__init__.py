"""Loop engineering for Sanwaad's support conversations.

    budget.py    stopping conditions and loop economics
    window.py    context budget: shaping, external memory, compaction
    verify.py    cheap verifiers in the loop, humans where verification is expensive
    kernel.py    act, observe, verify, retry
    policy.py    what decides the next action: Gemini, or the offline stand-in
    subagent.py  a policy sub-agent with its own clean context
    support.py   the running example: workflows and memory across turns
    mint.py      Minimal Intelligence, Necessary Tools — the layer ladder
    outer.py     the developer's and the world's loops around the agent's

Taught in DESIGN.md, Part II.
"""

from .budget import CLEAN_STOPS, LoopBudget, StopReason
from .kernel import Decision, LoopConfig, LoopRun, run_loop
from .mint import LADDER, LayeringError, check_layering, config_for

__all__ = [
    "CLEAN_STOPS", "Decision", "LADDER", "LayeringError", "LoopBudget", "LoopConfig",
    "LoopRun", "StopReason", "check_layering", "config_for", "run_loop",
]
