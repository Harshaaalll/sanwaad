"""Watch the support loop think, pass by pass.

    python -m sanwaad.loop             the running example at MINT rung 5
    python -m sanwaad.loop --rung 1    the same conversations with fewer layers

Offline it uses the scripted policy; with GOOGLE_API_KEY in .env it uses Gemini.
"""

from __future__ import annotations

import asyncio
import sys

from ..tools import REGISTRY, ErrorCode, ToolError
from ..tools.ledger import BACKEND
from .kernel import Decision
from .mint import LADDER, config_for
from .support import CONVERSATIONS_PATH, handle_turn

BOLD, DIM, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
STOP_COLOUR = {"done": GREEN, "needs_human": YELLOW}


class NeverFinishes:
    """A policy that keeps asking the same question. The loop must stop it."""

    name = "never-finishes"

    async def decide(self, view, ctx):
        return Decision(thought="let me check that again", action="lookup_transaction",
                        args={"amount_inr": 640}), {}


def _show(title: str, run) -> None:
    colour = STOP_COLOUR.get(run.stop_reason.value, RED)
    m, w = run.meter, run.window
    print(f"\n{BOLD}{title}{RESET}")
    print(f"  {DIM}“{run.message}”{RESET}")
    for step in run.steps:
        outcome = step.get("outcome", "")
        mark = RED if ("failed" in outcome or "error" in outcome or "blocked" in outcome) else DIM
        args = ", ".join(f"{k}={v}" for k, v in step["args"].items())
        print(f"  {step['iteration']}. {step['action']:<20}{DIM}{args[:46]:<46}{RESET} {mark}{outcome}{RESET}")
    for verdict in run.verdicts:
        if not verdict["passed"]:
            print(f"     {RED}✗ {verdict['verifier']}{RESET} {DIM}{verdict['feedback'][:88]}{RESET}")
    print(f"  → {colour}{run.stop_reason.value}{RESET}  {DIM}{run.detail[:90]}{RESET}")
    if run.answer:
        cites = f"  {DIM}cites {run.answer.cited_clauses}{RESET}" if run.answer.cited_clauses else ""
        print(f"  “{run.answer.text}”{cites}")
    if run.handoff and run.handoff.get("proposal"):
        p = run.handoff["proposal"]
        print(f"  {YELLOW}for a person to approve:{RESET} reversal {p['reference']} ₹{p['amount_inr']:,.0f}")
    estimate = " (estimated)" if m["cost_is_estimate"] else ""
    print(f"  {DIM}{m['iterations']} passes · ₹{m['cost_inr']:.4f}{estimate} · "
          f"peak context {w['peak_tokens']} tokens · {w['compactions']} compactions{RESET}")


async def main(argv: list[str]) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    rung = int(argv[argv.index("--rung") + 1]) if "--rung" in argv else 5
    print(f"{BOLD}MINT rung M{rung}: {LADDER[rung].name}{RESET}  {DIM}{LADDER[rung].adds}{RESET}")

    BACKEND.reset()
    CONVERSATIONS_PATH.unlink(missing_ok=True)
    config = config_for(rung)

    _show("1. Where's my refund?",
          await handle_turn("Where is my refund for the ₹640 double debit?",
                            handle="u/karthik_rn", config=config))
    _show("2. A timeline the agent 'remembers' wrongly",
          await handle_turn("My UPI transfer of ₹2,000 failed 2 days ago, when will it come back?",
                            handle="u/asha_v", config=config))
    _show("3. A policy question",
          await handle_turn("How long do duplicate debit reversals take?",
                            handle="u/curious", config=config))

    REGISTRY.inject_fault("lookup_transaction",
                          ToolError(code=ErrorCode.UPSTREAM, message="ledger unavailable", retryable=True),
                          times=3)
    _show("4. The ledger is down",
          await handle_turn("Where is my refund for the ₹640 double debit?",
                            handle="u/karthik_rn", config=config))
    REGISTRY.clear_faults()

    _show("5. A policy that never finishes",
          await handle_turn("Where is my refund?", handle="u/karthik_rn", config=config,
                            policy=NeverFinishes()))

    print(f"\n{DIM}Every layer's value, measured: python -m sanwaad.evals.loop_eval --ladder{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
