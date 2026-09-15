"""The loop kernel: act, observe, verify, retry.

This is the primitive underneath every agent. The model is called with the
current context; it decides on one action, usually a tool call; the harness
executes that action; the result comes back as an observation; the observation
is appended to the context; and the model is called again. Reason, act,
observe, repeat — the ReAct pattern.

Loop engineering is designing everything around that primitive rather than
hand-writing each step: what happens between passes, when the loop checks its
own work, when it is allowed to stop, and what it does when a step fails.

What the kernel owns, and the model never does:

- WHICH ACTIONS EXIST. A tool the configuration does not grant is refused as
  an observation, not an exception, so the agent can recover.
- EXECUTION. Tool calls go through the tool registry, the harness from the
  earlier lessons: contracts, least privilege, timeouts, safe retries, audit.
- TERMINATION. Budgets are checked before every pass. "I'm done" is a request;
  it is granted only when the answer passes every cheap verifier.
- THE HAND-OVER. A validated money action, or language no cheap check can
  judge, ends the loop with `needs_human` — never with the agent deciding.
- THE RECORD. Every run is written to `loop_runs.jsonl`, redacted, because the
  outer loops (the developer's and the world's) learn from real runs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from ..config import DATA_DIR
from ..guardrails import redact
from ..obs import TRACER
from .budget import CLEAN_STOPS, LoopBudget, Meter, StopReason, estimate_tokens
from .verify import FinalAnswer, needs_human, run_verifiers
from .window import ContextWindow

LOOP_RUNS_PATH = DATA_DIR / "loop_runs.jsonl"

# Actions the kernel itself handles. Everything else must be a granted tool.
FINAL = "final"
CONSULT_POLICY = "consult_policy"     # delegate to the policy sub-agent
PROPOSE_REVERSAL = "propose_reversal" # validated in code, then handed to a person
HANDOFF = "handoff"                   # the agent asks for a person itself
KERNEL_ACTIONS = frozenset({FINAL, CONSULT_POLICY, PROPOSE_REVERSAL, HANDOFF})


class Decision(BaseModel):
    """One pass's decision. Structured, so the kernel never parses prose."""

    thought: str = Field(default="", description="one short sentence: why this action")
    action: str = Field(description="a tool name, consult_policy, propose_reversal, handoff or final")
    args: dict = Field(default_factory=dict)
    answer: Optional[FinalAnswer] = None


@dataclass(frozen=True)
class LoopConfig:
    """Which capabilities this loop runs with. See mint.py for why they are
    switched on one at a time, and in this order."""

    tools: tuple[str, ...] = ()
    verifiers: bool = False
    memory: bool = False
    compaction: bool = False
    shaping: bool = True
    workflows: bool = False
    human_handoff: bool = False
    subagents: bool = False
    budget: LoopBudget = LoopBudget()
    agent: str = "resolver"
    record: bool = True


@dataclass
class LoopContext:
    """What a policy may consult when deciding. The window is the only record
    of what happened; the rest is who is asking and what the loop may do."""

    run_id: str
    message: str
    handle: str
    case_id: str
    config: LoopConfig
    window: ContextWindow
    meter: Meter
    workflow: Optional[str] = None
    variant: Optional[str] = None


class LoopPolicy(Protocol):
    name: str

    async def decide(self, view: str, ctx: LoopContext) -> tuple[Decision, dict]:
        """Return the next decision and its usage: prompt_tokens, output_tokens
        and, when a model was really called, cost_inr."""


@dataclass
class LoopRun:
    run_id: str
    message: str
    handle: str
    policy: str
    stop_reason: StopReason
    answer: Optional[FinalAnswer]
    steps: list[dict] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)
    handoff: Optional[dict] = None
    meter: dict = field(default_factory=dict)
    window: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    workflow: Optional[str] = None
    variant: Optional[str] = None
    detail: str = ""
    # The live window, kept for post-hoc checks. Never serialised.
    window_obj: Optional[ContextWindow] = field(default=None, repr=False, compare=False)

    @property
    def clean(self) -> bool:
        return self.stop_reason in CLEAN_STOPS

    def record(self) -> dict:
        return {
            "at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "handle": self.handle,
            "message": redact(self.message)[0],
            "policy": self.policy,
            "variant": self.variant,
            "workflow": self.workflow,
            "stop_reason": self.stop_reason.value,
            "clean": self.clean,
            "detail": self.detail,
            "answer": redact(self.answer.text)[0] if self.answer else None,
            "cited_clauses": self.answer.cited_clauses if self.answer else [],
            "steps": self.steps,
            "verdicts": self.verdicts,
            "handoff": self.handoff,
            "meter": self.meter,
            "window": self.window,
            "config": self.config,
        }


def _config_dict(config: LoopConfig) -> dict:
    d = asdict(config)
    d["tools"] = list(config.tools)
    return d


async def run_loop(message: str, *, handle: str, policy: LoopPolicy, config: LoopConfig,
                   case_id: Optional[str] = None, workflow: Optional[str] = None,
                   variant: Optional[str] = None,
                   memory_notes: Optional[list[str]] = None) -> LoopRun:
    run_id = f"loop_{uuid.uuid4().hex[:10]}"
    case_id = case_id or f"case_{uuid.uuid4().hex[:10]}"
    window = ContextWindow(message, budget_tokens=config.budget.context_tokens,
                           memory=config.memory, compaction=config.compaction,
                           shaping=config.shaping)
    for note in memory_notes or []:
        window.note(0, "memory", note)
    meter = Meter(config.budget)
    ctx = LoopContext(run_id, message, handle, case_id, config, window, meter, workflow, variant)
    run = LoopRun(run_id, message, handle, getattr(policy, "name", type(policy).__name__),
                  StopReason.ERROR, None, config=_config_dict(config), workflow=workflow,
                  variant=variant)

    with TRACER.span("loop", trace_id=run_id, policy=run.policy, workflow=workflow) as span:
        try:
            await _drive(run, ctx, policy)
        except Exception as exc:  # the harness records a crash; it does not hide one
            run.stop_reason = StopReason.ERROR
            run.detail = f"{type(exc).__name__}: {exc}"[:200]
        run.meter = meter.snapshot()
        run.window = window.stats()
        run.window_obj = window
        span.set(stop_reason=run.stop_reason.value, **run.meter, **run.window)

    if config.record:
        _record(run)
    return run


async def _drive(run: LoopRun, ctx: LoopContext, policy: LoopPolicy) -> None:
    config, window, meter = ctx.config, ctx.window, ctx.meter

    # Some requests are decided before a single token is spent.
    if config.human_handoff:
        reason = needs_human(ctx.message, money_action_ready=False)
        if reason:
            run.stop_reason, run.detail = StopReason.NEEDS_HUMAN, reason
            run.handoff = {"reason": reason}
            return

    while True:
        stop = meter.exhausted()
        if stop:
            run.stop_reason, run.detail = stop, f"stopped before pass {meter.iterations + 1}"
            return

        view = window.render()
        decision, usage = await policy.decide(view, ctx)
        meter.iterations += 1
        meter.charge(
            prompt_tokens=int(usage.get("prompt_tokens") or estimate_tokens(view)),
            output_tokens=int(usage.get("output_tokens")
                              or estimate_tokens(decision.model_dump_json())),
            cost_inr=usage.get("cost_inr"),
        )
        meter.record_action(decision.action, decision.args)
        step = {
            "iteration": meter.iterations,
            "action": decision.action,
            "args": _redact_args(decision.args),
            "thought": redact(decision.thought or "")[0][:160],
            "context_tokens": estimate_tokens(view),
        }
        run.steps.append(step)

        if decision.action == FINAL:
            if await _finish(run, ctx, decision, step):
                return
            continue

        if decision.action == HANDOFF:
            if config.human_handoff:
                reason = str(decision.args.get("reason") or "the agent asked for a person")[:200]
                run.stop_reason, run.detail = StopReason.NEEDS_HUMAN, reason
                run.handoff = {"reason": reason}
                return
            window.note(meter.iterations, "kernel", "handoff is not available in this configuration")
            step["outcome"] = "refused"
            continue

        if decision.action == PROPOSE_REVERSAL:
            if await _propose(run, ctx, decision, step):
                return
            continue

        if decision.action == CONSULT_POLICY:
            await _consult(ctx, decision, step)
            continue

        await _call_tool(ctx, decision, step)


async def _finish(run: LoopRun, ctx: LoopContext, decision: Decision, step: dict) -> bool:
    answer = decision.answer or FinalAnswer(text="")
    if not answer.text.strip():
        ctx.window.note(ctx.meter.iterations, "kernel", "a final answer needs text")
        step["outcome"] = "empty_final"
        return False

    if not ctx.config.verifiers:
        run.stop_reason, run.answer, run.detail = StopReason.DONE, answer, "no in-loop verifiers configured"
        step["outcome"] = "done_unverified"
        return True

    verdicts = run_verifiers(answer, ctx.window)
    failed = [v for v in verdicts if not v.passed]
    run.verdicts.extend({"iteration": ctx.meter.iterations, "verifier": v.verifier,
                         "passed": v.passed, "feedback": v.feedback} for v in verdicts)
    if not failed:
        run.stop_reason, run.answer = StopReason.DONE, answer
        step["outcome"] = "verified"
        return True

    # A cheap verifier failed: the loop's reflection step. The feedback becomes
    # an observation, and the agent gets another pass — within a bounded budget.
    ctx.meter.verifier_retries += 1
    step["outcome"] = "verifier_failed:" + ",".join(v.verifier for v in failed)
    if ctx.meter.verifier_retries > ctx.config.budget.max_verifier_retries:
        run.stop_reason = StopReason.VERIFIER_EXHAUSTED
        run.detail = "; ".join(v.feedback for v in failed)[:200]
        return True
    ctx.window.note(ctx.meter.iterations, "verifier",
                    "Your answer was not sent. " + " ".join(v.feedback for v in failed))
    return False


async def _propose(run: LoopRun, ctx: LoopContext, decision: Decision, step: dict) -> bool:
    from ..actions import make_action, validate_action

    args = decision.args or {}
    action = make_action("reversal", case_id=ctx.case_id,
                         reason=str(args.get("reason") or "customer requested a reversal"),
                         reference=args.get("reference"), amount_inr=args.get("amount_inr"),
                         proposed_by=ctx.config.agent)
    validation = validate_action(action, author=ctx.handle)
    failed = [c.name for c in validation.failed()]
    step["outcome"] = "proposal_valid" if validation.ok else "proposal_blocked:" + ",".join(failed)

    if validation.ok and ctx.config.human_handoff:
        reason = needs_human(ctx.message, money_action_ready=True)
        run.stop_reason, run.detail = StopReason.NEEDS_HUMAN, reason or ""
        run.handoff = {"reason": reason, "proposal": action.model_dump(mode="json"),
                       "validation": validation.model_dump(mode="json")}
        return True

    if validation.ok:
        text = ("Reversal proposal passed every check, but this configuration has no human "
                "hand-over, so nothing can be approved. Tell the customer a person will review it.")
    else:
        text = "Reversal proposal refused: " + "; ".join(c.detail for c in validation.failed())
    ctx.window.note(ctx.meter.iterations, "validator", text,
                    facts=[(f"proposal:{action.reference}", text)])
    return False


async def _consult(ctx: LoopContext, decision: Decision, step: dict) -> None:
    if not ctx.config.subagents:
        ctx.window.note(ctx.meter.iterations, "kernel", "consult_policy is not available in this configuration")
        step["outcome"] = "refused"
        return
    from .subagent import consult_policy

    question = str((decision.args or {}).get("question") or ctx.message)
    result = await consult_policy(question)
    # The sub-agent's own usage is real spend: charge it to this loop.
    ctx.meter.charge(prompt_tokens=result.prompt_tokens, output_tokens=result.output_tokens,
                     cost_inr=result.cost_inr)
    ctx.window.note(ctx.meter.iterations, "subagent:policy", result.as_observation(),
                    facts=[(f"clause:{c}", f"[{c}] cited by the policy sub-agent")
                           for c in result.clause_ids])
    step["outcome"] = f"subagent_returned:{result.returned_tokens}t_of_{result.consumed_tokens}t"


async def _call_tool(ctx: LoopContext, decision: Decision, step: dict) -> None:
    from ..tools import REGISTRY

    tool, args = decision.action, dict(decision.args or {})
    if tool in KERNEL_ACTIONS or tool not in ctx.config.tools:
        ctx.window.note(ctx.meter.iterations, "kernel",
                        f"{tool!r} is not an available action. Available: "
                        f"{', '.join(ctx.config.tools) or 'none'}, final")
        step["outcome"] = "refused"
        return

    # Scope keys come from the system, never from the model.
    if tool in ("lookup_transaction", "reversal_status"):
        args["handle"] = ctx.handle
    if tool == "open_ticket":
        args.setdefault("case_id", ctx.case_id)
        args.setdefault("idempotency_key", f"{ctx.case_id}:loop-ticket")

    result = await REGISTRY.call(tool, args, agent=ctx.config.agent, trace_id=ctx.run_id)
    if result.ok:
        ctx.window.observe(ctx.meter.iterations, tool, output=result.output)
        step["outcome"] = "ok"
    else:
        ctx.window.observe(ctx.meter.iterations, tool,
                           error=f"{result.error.code.value}: {result.error.message}")
        step["outcome"] = f"tool_error:{result.error.code.value}"


def _redact_args(args: dict) -> dict:
    return {k: redact(v)[0] if isinstance(v, str) else v for k, v in (args or {}).items()}


def _record(run: LoopRun) -> None:
    path = LOOP_RUNS_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(run.record(), ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # the record must never be why a conversation fails
