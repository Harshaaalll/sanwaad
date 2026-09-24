"""Agents handing work to each other, without handing over control.

The fixed pipeline in `graph/graph.py` is right for one workflow whose shape is
known in advance. It is the wrong shape for "a lead arrives, whoever qualifies
it decides who sees it next" — there the route depends on what the first agent
found, and writing every branch as an edge means editing the graph to add a
step.

The obvious fix is to let agents call each other. That is also how multi-agent
systems become impossible to reason about: control flow lives in prompts, two
agents pass work back and forth until a budget dies, and the reason a lead
ended up somewhere exists only in a transcript.

So this module keeps the property that makes the rest of the system
defensible, and gives up only the rigidity. An agent **proposes** a handoff.
Code validates it against a declared routing table and either grants or
refuses it. The model never transfers control; it requests a transfer. That is
the same shape as the tool layer — propose, validate, execute — applied to
routing instead of to actions, and it buys the same thing: the model can be
wrong without the system being unsafe.

## What a refusal protects against

    undeclared      the target is not somewhere this agent may send work
    unknown         the target does not exist at all
    missing data    the target needs a field this payload does not carry
    a loop          this agent is already in the trail
    too far         the chain has run longer than the mission allows

Each refusal ends the run with a reason a person can read, rather than
becoming an exception in a log. A mission that cannot finish is a fact about
the mission, and the trail says exactly where it stopped.

## Declarations are checked, not trusted

`Mission.check()` runs at declaration time: every target exists, every
non-terminal step can hand work somewhere, every step is reachable from the
entry, and some terminal step is reachable. A routing table with a typo or an
orphaned step fails when the module is imported rather than when a lead hits
that branch at two in the morning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Union

from loguru import logger

from .obs import TRACER


class MissionError(ValueError):
    """A routing table that does not describe a workable mission."""


@dataclass(frozen=True)
class Step:
    """One agent's place in a mission: what it needs, and where it may send work."""

    agent: str
    does: str
    # Payload keys that must be present on entry. The same idea as a tool's
    # input contract: a step that needs a phone number should refuse the work
    # rather than discover the gap halfway through its own prompt.
    expects: tuple[str, ...] = ()
    hands_to: tuple[str, ...] = ()
    # Tools this step may reach, which is what caps how much of it can run
    # unattended — see `autonomy.ceiling_for`.
    tools: tuple[str, ...] = ()
    terminal: bool = False


@dataclass(frozen=True)
class Mission:
    name: str
    entry: str
    steps: tuple[Step, ...]
    # A chain longer than this is a mission that is not converging. Eight is
    # generous for a pipeline whose longest honest path is four.
    max_hops: int = 8

    def step(self, agent: str) -> Optional[Step]:
        return next((s for s in self.steps if s.agent == agent), None)

    def check(self) -> "Mission":
        """Fail at declaration time rather than at two in the morning."""
        names = [s.agent for s in self.steps]
        if len(names) != len(set(names)):
            raise MissionError(f"{self.name}: two steps share a name")
        if self.entry not in names:
            raise MissionError(f"{self.name}: entry {self.entry!r} is not a step")

        for s in self.steps:
            unknown = [t for t in s.hands_to if t not in names]
            if unknown:
                raise MissionError(f"{self.name}: {s.agent} hands to unknown {unknown}")
            if not s.terminal and not s.hands_to:
                raise MissionError(
                    f"{self.name}: {s.agent} is not terminal and hands to nobody, "
                    f"so work that reaches it can only stop there")

        reachable, frontier = {self.entry}, [self.entry]
        while frontier:
            current = self.step(frontier.pop())
            for target in current.hands_to if current else ():
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        orphans = sorted(set(names) - reachable)
        if orphans:
            raise MissionError(f"{self.name}: {orphans} cannot be reached from {self.entry}")
        if not any(self.step(a).terminal for a in reachable):
            raise MissionError(f"{self.name}: no terminal step is reachable; nothing can finish")
        return self


@dataclass
class Handoff:
    """An agent's request to pass the work on. A request, not an instruction."""

    to: str
    reason: str
    payload: dict = field(default_factory=dict)


@dataclass
class Done:
    """An agent's decision that the mission is finished."""

    outcome: str
    payload: dict = field(default_factory=dict)


@dataclass
class Hop:
    frm: str
    to: str
    reason: str
    hop: int


@dataclass
class MissionRun:
    mission: str
    trail: list[Hop] = field(default_factory=list)
    visited: list[str] = field(default_factory=list)
    outcome: str = ""
    stopped: str = ""
    detail: str = ""
    payload: dict = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.stopped == "done"

    def path(self) -> str:
        return " → ".join(self.visited)


Result = Union[Handoff, Done]
Handler = Callable[[dict], Awaitable[Result]]


def refusal_for(mission: Mission, frm: str, handoff: Handoff,
                visited: list[str]) -> Optional[str]:
    """Why this handoff may not happen, or None if it may.

    Every branch here is a way a self-routing system eats itself, and each one
    returns a sentence rather than raising, because "the mission stopped and
    here is why" is an outcome a person can act on.
    """
    target = mission.step(handoff.to)
    if target is None:
        return f"{handoff.to!r} is not a step in {mission.name}"

    source = mission.step(frm)
    if source is not None and handoff.to not in source.hands_to:
        return (f"{frm} may not hand to {handoff.to}; "
                f"it may hand to {list(source.hands_to) or 'nobody'}")

    missing = [k for k in target.expects if k not in handoff.payload]
    if missing:
        return f"{handoff.to} needs {missing}, which this handoff does not carry"

    if handoff.to in visited:
        return (f"{handoff.to} has already seen this work "
                f"({' → '.join(visited)}); a mission may not revisit a step")

    if len(visited) >= mission.max_hops:
        return f"{mission.name} ran {len(visited)} hops, its limit"

    return None


async def run_mission(mission: Mission, handlers: dict[str, Handler], payload: dict,
                      *, trace_id: Optional[str] = None) -> MissionRun:
    """Drive one piece of work through a mission, one validated hop at a time.

    The handlers decide *what* to propose. This decides what actually happens,
    which is the whole point: an agent that proposes a route it may not take
    stops the mission with a reason, rather than taking it.
    """
    run = MissionRun(mission=mission.name, payload=dict(payload))
    current = mission.entry

    with TRACER.span(f"mission.{mission.name}", trace_id=trace_id) as span:
        while True:
            step = mission.step(current)
            if step is None:                       # unreachable: check() forbids it
                run.stopped, run.detail = "error", f"{current!r} is not a step"
                break

            missing = [k for k in step.expects if k not in run.payload]
            if missing:
                run.stopped = "refused"
                run.detail = f"{current} needs {missing}, which the work does not carry"
                break

            run.visited.append(current)
            handler = handlers.get(current)
            if handler is None:
                run.stopped, run.detail = "error", f"no handler for {current}"
                break

            with TRACER.span(f"mission.{mission.name}.{current}", trace_id=trace_id):
                result = await handler(run.payload)

            if isinstance(result, Done):
                run.payload.update(result.payload)
                run.outcome, run.stopped = result.outcome, "done"
                run.detail = f"{current} finished the mission"
                break

            if not isinstance(result, Handoff):
                run.stopped = "error"
                run.detail = f"{current} returned {type(result).__name__}, not a handoff"
                break

            merged = {**run.payload, **result.payload}
            proposed = Handoff(result.to, result.reason, merged)
            refusal = refusal_for(mission, current, proposed, run.visited)
            if refusal:
                run.stopped, run.detail = "refused", refusal
                logger.warning(f"[{mission.name}] handoff refused: {refusal}")
                break

            run.payload = merged
            run.trail.append(Hop(current, result.to, result.reason, len(run.trail) + 1))
            current = result.to

        span.set(path=run.path(), stopped=run.stopped, hops=len(run.trail),
                 outcome=run.outcome)

    logger.info(f"[{mission.name}] {run.path()} — {run.stopped}: {run.detail}")
    return run
