"""The tool registry: the one door between agents and the outside world.

Every tool call in Sanwaad goes through `ToolRegistry.call`, which applies the
same checks in the same order no matter which agent is asking:

    1. does the tool exist?
    2. is THIS agent allowed to call it?            (least privilege)
    3. do the arguments satisfy the input contract? (before the backend sees them)
    4. high-risk write? is there an approval bound to these exact arguments,
       from someone allowed to give it?
    5. run with a timeout; retry only if the error is retryable AND the tool
       is a read or idempotent
    6. does the result satisfy the output contract?
    7. write an audit record, with string arguments redacted

Putting this in one place is what makes "the model should not be the source of
truth for business rules" enforceable. An agent can propose anything; it can
only *do* what passes this function.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from ..agents import may_call
from ..config import DATA_DIR
from ..guardrails import redact
from ..obs import TRACER
from .breaker import CircuitBreaker
from .contracts import (
    Approval,
    ErrorCode,
    Risk,
    ToolError,
    ToolFailure,
    ToolResult,
    ToolSpec,
    args_digest,
)

AUDIT_PATH = DATA_DIR / "tool_audit.jsonl"


def _redact_args(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)[0]
    if isinstance(value, dict):
        return {k: _redact_args(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_args(v) for v in value]
    return value


def _summarise(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "input"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _approval_problem(spec: ToolSpec, inp: BaseModel, approval: Optional[Approval]) -> Optional[str]:
    if approval is None:
        return f"{spec.name} is a high-risk write and needs an approval"
    if approval.tool != spec.name:
        return "approval was issued for a different tool"
    if approval.args_digest != args_digest(inp):
        return "approval does not match these exact arguments"
    if not approval.human and not spec.auto_approvable:
        return f"{spec.name} needs a human approval; a policy gate cannot approve it"
    return None


class ToolRegistry:
    def __init__(self, audit_path: Optional[Path] = None,
                 breaker: Optional[CircuitBreaker] = None):
        self._tools: dict[str, ToolSpec] = {}
        self._faults: dict[str, list[ToolError]] = {}
        self.audit_path = audit_path
        self.breaker = breaker or CircuitBreaker()

    # --- registration -----------------------------------------------------

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    # --- fault injection, for evals ----------------------------------------

    def inject_fault(self, name: str, error: ToolError, times: int = 1) -> None:
        """Make the next `times` attempts at `name` fail with `error`.

        A system that has only ever been run against healthy backends has an
        untested failure path, and the untested path is the one production
        takes on its worst day.
        """
        self._faults.setdefault(name, []).extend([error] * times)

    def clear_faults(self) -> None:
        # The breakers too: a scenario that injected an outage leaves the
        # circuit it tripped behind it, and the next scenario would start
        # against a tool that is still failing fast for reasons of its own.
        self._faults.clear()
        self.breaker.reset()

    def _next_fault(self, name: str) -> Optional[ToolError]:
        queue = self._faults.get(name)
        return queue.pop(0) if queue else None

    # --- approvals ----------------------------------------------------------

    def approval_for(self, name: str, args: dict, *, by: str, human: bool,
                     note: str = "") -> Approval:
        """Issue an approval bound to exactly these arguments."""
        spec = self._tools.get(name)
        digest = args_digest(args)
        if spec is not None:
            try:
                digest = args_digest(spec.input_model.model_validate(args))
            except ValidationError:
                pass  # the call itself will reject the arguments
        return Approval(by=by, human=human, tool=name, args_digest=digest, note=note)

    # --- calling ------------------------------------------------------------

    async def call(self, name: str, args: dict, *, agent: str,
                   approval: Optional[Approval] = None,
                   trace_id: Optional[str] = None) -> ToolResult:
        started = time.perf_counter()
        spec = self._tools.get(name)
        with TRACER.span(f"tool.{name}", trace_id=trace_id, agent=agent) as span:
            result = await self._call(spec, name, args, agent=agent, approval=approval)
            result.ms = round((time.perf_counter() - started) * 1000, 2)
            result.audit_id = uuid.uuid4().hex[:10]
            span.set(ok=result.ok, attempts=result.attempts,
                     risk=spec.risk.value if spec else None,
                     circuit=self.breaker.state(name),
                     tool_error=result.error.code.value if result.error else None)
        self._audit(result, spec, agent, args, approval, trace_id)
        return result

    async def _call(self, spec: Optional[ToolSpec], name: str, args: dict, *,
                    agent: str, approval: Optional[Approval]) -> ToolResult:
        def refuse(code: ErrorCode, message: str) -> ToolResult:
            return ToolResult(tool=name, ok=False, attempts=0,
                              error=ToolError(code=code, message=message))

        if spec is None:
            return refuse(ErrorCode.NOT_FOUND, f"no tool named {name!r}")
        if not may_call(agent, name):
            return refuse(ErrorCode.NOT_PERMITTED, f"agent {agent!r} may not call {name!r}")
        try:
            inp = spec.input_model.model_validate(args)
        except ValidationError as exc:
            return refuse(ErrorCode.INVALID_INPUT, _summarise(exc))

        if spec.risk is Risk.WRITE_HIGH:
            problem = _approval_problem(spec, inp, approval)
            if problem:
                return refuse(ErrorCode.NEEDS_APPROVAL, problem)

        # Retrying is only safe when repeating the call cannot repeat the
        # effect. A timed-out "post reply" may well have posted.
        retry_safe = spec.risk is Risk.READ or spec.idempotent
        max_attempts = 1 + (spec.max_retries if retry_safe else 0)

        # Ask the breaker before spending anything. A tool that is already
        # failing does not need this call's retries to prove it again, and the
        # caller gets its answer in a millisecond instead of a timeout.
        if not self.breaker.allows(name):
            return refuse(ErrorCode.CIRCUIT_OPEN,
                          f"{name} is failing; not called (circuit open)")

        last: Optional[ToolError] = None
        attempt = 0
        reported = False
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    injected = self._next_fault(name)
                    if injected is not None:
                        raise ToolFailure(injected.code, injected.message, injected.retryable)
                    raw = await asyncio.wait_for(spec.handler(inp), timeout=spec.timeout_s)
                except (asyncio.TimeoutError, TimeoutError):
                    last = ToolError(code=ErrorCode.TIMEOUT, retryable=True,
                                     message=f"no response within {spec.timeout_s}s")
                except ToolFailure as exc:
                    last = ToolError(code=exc.code, message=exc.message, retryable=exc.retryable)
                except Exception as exc:
                    # Never forward a raw exception message to an agent: it can
                    # carry internals, and it is not something to branch on.
                    last = ToolError(code=ErrorCode.UPSTREAM, retryable=False,
                                     message=f"unexpected {type(exc).__name__}")
                else:
                    try:
                        payload = raw.model_dump() if isinstance(raw, BaseModel) else raw
                        out = spec.output_model.model_validate(payload)
                    except ValidationError as exc:
                        # The backend answered, just not in contract. That is our
                        # disagreement with it, not evidence that it is down.
                        self.breaker.record_outcome(name, ErrorCode.INVALID_OUTPUT)
                        reported = True
                        return ToolResult(tool=name, ok=False, attempts=attempt, error=ToolError(
                            code=ErrorCode.INVALID_OUTPUT, message=_summarise(exc)))
                    self.breaker.record_outcome(name)
                    reported = True
                    return ToolResult(tool=name, ok=True, attempts=attempt,
                                      output=out.model_dump(mode="json"))
                # Counted per attempt, not per call: three retries against a dead
                # backend are three dead requests, and the breaker should see the
                # damage at the rate it is actually being done.
                self.breaker.record_outcome(name, last.code)
                reported = True
                if not last.retryable:
                    break
                # Re-ask before spending another attempt. That failure may have
                # re-opened the circuit — and if this call was the half-open probe,
                # its remaining retries would make "exactly one probe" three
                # requests to a backend we just confirmed is down.
                if not self.breaker.allows(name):
                    break
            return ToolResult(tool=name, ok=False, attempts=attempt, error=last)
        finally:
            # A cancelled task unwinds through here without an outcome —
            # CancelledError is a BaseException, so none of the handlers above
            # see it. If this call held the half-open probe, leaving the flag
            # set would refuse every later call to this tool for the life of
            # the process. Hand the probe back: we learned nothing either way.
            if not reported:
                self.breaker.abandon(name)

    # --- audit ----------------------------------------------------------------

    def _audit(self, result: ToolResult, spec: Optional[ToolSpec], agent: str,
               args: Any, approval: Optional[Approval], trace_id: Optional[str]) -> None:
        record = {
            "at": datetime.now(timezone.utc).isoformat(),
            "audit_id": result.audit_id,
            "trace_id": trace_id,
            "agent": agent,
            "tool": result.tool,
            "risk": spec.risk.value if spec else None,
            "args": _redact_args(args),
            "ok": result.ok,
            "error_code": result.error.code.value if result.error else None,
            "error": result.error.message if result.error else None,
            "attempts": result.attempts,
            "ms": result.ms,
            "approved_by": approval.by if approval else None,
            "human_approved": approval.human if approval else None,
        }
        path = self.audit_path or AUDIT_PATH
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # auditing must never be the reason a call fails


REGISTRY = ToolRegistry()
