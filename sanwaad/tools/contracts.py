"""Tool contracts: a tool is an API, and gets designed like one.

A tool is the boundary between a model and the outside world. The model is
never allowed to send arbitrary instructions across it. Every tool therefore
declares, up front:

  name + description   what it does, in words a model can choose by
  input model          exact fields, types and bounds — validated BEFORE the
                       backend sees anything
  output model         machine-readable success, validated before an agent
                       sees it
  risk                 READ · WRITE_LOW · WRITE_HIGH, which decides what else
                       has to be true before it runs
  timeout, retries     and whether retrying is even safe (only for reads and
                       idempotent writes — retrying "post reply" posts twice)
  errors               one structured shape, so an agent branches on a code
                       instead of parsing a stack trace

The risk ladder is the design pattern: ship READ tools first, add low-risk
writes next, and add high-risk writes last and only behind validation plus an
approval that is bound to the exact arguments it approved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Type

from pydantic import BaseModel, Field


class Risk(str, Enum):
    READ = "read"               # fetches; changes nothing
    WRITE_LOW = "write_low"     # changes something recoverable and internal
    WRITE_HIGH = "write_high"   # public, financial or irreversible


class ErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"     # arguments failed the input contract
    NOT_PERMITTED = "not_permitted"     # this agent may not call this tool
    NEEDS_APPROVAL = "needs_approval"   # high-risk write without a matching approval
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"               # e.g. already reversed under another request
    TIMEOUT = "timeout"
    UPSTREAM = "upstream_error"
    INVALID_OUTPUT = "invalid_output"   # backend answered, but not in contract


class ToolError(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = False


class ToolFailure(Exception):
    """Raised by a handler to return a structured error instead of a crash."""

    def __init__(self, code: ErrorCode, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class Approval(BaseModel):
    """Permission for ONE call with ONE set of arguments.

    `args_digest` is the point. An approval that says "reversal approved"
    without binding the reference and amount can be replayed onto a different
    transaction — by a bug, or by a model that was talked into it.
    """

    by: str
    human: bool
    tool: str
    args_digest: str
    note: str = ""


class ToolResult(BaseModel):
    tool: str
    ok: bool
    output: Optional[dict] = None
    error: Optional[ToolError] = None
    attempts: int = 0
    ms: float = 0.0
    audit_id: str = ""


Handler = Callable[[Any], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    handler: Handler
    risk: Risk
    timeout_s: float = 5.0
    max_retries: int = 0
    idempotent: bool = False
    # May a deterministic policy gate approve this, or only a person? A public
    # reply can be auto-approved by the auto-post policy; moving money cannot.
    auto_approvable: bool = False

    def as_mcp_tool(self) -> dict:
        """The same contract, in Model Context Protocol shape.

        MCP standardises how tools are exposed to agents; it does not change
        what a good tool contract is. Everything a well-designed tool already
        declares maps straight across — which is the argument for designing
        the contract first and choosing the transport second.
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_model.model_json_schema(),
            "annotations": {
                "readOnlyHint": self.risk is Risk.READ,
                "destructiveHint": self.risk is Risk.WRITE_HIGH,
                "idempotentHint": self.idempotent,
            },
        }


def args_digest(args: Any) -> str:
    """Stable fingerprint of a call's arguments, for binding approvals."""
    data = args.model_dump(mode="json") if isinstance(args, BaseModel) else args
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "Approval", "ErrorCode", "Field", "Handler", "Risk", "ToolError", "ToolFailure",
    "ToolResult", "ToolSpec", "args_digest",
]
