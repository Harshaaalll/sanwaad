"""Sanwaad's tool layer. Importing this package registers the built-in tools."""

from . import (
    builtin,  # noqa: F401  (registers lookup_transaction, open_ticket, post_reply, initiate_reversal)
)
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
from .registry import REGISTRY, ToolRegistry

__all__ = [
    "REGISTRY", "Approval", "ErrorCode", "Risk", "ToolError", "ToolFailure",
    "ToolRegistry", "ToolResult", "ToolSpec", "args_digest",
]
