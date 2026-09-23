"""Runtime around the graph: run, pause, resume, inspect.

Every case is a LangGraph thread persisted to SQLite. A case waiting on a
human reviewer or on a callback is not a process holding memory — it is a row.
That is what makes "the customer answers the callback two days later" an
ordinary path rather than an architectural problem.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from langgraph.types import Command

from .config import CHECKPOINT_PATH
from .graph import build_graph
from .limits import CASES
from .models import Complaint


def new_case_id() -> str:
    return f"case_{uuid.uuid4().hex[:10]}"


@asynccontextmanager
async def _session():
    """A checkpointer session, in WAL mode.

    Each call opens its own connection, and ingest now runs cases concurrently,
    so several of them write this file at once. SQLite's default rollback
    journal locks the whole database for a write, and a checkpoint that waits
    past the busy timeout raises "database is locked" — which the ingest path
    catches and turns into a delivery-log failure. A bounded-concurrency change
    would have been manufacturing dead letters under exactly the burst it was
    added to survive. WAL lets readers and one writer proceed together, and the
    busy timeout makes a contended write wait rather than fail.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as saver:
        await saver.conn.execute("PRAGMA journal_mode=WAL")
        await saver.conn.execute("PRAGMA busy_timeout=5000")
        yield build_graph(checkpointer=saver), saver


def _config(case_id: str) -> dict:
    return {"configurable": {"thread_id": case_id}}


def _interrupt_of(result: dict) -> Optional[dict]:
    """Pull the pending interrupt payload out of a run result, if any."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


async def run_case(complaint: Complaint, case_id: Optional[str] = None) -> dict:
    """Start a case. Returns the state, plus `pending` if it stopped at a gate.

    Bounded by the CASES pool. This is the one place every complaint enters the
    graph, so it is the only place a limit has to be applied to hold — and
    complaints arrive in bursts by their nature, because the thing people are
    complaining about is one outage.
    """
    case_id = case_id or new_case_id()
    async with CASES.slot(), _session() as (graph, _):
        result = await graph.ainvoke(
            {
                "case_id": case_id,
                "complaint": complaint.model_dump(mode="json"),
                "costs": [],
                "events": [],
                "revision_count": 0,
            },
            config=_config(case_id),
        )
    return {"case_id": case_id, "state": result, "pending": _interrupt_of(result)}


async def resume_case(case_id: str, payload: Any) -> dict:
    """Resume a paused case with a review decision or a voice outcome.

    Bounded by the same pool: resuming runs the rest of the graph, models and
    tools included, so it is the same work under a different name.
    """
    async with CASES.slot(), _session() as (graph, _):
        result = await graph.ainvoke(Command(resume=payload), config=_config(case_id))
    return {"case_id": case_id, "state": result, "pending": _interrupt_of(result)}


async def get_case(case_id: str) -> Optional[dict]:
    async with _session() as (graph, _):
        snapshot = await graph.aget_state(_config(case_id))
    if not snapshot or not snapshot.values:
        return None
    pending = None
    if snapshot.interrupts:
        pending = getattr(snapshot.interrupts[0], "value", snapshot.interrupts[0])
    return {
        "case_id": case_id,
        "state": snapshot.values,
        "pending": pending,
        "next": list(snapshot.next),
    }


async def list_cases() -> list[dict]:
    """Every case the checkpointer knows about, newest first."""
    async with _session() as (graph, saver):
        seen: dict[str, dict] = {}
        async for cp in saver.alist(None):
            tid = cp.config["configurable"]["thread_id"]
            if tid in seen:
                continue
            values = cp.checkpoint.get("channel_values", {}) or {}
            if not values.get("complaint"):
                continue
            seen[tid] = {
                "case_id": tid,
                "complaint": values.get("complaint"),
                "triage": values.get("triage"),
                "verdict": values.get("verdict"),
                "pattern": values.get("pattern"),
                "priority": values.get("priority"),
                "review": values.get("review"),
                "closure": values.get("closure"),
                "escalation": values.get("escalation"),
                "draft": values.get("draft"),
            }
    return list(seen.values())
