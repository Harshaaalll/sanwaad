"""Runtime around the graph: run, pause, resume, inspect.

Every case is a LangGraph thread persisted to SQLite. A case waiting on a
human reviewer or on a callback is not a process holding memory — it is a row.
That is what makes "the customer answers the callback two days later" an
ordinary path rather than an architectural problem.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import aiosqlite
from langgraph.types import Command

from .config import CHECKPOINT_PATH
from .graph import build_graph
from .limits import CASES
from .models import Complaint


def new_case_id() -> str:
    return f"case_{uuid.uuid4().hex[:10]}"


# One checkpointer per (event loop, database file), shared by every case.
_SAVERS: dict[tuple[int, str], Any] = {}
_SAVERS_LOCK: dict[int, asyncio.Lock] = {}


async def _shared_saver():
    """The process's one connection to the checkpoint database.

    This used to open a connection per call. Ingest runs cases concurrently, so
    that meant several connections writing one file, and WAL plus a busy
    timeout was supposed to make them queue. It did not, entirely: under WAL, a
    connection whose read snapshot is older than another connection's commit
    gets "database is locked" *immediately* when it tries to write — SQLite
    cannot wait its way out of a stale snapshot, so the busy timeout never
    applies. Four concurrent cases lost one in forty-five that way, each turned
    into a delivery failure a person had to requeue.

    One shared connection has no second writer to conflict with: the saver's
    own lock serialises its statements, which take milliseconds against a model
    call that takes seconds. The busy timeout stays for the other process that
    can open this file — the listener — which is the case it does cover.

    Keyed by event loop because the saver binds to the loop that made it, and
    by path so a test that points CHECKPOINT_PATH elsewhere gets its own.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    loop = asyncio.get_running_loop()
    key = (id(loop), str(CHECKPOINT_PATH))
    saver = _SAVERS.get(key)
    if saver is not None and saver.loop is loop:
        return saver
    lock = _SAVERS_LOCK.setdefault(id(loop), asyncio.Lock())
    async with lock:
        saver = _SAVERS.get(key)
        if saver is not None and saver.loop is loop:
            return saver
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = aiosqlite.connect(str(CHECKPOINT_PATH))
        # aiosqlite's worker thread is not a daemon, and a shared connection
        # outlives any one call, so every CLI that ran a case — the demo, the
        # evals, the listener — finished its work and then never exited,
        # waiting on a thread nobody would stop. The server closes it on
        # shutdown; a daemon thread covers everything else. Nothing is lost by
        # it: every write is awaited before a caller moves on, and WAL makes an
        # interrupted transaction roll back rather than corrupt the file.
        thread = getattr(conn, "_thread", None)
        if thread is not None:
            thread.daemon = True
        conn = await conn
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        saver = AsyncSqliteSaver(conn)
        _SAVERS[key] = saver
        return saver


async def close_sessions() -> None:
    """Close this loop's checkpoint connections. Called on server shutdown."""
    loop_id = id(asyncio.get_running_loop())
    for key in [k for k in _SAVERS if k[0] == loop_id]:
        saver = _SAVERS.pop(key)
        try:
            await saver.conn.close()
        except Exception:
            pass
    _SAVERS_LOCK.pop(loop_id, None)


@asynccontextmanager
async def _session():
    """A graph wired to the shared checkpointer."""
    saver = await _shared_saver()
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
                # For the overview: what a case has cost so far, open or not,
                # and when the pipeline first saw it.
                "cost_inr": round(sum(float(c.get("inr", 0.0))
                                      for c in values.get("costs") or []), 6),
                "opened_at": ((values.get("events") or [{}])[0]).get("at"),
            }
    return list(seen.values())
