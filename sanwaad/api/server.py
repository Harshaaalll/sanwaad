"""FastAPI surface: ingestion, the review queue, and WebRTC signalling.

Runs on port 7870 by default (SANWAAD_PORT), leaving the common 7860 free for
other local services during a demo.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sanwaad.autonomy import LEDGER as AUTONOMY  # noqa: E402
from sanwaad.connectors import get_connector  # noqa: E402
from sanwaad.delivery import DEAD, DELIVERY  # noqa: E402
from sanwaad.limits import CALLS, CASES  # noqa: E402
from sanwaad.limits import report as pool_report  # noqa: E402
from sanwaad.models import Citation, Complaint  # noqa: E402
from sanwaad.pipeline import (  # noqa: E402
    get_case,
    list_cases,
    resume_case,
    run_case,
)
from sanwaad.rag.store import get_store  # noqa: E402
from sanwaad.tools import REGISTRY  # noqa: E402

_BOOTED_AT = time.time()

# What readiness actually depends on. The policy index is the only hard one:
# without it there is no retrieval, and an ungrounded reply is precisely what
# this system exists to prevent.
_READY: dict[str, Any] = {"index": "cold", "clauses": 0, "error": None}


async def _warm_index() -> None:
    """Build or load the policy index, off the event loop.

    First run downloads the ONNX embedding model (~470MB) and takes about a
    minute; later starts are quick because the index is cached to disk. This
    warms in the background rather than blocking startup, so during that minute
    the process answers /health while /ready says it is still warming — an
    orchestrator can then tell "coming up" from "broken", which is the whole
    reason to have two probes. `get_store` holds a lock, so a request arriving
    mid-warm waits for this same build instead of starting a second one.
    """
    try:
        store = await asyncio.to_thread(get_store)
        _READY.update(index="ready", clauses=len(store.clauses), error=None)
        logger.info(f"Policy index ready: {len(store.clauses)} clauses")
    except Exception as exc:
        _READY.update(index="failed", error=f"{type(exc).__name__}: {exc}")
        logger.exception("policy index failed to build")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    warm = asyncio.create_task(_warm_index())
    try:
        yield
    finally:
        warm.cancel()


app = FastAPI(title="Sanwaad", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Live WebRTC calls, keyed by case. A case can only be on one call at a time.
_calls: dict[str, Any] = {}
# Strong references to the tasks driving them, because asyncio does not keep any.
_call_tasks: set = set()


async def _close_quietly(connection: Any) -> None:
    """Let go of a peer connection without letting the attempt raise.

    A leaked connection holds a port and an ICE agent for the life of the
    process, and this runs on the path where something already went wrong.
    """
    for method in ("disconnect", "close"):
        closer = getattr(connection, method, None)
        if closer is None:
            continue
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
            return
        except Exception:
            logger.warning(f"could not {method}() a webrtc connection", exc_info=True)
            return


# ---------------------------------------------------------------------------
# Probes
#
# Two endpoints, because they answer different questions and a deployment
# needs both. /health asks "is this process alive", and must stay cheap and
# dependency-free — a liveness probe that touches a database restarts the
# container every time the database hiccups. /ready asks "should traffic come
# here yet", and is allowed to say no.
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": app.version,
            "uptime_s": round(time.time() - _BOOTED_AT, 1)}


@app.get("/ready")
async def ready():
    """503 until the policy index is loaded, and after it has failed.

    Returning 200 while retrieval is unavailable would let a load balancer send
    real complaints to a process that can only answer ungrounded.
    """
    state = {
        "ready": _READY["index"] == "ready",
        "index": _READY["index"],
        "clauses": _READY["clauses"],
        "uptime_s": round(time.time() - _BOOTED_AT, 1),
        # Not a dependency: without a key the models degrade to offline stubs
        # by design, so it is reported, never a reason to fail the probe.
        "models": "live" if os.getenv("GOOGLE_API_KEY") else "offline",
    }
    if _READY["error"]:
        state["error"] = _READY["error"]

    # Also reported, also not a reason to fail: an open circuit means a tool is
    # down and the system is deliberately degrading around it, which is working
    # as designed. Refusing traffic here would turn one dead dependency into a
    # dead service, which is the thing the breaker exists to avoid.
    tripped = [c for c in REGISTRY.breaker.report() if c["state"] != "closed"]
    if tripped:
        state["circuits"] = tripped

    # A saturated pool and an idle one look identical from outside until
    # something reports the queue. By the time the only signal is latency, it
    # is too late to act on.
    state["concurrency"] = pool_report()
    # What the system may currently do without anyone: the number an operator
    # actually wants when they ask "is this thing running itself yet".
    earned = [r for r in AUTONOMY.report() if r["level"] in ("SUPERVISED", "AUTONOMOUS")]
    state["autonomous_capabilities"] = earned
    return JSONResponse(state, status_code=200 if state["ready"] else 503)


@app.get("/api/delivery")
async def api_delivery():
    """The dead-letter queue, for the console.

    Excerpts were redacted when they were written, so this is safe to render.
    """
    return {"dead": [asdict(f) for f in DELIVERY.dead()],
            "retrying": [asdict(f) for f in DELIVERY.retrying()]}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    channel: str = "mock"
    limit: int = 8


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    """Pull inbound items and run each through the graph until it needs a human."""
    connector = get_connector(req.channel)
    complaints = await connector.fetch(limit=req.limit)

    async def _one(complaint: Complaint) -> dict:
        # A known-dead item is not retried here either. Without this, clicking
        # Ingest against a poisoned feed runs the whole graph on it again every
        # time and drives its attempt count past the limit that declared it
        # dead in the first place.
        if DELIVERY.is_dead(complaint):
            return {"external_id": complaint.external_id, "status": DEAD,
                    "error": "dead-lettered; requeue it first"}
        try:
            out = await run_case(complaint)
            # The listener clears a past failure on success and this did not,
            # so an item that failed once through the API stayed in the queue
            # forever, and its next two failures anywhere dead-lettered it after
            # what an operator experiences as two attempts, not three.
            DELIVERY.clear(complaint)
            return {
                "case_id": out["case_id"],
                "author": complaint.author,
                "pending": bool(out["pending"]),
                "triage": out["state"].get("triage"),
            }
        except Exception as exc:  # one bad item must not stall the batch
            # Recorded in the same place the listener records its failures, so
            # there is one queue to read rather than one per entry point.
            failure = DELIVERY.record_failure(complaint, exc)
            logger.exception(f"case failed for {complaint.external_id}")
            return {"external_id": complaint.external_id, "error": str(exc),
                    "attempts": failure.attempts, "status": failure.status}

    # Concurrent, but bounded: `run_case` holds a slot in the CASES pool, so a
    # batch of fifty runs four at a time rather than fifty at once or one after
    # another. The bound lives at the choke point, not here, so every other
    # caller gets it too.
    results = list(await asyncio.gather(*(_one(c) for c in complaints)))
    return {"ingested": len(results), "cases": results,
            "concurrency": CASES.report()}


# ---------------------------------------------------------------------------
# Cases and review
# ---------------------------------------------------------------------------

@app.get("/api/cases")
async def api_cases():
    return {"cases": await list_cases()}


@app.get("/api/cases/{case_id}")
async def api_case(case_id: str):
    case = await get_case(case_id)
    if not case:
        raise HTTPException(404, "no such case")
    return case


class ReviewRequest(BaseModel):
    decision: str          # approve | edit | reject
    final_text: Optional[str] = None
    reviewer: str = "human"
    note: str = ""
    actions: dict[str, str] = {}   # action_id -> approve | reject


@app.post("/api/cases/{case_id}/review")
async def api_review(case_id: str, req: ReviewRequest):
    if req.decision not in ("approve", "edit", "reject"):
        raise HTTPException(400, "decision must be approve, edit or reject")
    bad = {k: v for k, v in req.actions.items() if v not in ("approve", "reject")}
    if bad:
        raise HTTPException(400, f"each action decision must be approve or reject: {bad}")
    out = await resume_case(case_id, req.model_dump())
    return {"case_id": case_id, "pending": out["pending"],
            "state": _thin(out["state"])}


class VoiceOutcomeRequest(BaseModel):
    happened: bool = True
    channel: str = "webrtc"
    duration_s: float = 0.0
    resolved: bool = False
    summary: str = ""
    citations_used: list[str] = []


@app.post("/api/cases/{case_id}/voice-outcome")
async def api_voice_outcome(case_id: str, req: VoiceOutcomeRequest):
    """Close the voice leg and let the graph finish."""
    out = await resume_case(case_id, req.model_dump())
    return {"case_id": case_id, "state": _thin(out["state"])}


def _thin(state: dict) -> dict:
    """Trim clause bodies out of API responses; the console fetches those separately."""
    thin = dict(state)
    if "citations" in thin:
        thin["citations"] = [
            {k: v for k, v in c.items() if k != "text"} for c in thin["citations"]
        ]
    return thin


# ---------------------------------------------------------------------------
# Policy search (useful on its own, and it makes retrieval debuggable)
# ---------------------------------------------------------------------------

@app.get("/api/policy/search")
async def api_policy_search(q: str, k: int = 5):
    return {"results": [c.model_dump() for c in get_store().search(q, k=k)]}


# ---------------------------------------------------------------------------
# WebRTC signalling
# ---------------------------------------------------------------------------

class OfferRequest(BaseModel):
    case_id: str
    sdp: str
    type: str


@app.post("/api/offer")
async def api_offer(req: OfferRequest):
    """Browser SDP offer -> answer, and start the voice pipeline for the case."""
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

    from sanwaad.voice.agent import run_voice_call

    case = await get_case(req.case_id)
    if not case:
        raise HTTPException(404, "no such case")
    state = case["state"]
    if not (case.get("pending") or {}).get("await") == "voice_call":
        raise HTTPException(409, "case is not waiting on a voice call")

    brief = case["pending"]["brief"]
    citations = [Citation(**c) for c in state.get("citations", [])]

    if req.case_id in _calls:
        # Overwriting would strand the live connection with no one holding it.
        raise HTTPException(409, "this case is already on a call")

    # The slot is taken HERE, before an SDP answer exists. The first version
    # checked capacity here and took the slot inside the background task, so
    # three offers arriving in one tick all passed the check, all got a valid
    # answer, and the third was refused after the browser had already negotiated
    # a session — a caller connected to nobody, which is the dead air a busy
    # signal exists to prevent. Refusing before there is anything to connect to
    # is the only ordering that keeps that promise.
    if not await CALLS.acquire_now():
        raise HTTPException(503, f"all {CALLS.limit} call slots are busy; try again shortly")

    try:
        connection = SmallWebRTCConnection(ice_servers=["stun:stun.l.google.com:19302"])
        await connection.initialize(sdp=req.sdp, type=req.type)
    except Exception:
        CALLS.release_slot()      # never hold a slot for a call that never began
        raise
    _calls[req.case_id] = connection

    async def _drive():
        try:
            outcome = await run_voice_call(
                connection,
                case_id=req.case_id,
                complaint=state["complaint"]["text"],
                public_reply=(state.get("review") or {}).get("final_text", ""),
                summary=brief["summary"],
                category=brief["category"],
                language=brief["language"],
                citations=citations,
            )
            await resume_case(req.case_id, outcome.model_dump(mode="json"))
        except Exception:
            logger.exception(f"voice call failed for {req.case_id}")
        finally:
            _calls.pop(req.case_id, None)
            CALLS.release_slot()
            await _close_quietly(connection)

    # Held in a set, not fire-and-forget: asyncio keeps only a weak reference to
    # a running task, so a bare create_task can be collected mid-call.
    task = asyncio.create_task(_drive())
    _call_tasks.add(task)
    task.add_done_callback(_call_tasks.discard)

    answer = connection.get_answer()
    return JSONResponse({"sdp": answer["sdp"], "type": answer["type"]})


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def console():
    return (_STATIC / "console.html").read_text(encoding="utf-8")


@app.get("/call/{case_id}", response_class=HTMLResponse)
async def call_page(case_id: str):
    html = (_STATIC / "call.html").read_text(encoding="utf-8")
    return html.replace("__CASE_ID__", case_id)


def main():
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SANWAAD_PORT", "7870")))


if __name__ == "__main__":
    main()
