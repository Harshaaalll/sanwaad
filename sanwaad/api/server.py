"""FastAPI surface: ingestion, the review queue, and WebRTC signalling.

Runs on port 7870 by default (SANWAAD_PORT), leaving the common 7860 free for
other local services during a demo.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
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

from sanwaad.connectors import get_connector          # noqa: E402
from sanwaad.models import Citation, Complaint        # noqa: E402
from sanwaad.pipeline import (                        # noqa: E402
    get_case,
    list_cases,
    resume_case,
    run_case,
)
from sanwaad.rag.store import get_store               # noqa: E402

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Build or load the policy index once, at boot. First run downloads the
    # ONNX embedding model (~470MB) and takes a minute; every later start is
    # instant because the index is cached to disk.
    store = get_store()
    logger.info(f"Policy index ready: {len(store.clauses)} clauses")
    yield


app = FastAPI(title="Sanwaad", version="0.1.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Live WebRTC calls, keyed by case. A case can only be on one call at a time.
_calls: dict[str, Any] = {}


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

    results = []
    for complaint in complaints:
        try:
            out = await run_case(complaint)
            results.append({
                "case_id": out["case_id"],
                "author": complaint.author,
                "pending": bool(out["pending"]),
                "triage": out["state"].get("triage"),
            })
        except Exception as exc:  # one bad item must not stall the batch
            logger.exception(f"case failed for {complaint.external_id}")
            results.append({"external_id": complaint.external_id, "error": str(exc)})

    return {"ingested": len(results), "cases": results}


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

    connection = SmallWebRTCConnection(ice_servers=["stun:stun.l.google.com:19302"])
    await connection.initialize(sdp=req.sdp, type=req.type)
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

    asyncio.create_task(_drive())

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
