"""Evaluation harness: run real cases in a sealed room.

An eval that shares state with the last run measures the last run. Sanwaad
keeps state in eight places (see memory.py), so a scenario that runs a
reversal would make the next scenario's identical reversal look "already
reversed" — a pass or a fail that has nothing to do with the code under test.

`isolated()` points every store at a fresh temporary directory, clears the
triage cache and any injected tool faults, and restores everything afterwards.
`run_to_completion()` then drives a case through every pause the graph makes,
playing a careful reviewer and a completed phone call.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional


@contextmanager
def isolated(root: Optional[Path] = None) -> Iterator[Path]:
    from .. import feedback, obs, pattern, pipeline
    from ..caching import TRIAGE_CACHE
    from ..connectors import get_connector
    from ..tools import REGISTRY
    from ..tools import ledger as ledger_mod
    from ..tools import registry as registry_mod

    tmp = tempfile.TemporaryDirectory(prefix="sanwaad-eval-") if root is None else None
    base = Path(tmp.name) if tmp else root
    base.mkdir(parents=True, exist_ok=True)

    mock = get_connector("mock")
    saved = {
        "checkpoint": pipeline.CHECKPOINT_PATH,
        "pattern_store": pattern._store,
        "audit": registry_mod.AUDIT_PATH,
        "backend": ledger_mod.BACKEND.path,
        "trace": obs.TRACER.path,
        "feedback": feedback.FEEDBACK_PATH,
        "replies": mock._replies_path,
        "cache": dict(TRIAGE_CACHE._store),
    }
    try:
        pipeline.CHECKPOINT_PATH = base / "checkpoints.sqlite"
        pattern._store = pattern.PatternStore(path=base / "pattern_memory.json")
        registry_mod.AUDIT_PATH = base / "tool_audit.jsonl"
        ledger_mod.BACKEND.path = base / "backend.json"
        obs.TRACER.path = base / "traces.jsonl"
        feedback.FEEDBACK_PATH = base / "feedback.jsonl"
        mock._replies_path = base / "replies.json"
        TRIAGE_CACHE._store.clear()
        REGISTRY.clear_faults()
        yield base
    finally:
        pipeline.CHECKPOINT_PATH = saved["checkpoint"]
        pattern._store = saved["pattern_store"]
        registry_mod.AUDIT_PATH = saved["audit"]
        ledger_mod.BACKEND.path = saved["backend"]
        obs.TRACER.path = saved["trace"]
        feedback.FEEDBACK_PATH = saved["feedback"]
        mock._replies_path = saved["replies"]
        TRIAGE_CACHE._store.clear()
        TRIAGE_CACHE._store.update(saved["cache"])
        REGISTRY.clear_faults()
        if tmp:
            tmp.cleanup()


async def run_to_completion(complaint, *, approve_actions: bool = True,
                            before_resume: Optional[Callable[[], None]] = None,
                            max_resumes: int = 6) -> dict:
    """Drive one case to the end. Returns the final run plus what was held.

    The simulated reviewer approves the reply and approves exactly the
    money actions whose plan-time validation passed — a careful human, not a
    rubber stamp. The simulated call reuses the reply's clauses, so the
    consistency receipt is testing the plumbing, not the phone call.
    """
    from ..pipeline import resume_case, run_case

    out = await run_case(complaint)
    held: Optional[str] = None
    called = False

    for _ in range(max_resumes):
        pending = out.get("pending")
        if not pending:
            break
        if pending.get("await") == "voice_call":
            called = True
            payload = {
                "happened": True, "channel": "webrtc", "duration_s": 120.0, "resolved": True,
                "summary": "simulated call in the trajectory eval",
                "citations_used": (out["state"].get("draft") or {}).get("citations", []),
            }
        else:
            held = held or pending.get("reason", "held")
            if before_resume is not None:
                before_resume()
                before_resume = None
            decisions = {
                a["proposal"]["id"]: "approve"
                for a in pending.get("actions", [])
                if approve_actions and a["proposal"]["risk"] == "write_high" and a["validation"]["ok"]
            }
            payload = {"decision": "approve", "reviewer": "eval-reviewer",
                       "note": "trajectory eval", "actions": decisions}
        out = await resume_case(out["case_id"], payload)

    return {"out": out, "held": held, "called": called}
