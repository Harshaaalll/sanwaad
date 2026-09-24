"""What a degraded run must not leave behind.

An outage that ends is supposed to stop mattering. These pin the places where
it did not: a stub answer cached and served for hours afterwards, a case
closing `resolved` with nothing sent, a call whose cost never reached the
rollup, and the most expensive model calls reported as free.

The common shape is worth naming, because it is the one an adversarial review
kept finding: the failure path works, and then writes something durable that
the recovery path never cleans up.
"""

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import llm
from sanwaad.caching import TRIAGE_CACHE
from sanwaad.graph.nodes import close_node, triage_node


class Tiny(BaseModel):
    label: str


@pytest.fixture(autouse=True)
def _clean_cache():
    TRIAGE_CACHE._store.clear()
    yield
    TRIAGE_CACHE._store.clear()


def _case(text: str, case_id: str) -> dict:
    return {"case_id": case_id, "costs": [],
            "complaint": {"text": text, "channel": "mock", "author": "u/x"}}


# --- an outage must not be cached -----------------------------------------

@pytest.mark.asyncio
async def test_a_degraded_triage_is_never_served_to_a_later_case(monkeypatch):
    """A thirty-second provider blip degrades one case to the keyword stub. If
    that stub is cached, every similar complaint gets it back as a clean cache
    hit with nothing marked degraded, and the outage goes on answering long
    after it ended."""
    monkeypatch.setattr(llm, "_get_client", lambda: object())

    async def down(*_a, **_kw):
        raise RuntimeError("provider 503")

    monkeypatch.setattr(llm, "_generate", down)
    text = "Aapke app se paise kat gaye lekin refund nahi mila"
    await triage_node(_case(text, "case_outage"))

    assert TRIAGE_CACHE._store == {}, "a degraded answer was written to the cache"


@pytest.mark.asyncio
async def test_a_real_answer_is_still_cached(monkeypatch):
    """The fix must not cost the cache its job: the same complaint arriving
    twice is the case it exists for."""
    monkeypatch.setattr(llm, "_get_client", lambda: object())
    calls = {"n": 0}

    async def ok(*_a, **_kw):
        calls["n"] += 1
        return object()

    monkeypatch.setattr(llm, "_generate", ok)
    monkeypatch.setattr(llm, "_parse", lambda _r, schema: schema.model_validate({
        "is_complaint": True, "category": "refund", "severity": 2,
        "sentiment": "frustrated", "language": "en-IN", "summary": "s",
        "entities": {}, "needs_private_data": False}))

    text = "refund not received for my order"
    await triage_node(_case(text, "case_a"))
    await triage_node(_case(text, "case_b"))
    assert calls["n"] == 1, "the second identical complaint should have hit the cache"


# --- resolved must mean something reached the customer ---------------------

@pytest.mark.asyncio
async def test_a_rejected_reply_does_not_close_resolved():
    """`resolved` was "no escalation needed", so a reply a reviewer refused
    closed as a success with nothing sent — the one field a person scanning a
    queue trusts."""
    out = await close_node({
        "case_id": "c1", "costs": [], "draft": {"citations": []},
        "review": {"decision": "reject", "final_text": ""},
        "published": {"blocked": True, "reasons": ["rejected"]},
        "escalation": {"needed": False}, "action_results": [],
    })
    closure = out["closure"]
    assert closure["resolved"] is False
    assert "rejected" in closure["resolution_reason"]


@pytest.mark.asyncio
async def test_a_case_whose_actions_all_failed_is_not_resolved():
    """A backend outage during `act` used to close resolved, silently dropping
    the follow-up the reply had just promised."""
    out = await close_node({
        "case_id": "c2", "costs": [], "draft": {"citations": []},
        "review": {"decision": "approve", "final_text": "we are on it"},
        "published": {"posted": True}, "escalation": {"needed": False},
        "action_results": [{"action_id": "a1", "status": "failed"}],
    })
    assert out["closure"]["resolved"] is False
    assert "a1" in out["closure"]["resolution_reason"]


@pytest.mark.asyncio
async def test_an_owed_callback_is_not_resolved():
    out = await close_node({
        "case_id": "c3", "costs": [], "draft": {"citations": []},
        "review": {"decision": "approve", "final_text": "we are on it"},
        "published": {"posted": True}, "escalation": {"needed": True},
        "voice": {"happened": False}, "action_results": [],
    })
    assert out["closure"]["resolved"] is False


@pytest.mark.asyncio
async def test_a_case_that_actually_worked_is_resolved():
    out = await close_node({
        "case_id": "c4", "costs": [], "draft": {"citations": []},
        "review": {"decision": "approve", "final_text": "we are on it"},
        "published": {"posted": True}, "escalation": {"needed": False},
        "action_results": [{"action_id": "a1", "status": "executed"}],
    })
    assert out["closure"]["resolved"] is True


# --- the expensive things must show their cost -----------------------------

@pytest.mark.asyncio
async def test_the_degraded_span_carries_what_it_spent(monkeypatch, tmp_path):
    """Every model tried, every retry spent, and the span said ₹0 — so `obs`
    reported the cheapest figure for the most expensive thing the layer does."""
    from sanwaad.obs import Tracer

    tracer = Tracer(path=tmp_path / "traces.jsonl")
    monkeypatch.setattr(llm, "TRACER", tracer)
    monkeypatch.setattr(llm, "_get_client", lambda: object())

    class _Usage:
        prompt_token_count, candidates_token_count = 4000, 800

    async def bad_shape(*_a, **_kw):
        return type("R", (), {"usage_metadata": _Usage(), "parsed": None, "text": "{}"})()

    monkeypatch.setattr(llm, "_generate", bad_shape)
    await llm.structured(model="gemini-2.5-flash", fallback_model="gemini-2.5-pro",
                         system="s", user="u", schema=Tiny, stage="triage",
                         offline_fallback={"label": "stub"})

    span = tracer.spans[-1]
    assert span.attrs["degraded"] is True
    assert span.attrs["cost_inr"] > 0, "the priciest call in the system reported as free"


@pytest.mark.asyncio
async def test_a_voice_call_puts_its_cost_in_the_case_total():
    """A four-minute callback bills for speech, synthesis and tokens. It was
    computed and then dropped on the floor, so cost-per-resolution was a
    text-only number wearing a whole-case label."""
    from sanwaad.graph.nodes import voice_node

    entries = [{"stage": "voice_stt", "model": "sarvam", "inr": 2.0, "usd": 0.02},
               {"stage": "voice_tts", "model": "murf", "inr": 2.5, "usd": 0.03}]

    def _resume(_payload):
        return {"happened": True, "channel": "webrtc", "duration_s": 240.0,
                "resolved": True, "citations_used": [], "costs": entries}

    import langgraph.types as lg

    original, lg.interrupt = lg.interrupt, _resume
    try:
        out = await voice_node({
            "case_id": "c5", "escalation": {"channel": "webrtc"},
            "triage": {"summary": "s", "language": "en-IN", "category": "refund"},
            "review": {"final_text": "we are on it"}, "citations": [],
            "action_results": [],
        })
    finally:
        lg.interrupt = original

    assert out["costs"] == entries
