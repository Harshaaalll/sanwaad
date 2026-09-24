"""Tests for the probes a deployment actually depends on.

A health endpoint that always returns 200 is decoration. These pin the two
behaviours that make it worth having: liveness must not depend on anything
that can fail, and readiness must be willing to say no — because returning 200
while retrieval is unavailable hands real complaints to a process that can only
answer ungrounded, which is the one outcome this system exists to prevent.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad.api import server


@pytest.fixture
def client(monkeypatch):
    """A client with no lifespan, so the index state is ours to set.

    `TestClient` only runs startup inside a `with` block; constructing it
    plainly leaves `_READY` exactly as each test arranges it.
    """
    from sanwaad.rag.store import get_store

    monkeypatch.setattr(server, "_READY",
                        {"index": "ready", "clauses": len(get_store().clauses),
                         "error": None})
    return TestClient(server.app)


def test_health_does_not_depend_on_the_index(client, monkeypatch):
    """Liveness must stay up while a dependency is down. A liveness probe that
    fails on a broken index would have the orchestrator restart the container
    on a loop, which fixes nothing and loses the running cases."""
    monkeypatch.setattr(server, "_READY",
                        {"index": "failed", "clauses": 0, "error": "OSError: disk"})
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_refuses_traffic_while_the_index_is_still_warming(client, monkeypatch):
    monkeypatch.setattr(server, "_READY", {"index": "cold", "clauses": 0, "error": None})
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["ready"] is False
    assert r.json()["index"] == "cold"


def test_ready_reports_the_error_when_the_index_failed(client, monkeypatch):
    monkeypatch.setattr(server, "_READY",
                        {"index": "failed", "clauses": 0, "error": "OSError: disk"})
    r = client.get("/ready")
    assert r.status_code == 503
    assert "OSError" in r.json()["error"]


def test_ready_is_true_once_the_index_is_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["index"] == "ready"
    # Against the real index, not the number this test's own fixture planted.
    # `== 33` passed just as happily with a fixture saying 7, so deleting a
    # clause from sanwaad/policy could never have failed it.
    from sanwaad.rag.store import get_store

    assert body["clauses"] == len(get_store().clauses)
    assert "error" not in body            # nothing to report when nothing broke


def test_a_missing_model_key_is_reported_but_is_not_a_failure(client, monkeypatch):
    """Without a key the model layer degrades to offline stubs by design. That
    is a fact worth reporting on the probe and not a reason to refuse traffic;
    treating it as one would make the offline demo undeployable."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["models"] == "offline"


def test_an_open_circuit_is_reported_but_does_not_refuse_traffic(client):
    """A tool being down is the case the system is built to degrade around —
    it opens a ticket instead of inventing an answer. Failing readiness here
    would turn one dead dependency into a dead service, which is precisely what
    the breaker exists to prevent."""
    from sanwaad.tools import ErrorCode
    from sanwaad.tools.breaker import BreakerPolicy, CircuitBreaker

    breaker = CircuitBreaker(BreakerPolicy(threshold=1))
    breaker.record_outcome("lookup_transaction", ErrorCode.UPSTREAM)
    server.REGISTRY.breaker, saved = breaker, server.REGISTRY.breaker
    try:
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["circuits"][0]["tool"] == "lookup_transaction"
        assert r.json()["circuits"][0]["state"] == "open"
    finally:
        server.REGISTRY.breaker = saved


def test_ready_keeps_the_shape_the_console_health_strip_reads(client):
    """The console renders operational state straight off /ready. Renaming a
    field here would empty the strip silently — the page would still load, and
    the one thing it exists to show would just be missing."""
    body = client.get("/ready").json()
    assert {"ready", "index", "clauses", "models", "concurrency"} <= set(body)
    for pool in body["concurrency"]:
        assert {"pool", "limit", "running", "peak_running", "refused"} <= set(pool)


def test_the_dead_letter_endpoint_returns_both_queues(client):
    body = client.get("/api/delivery").json()
    assert set(body) == {"dead", "retrying"}
    assert isinstance(body["dead"], list)


def test_a_call_is_refused_before_an_sdp_exists_when_it_cannot_be_staffed(client, monkeypatch):
    """The failure this closes: /api/offer answered 200 with a valid SDP and
    then died in the background on a missing key, so the browser negotiated a
    session with nobody and showed "connection: failed" — which tells whoever
    is watching nothing about why. Same shape as the capacity bug, arriving by
    a different route."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("MURF_API_KEY", raising=False)

    r = client.post("/api/offer", json={"case_id": "case_x", "sdp": "v=0", "type": "offer"})
    assert r.status_code in (404, 503)
    if r.status_code == 503:
        assert "SARVAM_API_KEY" in r.json()["detail"]


def test_missing_voice_requirements_names_each_one(monkeypatch):
    from sanwaad.voice.agent import missing_requirements

    for var in ("SARVAM_API_KEY", "MURF_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    lacking = missing_requirements()
    assert {"SARVAM_API_KEY", "MURF_API_KEY", "GOOGLE_API_KEY"} <= set(lacking)

    # With the keys set, only packages can still be missing — and whether the
    # voice extras are installed depends on the environment, so asserting an
    # empty list here would make this test pass or fail on where it ran.
    monkeypatch.setenv("SARVAM_API_KEY", "x")
    monkeypatch.setenv("MURF_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert all("pip install" in item for item in missing_requirements())


@pytest.mark.asyncio
async def test_warming_records_a_failure_instead_of_crashing_startup(monkeypatch):
    """If the index cannot be built, the process must come up and say so. A
    crash at startup gives an operator a restart loop and no explanation."""
    state = {"index": "cold", "clauses": 0, "error": None}
    monkeypatch.setattr(server, "_READY", state)

    def boom():
        raise OSError("no space left on device")

    monkeypatch.setattr(server, "get_store", boom)
    await server._warm_index()

    assert state["index"] == "failed"
    assert "no space left" in state["error"]
