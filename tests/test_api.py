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
    monkeypatch.setattr(server, "_READY", {"index": "ready", "clauses": 33, "error": None})
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
    assert body["clauses"] == 33          # every clause in sanwaad/policy
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
