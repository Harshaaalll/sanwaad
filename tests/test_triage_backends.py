"""Triage backends and the comparison that chooses between them.

Laya is never downloaded here. A fake stands in for its router, returning the
payload shape the real `laya` package produces, so these pin the parts that are
ours: how a decision model's answers become triage fields, when live triage
declines a decision model and falls back to Gemini, and whether the comparison
numbers are the numbers a person would compute by hand.
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import triage_backends as tb
from sanwaad.caching import TRIAGE_CACHE
from sanwaad.evals import triage_compare as tc
from sanwaad.graph.nodes import triage_node


def _payload(category="refund", cat_conf=0.9, severity_score=2.0,
             sentiment="frustrated", is_complaint=0.95, private=0.8) -> dict:
    """The shape `laya.Router.predict` returns, trimmed to what we read."""
    probs = {c: 0.0 for c in tb.CATEGORY_CRITERIA}
    probs[category] = cat_conf
    return {
        "answers": {
            "category": {"type": "choice", "choice": category, "probabilities": probs,
                         "confidence": 0.5, "answer_confidence": cat_conf},
            "severity": {"type": "score", "score": severity_score},
            "sentiment": {"type": "choice", "choice": sentiment},
            "is_complaint": {"type": "noul", "noul": is_complaint},
            "needs_private_data": {"type": "noul", "noul": private},
        },
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "routing": {"model": "multilingual"},
    }


class FakeLaya(tb.LayaBackend):
    def __init__(self, payload=None, error=None):
        super().__init__()
        self.payload, self.error, self.seen = payload, error, []

    def available(self):
        return True, "fake"

    def _predict(self, text):
        self.seen.append(text)
        if self.error:
            raise self.error
        return self.payload


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    TRIAGE_CACHE._store.clear()
    monkeypatch.setattr(tb, "_INSTANCES", {})
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield
    TRIAGE_CACHE._store.clear()


def _case(text: str) -> dict:
    return {"case_id": "case_test", "costs": [],
            "complaint": {"text": text, "channel": "reddit", "author": "u/x"}}


# --- mapping ----------------------------------------------------------------

def test_decision_answers_map_onto_triage_fields():
    labels = tb.labels_from_decision(_payload(severity_score=2.4), backend="laya",
                                     model="laya-multilingual", latency_ms=12.0)
    assert labels.category == "refund"
    # Score is the 0-based expected level: 2.4 rounds to level 2, which is severity 3.
    assert labels.severity == 3
    assert labels.is_complaint and labels.needs_private_data
    # Gated on the calibrated number, not the entropy-based `confidence`.
    assert labels.confidence == 0.9
    assert labels.cost["inr"] == 0.0 and labels.cost["prompt_tokens"] == 42


@pytest.mark.parametrize("score,expected", [(-0.3, 1), (0.0, 1), (3.6, 5), (9.0, 5)])
def test_severity_is_clamped_to_the_rubric(score, expected):
    labels = tb.labels_from_decision(_payload(severity_score=score), backend="laya",
                                     model="m", latency_ms=0)
    assert labels.severity == expected


def test_questions_cover_every_category_the_pipeline_knows():
    """A category missing from the criteria is one the decision model can never answer."""
    from sanwaad.models import Category

    assert set(tb.decision_questions()["category"]["criteria"]) == {c.value for c in Category}
    assert len(tb.decision_questions()["severity"]["criteria"]) == 5


def test_jev_reports_why_it_cannot_run(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    ok, why = tb.get_backend("jev").available()
    assert not ok and "TYPESAFE_API_KEY" in why
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    ok, why = tb.get_backend("jev").available()
    assert not ok and "not written" in why


def test_unknown_live_backend_falls_back_to_gemini(monkeypatch):
    monkeypatch.setenv("SANWAAD_TRIAGE_BACKEND", "gpt-9")
    assert tb.live_backend_name() == "gemini"


# --- live triage --------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_triage_uses_the_decision_model_when_it_is_sure(monkeypatch):
    fake = FakeLaya(_payload(category="account_access", cat_conf=0.88, severity_score=3.0))
    monkeypatch.setitem(tb._INSTANCES, "laya", fake)
    monkeypatch.setenv("SANWAAD_TRIAGE_BACKEND", "laya")

    out = await triage_node(_case("My wallet is frozen with ₹18,000 inside"))

    assert out["triage"]["category"] == "account_access"
    assert out["triage"]["severity"] == 4
    # Hybrid: the summary still exists, written by the (offline) triage tier.
    assert out["triage"]["summary"]
    assert out["events"][0]["backend"] == "laya"
    assert "laya-multilingual" in out["events"][0]["message"]
    assert fake.seen, "the decision model was never asked"


@pytest.mark.asyncio
async def test_an_unsure_decision_model_hands_the_comment_to_gemini(monkeypatch):
    monkeypatch.setitem(tb._INSTANCES, "laya", FakeLaya(_payload(cat_conf=0.31)))
    monkeypatch.setenv("SANWAAD_TRIAGE_BACKEND", "laya")
    monkeypatch.setenv("SANWAAD_TRIAGE_MIN_CONFIDENCE", "0.6")

    out = await triage_node(_case("double debit ₹640 taken twice"))

    ev = out["events"][0]
    assert ev["backend"] == "gemini"
    assert "unsure" in ev["message"] and "0.31" in ev["message"]


@pytest.mark.asyncio
async def test_a_broken_decision_model_does_not_stop_the_case(monkeypatch):
    monkeypatch.setitem(tb._INSTANCES, "laya", FakeLaya(error=RuntimeError("weights corrupt")))
    monkeypatch.setenv("SANWAAD_TRIAGE_BACKEND", "laya")

    out = await triage_node(_case("double debit ₹640 taken twice"))

    assert out["triage"]["category"] == "refund"     # the keyword stub, via the gemini path
    assert out["events"][0]["backend"] == "gemini"
    assert "failed (RuntimeError)" in out["events"][0]["message"]


@pytest.mark.asyncio
async def test_the_regulatory_override_still_outranks_a_decision_model(monkeypatch):
    monkeypatch.setitem(tb._INSTANCES, "laya", FakeLaya(_payload(severity_score=0.0)))
    monkeypatch.setenv("SANWAAD_TRIAGE_BACKEND", "laya")

    out = await triage_node(_case("Refund pending 40 days, filing with the RBI ombudsman"))

    assert out["triage"]["severity"] == 5


# --- metrics --------------------------------------------------------------------

def test_classification_report_matches_hand_arithmetic():
    pairs = [("refund", "refund"), ("refund", "billing"), ("billing", "billing"),
             ("data_privacy", "refund")]
    rep = tc.classification_report(pairs)
    assert rep["accuracy"] == 0.5
    # refund: tp 1, predicted 2, support 2 -> P 0.5 R 0.5 F1 0.5
    assert rep["per_class"]["refund"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    # billing: tp 1, predicted 2, support 1 -> P 0.5 R 1.0 F1 0.6667
    assert rep["per_class"]["billing"]["f1"] == 0.6667
    # data_privacy never caught: F1 0, and it drags the macro average down.
    assert rep["per_class"]["data_privacy"]["recall"] == 0.0
    assert rep["macro_f1"] == round((0.5 + 0.6667 + 0.0) / 3, 4)
    assert rep["confusion"]["data_privacy"] == {"refund": 1}


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    perfect = [(0.75, True), (0.75, True), (0.75, True), (0.75, False)]
    assert tc.expected_calibration_error(perfect) == 0.0
    overconfident = [(0.95, False), (0.95, False), (0.95, True), (0.95, True)]
    assert tc.expected_calibration_error(overconfident) == 0.45


def test_selective_accuracy_reports_coverage_and_accuracy_per_threshold():
    scored = [(0.95, True), (0.85, True), (0.65, False), (0.4, False)]
    rows = {s["threshold"]: s for s in tc.selective_accuracy(scored, (0.5, 0.8, 0.99))}
    assert rows[0.5] == {"threshold": 0.5, "coverage": 0.75, "accuracy": 0.6667}
    assert rows[0.8] == {"threshold": 0.8, "coverage": 0.5, "accuracy": 1.0}
    assert rows[0.99]["coverage"] == 0.0 and rows[0.99]["accuracy"] is None


# --- dataset --------------------------------------------------------------------

def test_loader_normalises_labels_and_keeps_unlabelled_rows(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("Text,Category\nmoney taken twice,Refund\nwallet frozen,account-access\n"
                 "does it work in Pune?,\n,billing\n", encoding="utf-8")
    rows = tc.load_rows(f)
    assert [r["gold"] for r in rows] == ["refund", "account_access", None]   # empty text dropped


def test_loader_names_every_category_outside_the_taxonomy(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("text,category\na,refund\nb,chargeback\nc,chargeback\nd,kyc\n", encoding="utf-8")
    with pytest.raises(tc.DatasetError) as err:
        tc.load_rows(f)
    assert "chargeback (2)" in str(err.value) and "kyc (1)" in str(err.value)


def test_loader_refuses_a_file_without_text(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("comment,category\nx,refund\n", encoding="utf-8")
    with pytest.raises(tc.DatasetError, match="no `text` column"):
        tc.load_rows(f)


def test_the_shipped_template_loads():
    rows = tc.load_rows(Path(tc.__file__).parent / "data" / "triage_template.csv")
    assert len(rows) >= 10 and any(r["gold"] is None for r in rows)


# --- end to end -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compare_scores_backends_and_redacts_what_it_keeps(monkeypatch):
    class Always(FakeLaya):
        def _predict(self, text):
            return _payload(category="billing", cat_conf=0.7)

    monkeypatch.setitem(tb._INSTANCES, "laya", Always())
    rows = [
        {"id": "a", "text": "charged a fee, call me on 9876543210", "channel": "reddit",
         "gold": "billing"},
        {"id": "b", "text": "₹640 debited twice", "channel": "reddit", "gold": "refund"},
    ]
    result = await tc.compare(rows, ["gemini", "laya", "jev"], dataset="t.csv")

    by = {b["backend"]: b for b in result["backends"]}
    assert set(by) == {"gemini", "laya"}
    assert result["skipped"][0]["backend"] == "jev"
    assert by["laya"]["category"]["accuracy"] == 0.5
    assert by["laya"]["calibration"]["reports_confidence"]
    assert not by["gemini"]["calibration"]["reports_confidence"]
    assert result["agreement"][0]["n"] == 2
    blob = json.dumps(result["examples"])
    assert "9876543210" not in blob, "a phone number reached the saved results"


def test_the_api_serves_the_latest_comparison(monkeypatch, tmp_path):
    from sanwaad.api import server

    path = tmp_path / "triage_compare.json"
    monkeypatch.setattr(tc, "RESULTS_PATH", path)
    client = TestClient(server.app)

    body = client.get("/api/evals/triage").json()
    assert body["result"] is None
    assert set(body["status"]["backends"]) == {"gemini", "laya", "jev"}

    path.write_text(json.dumps({"dataset": {"name": "x"}, "backends": []}), encoding="utf-8")
    assert client.get("/api/evals/triage").json()["result"]["dataset"]["name"] == "x"
