"""Explore: any company's public complaints, insight only.

No network here: the Play Store, Reddit and the embedding model are faked.
What is pinned: the featured app's missing id is recovered; reviewer identity
is never stored and identifiers are redacted; the numbers are right; nothing
touches the case pipeline's data; and only team leads and up can run it.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import auth, explore, pipeline


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.setattr(explore, "EXPLORE_DIR", tmp_path / "explore")
    monkeypatch.setattr(explore, "_JOBS", {})


def test_the_featured_app_gets_its_id_back_from_the_search_page(monkeypatch):
    import google_play_scraper

    monkeypatch.setattr(google_play_scraper, "search", lambda *a, **k: [
        {"appId": None, "title": "Zomato: Food Delivery", "developer": "Zomato", "score": 4.56},
        {"appId": "com.zomato.delivery", "title": "Zomato Delivery Partner", "developer": "Zomato", "score": 4.6}])
    monkeypatch.setattr(explore, "_featured_ids", lambda q: ["com.application.zomato", "com.zomato.delivery"])
    hits = explore._search("Zomato")
    assert [h["app_id"] for h in hits] == ["com.application.zomato", "com.zomato.delivery"]
    assert hits[0]["score"] == 4.56


def test_featured_ids_are_read_in_page_order(monkeypatch):
    html = ('<a href="/store/apps/details?id=com.application.zomato">x</a>'
            '<a href="/store/apps/details?id=com.zomato.delivery">y</a>'
            '<a href="/store/apps/details?id=com.application.zomato">dup</a>').encode()

    class Page:
        def read(self):
            return html

    monkeypatch.setattr(explore.urllib.request, "urlopen", lambda *a, **k: Page())
    assert explore._featured_ids("zomato") == ["com.application.zomato", "com.zomato.delivery"]


REVIEWS = [
    {"source": "playstore", "id": "r1", "stars": 1, "text": "Refund not received for 5 days, call me on 9876543210 now",
     "at": "2026-10-04T10:00:00+00:00", "replied_at": "2026-10-04T12:00:00+00:00", "app_version": "1.0"},
    {"source": "playstore", "id": "r2", "stars": 1, "text": "Refund not received for a week, money stuck",
     "at": "2026-10-03T10:00:00+00:00", "replied_at": None, "app_version": "1.0"},
    {"source": "playstore", "id": "r3", "stars": 2, "text": "good", "at": "2026-10-03T11:00:00+00:00",
     "replied_at": None, "app_version": "1.0"},
    {"source": "playstore", "id": "r4", "stars": 3, "text": "Delivery partner was rude and late again",
     "at": "2026-10-02T10:00:00+00:00", "replied_at": "2026-10-02T14:00:00+00:00", "app_version": "1.0"},
]


def _fake_sources(monkeypatch):
    monkeypatch.setattr(explore, "_play_details", lambda app_id: {
        "app_id": app_id, "title": "FoodCo", "developer": "FoodCo Ltd", "icon": None, "score": 4.1,
        "ratings": 1000, "installs": "1,000,000+", "histogram": [100, 50, 50, 200, 600]})
    monkeypatch.setattr(explore, "_play_complaints", lambda app_id, per_star: [dict(r) for r in REVIEWS])

    def fake_embed(texts):
        # "refund" texts point one way, everything else another: two clear groups.
        return np.array([[1.0, 0.0] if "Refund" in t else [0.0, 1.0] for t in texts], dtype=np.float32)

    import sanwaad.rag.embedder as embedder
    monkeypatch.setattr(embedder, "embed", fake_embed)


def test_a_report_is_built_without_keeping_who_wrote_what(monkeypatch, tmp_path):
    _fake_sources(monkeypatch)
    monkeypatch.setattr(pipeline, "CHECKPOINT_PATH", tmp_path / "cases.sqlite")
    report = asyncio.run(explore.analyse("com.foodco.app"))

    assert report["analysed"] == 3 and report["too_short"] == 1          # "good" counted, not classified
    assert report["models"] == "offline" and report["reddit"] == "not configured"
    assert report["company_replies"] == {"replied": 2, "of": 3, "rate": 0.6667, "median_hours": 3.0}
    assert report["themes"][0]["size"] == 2 and "Refund" in report["themes"][0]["example"]
    saved = (explore.EXPLORE_DIR / "com.foodco.app.json").read_text()
    assert "9876543210" not in saved, "contact details are redacted before saving"
    assert "userName" not in saved and "userImage" not in saved
    assert json.loads(saved)["app"]["title"] == "FoodCo"
    assert not (tmp_path / "cases.sqlite").exists(), "exploring must never write the case database"


def test_another_companys_triage_never_shares_a_cache_entry(monkeypatch):
    from sanwaad.caching import TRIAGE_CACHE
    from sanwaad.graph import nodes

    seen = []

    async def model(**kw):
        seen.append(kw["system"])
        from sanwaad.models import Triage
        return Triage(is_complaint=True, category="refund", severity=3, sentiment="frustrated", summary="s"), \
            {"stage": "triage", "model": "gemini-test"}

    monkeypatch.setattr(nodes, "structured", model)
    TRIAGE_CACHE._store.clear()
    asyncio.run(nodes.llm_triage("refund pending", "playstore", brand="FoodCo"))
    asyncio.run(nodes.llm_triage("refund pending", "playstore"))
    assert "for FoodCo" in seen[0] and "NimbusPay" not in seen[0]
    assert "NimbusPay" in seen[1], "the same text for NimbusPay was not served FoodCo's cached answer"
    TRIAGE_CACHE._store.clear()


def test_only_team_leads_and_up_can_explore(monkeypatch):
    from sanwaad.api import server

    _fake_sources(monkeypatch)
    auth.add_user("lead@x.com", "Lee", "lead", "a-long-password")
    auth.add_user("agent@x.com", "Asha", "agent", "a-long-password")
    agent, lead = TestClient(server.app), TestClient(server.app)
    agent.post("/api/login", json={"email": "agent@x.com", "password": "a-long-password"})
    lead.post("/api/login", json={"email": "lead@x.com", "password": "a-long-password"})
    assert agent.get("/api/explore").status_code == 403
    assert TestClient(server.app).post("/api/explore/com.foodco.app").status_code == 401
    assert lead.post("/api/explore/not a valid id!").status_code in (400, 404)


def test_a_broken_play_store_page_reads_as_a_message(monkeypatch):
    from sanwaad.api import server

    def broken(q):
        raise ValueError("layout changed")

    monkeypatch.setattr(explore, "_search", broken)
    r = TestClient(server.app).get("/api/explore/search?q=swiggy")
    assert r.status_code == 502 and "couldn't search the Play Store" in r.json()["detail"]
