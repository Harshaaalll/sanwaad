"""Explore any company: what its customers are complaining about in public.

Insight only. Nothing here drafts a reply: replying for a company needs that
company's own policy (and its consent), which is what onboarding is for. This
reads public complaints, triages them, groups them into themes, and measures
how the company itself answers, so a team lead or a prospect can see the
picture in a minute.

Sources: Google Play reviews (public, no key) and Reddit (when REDDIT_CLIENT_ID
and REDDIT_CLIENT_SECRET are set). Kept apart from the case pipeline on
purpose: nothing is written to the case database or the pattern memory that
crisis detection reads, so one company's complaints can never surface as
another's outage. Reviewer names and avatars are never stored, and text is
redacted before anything is saved.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from .config import CRISIS, DATA_DIR
from .guardrails import check_complaint, redact

EXPLORE_DIR = DATA_DIR / "explore"
PER_STAR = 100           # newest reviews fetched for each of 1, 2 and 3 stars
TREND_DAYS = 30
MIN_WORDS = 4          # shorter reviews are counted, not classified
_JOBS: dict[str, dict] = {}


def _report_path(app_id: str):
    return EXPLORE_DIR / f"{re.sub(r'[^A-Za-z0-9._-]', '_', app_id)}.json"


# --- finding the company ------------------------------------------------------------

def _featured_ids(query: str) -> list[str]:
    """App ids in the order the Play Store search page lists them.

    The scraper returns the top "featured" result without an app id; the page's
    own links carry it, and the first one is that featured app.
    """
    url = "https://play.google.com/store/search?" + urllib.parse.urlencode(
        {"q": query, "c": "apps", "hl": "en", "gl": "in"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    return list(dict.fromkeys(re.findall(r"/store/apps/details\?id=([\w.]+)", html)))


def _search(query: str, n: int = 5) -> list[dict]:
    from google_play_scraper import search

    hits = search(query, lang="en", country="in", n_hits=n)
    if any(not h.get("appId") for h in hits):
        known = {h["appId"] for h in hits if h.get("appId")}
        spare = [i for i in _featured_ids(query) if i not in known]
        for h in hits:
            if not h.get("appId") and spare:
                h["appId"] = spare.pop(0)
    return [{"app_id": h["appId"], "title": h.get("title"), "developer": h.get("developer"),
             "score": round(h["score"], 2) if h.get("score") else None, "installs": h.get("installs"),
             "icon": h.get("icon")} for h in hits if h.get("appId")]


async def find_apps(query: str) -> list[dict]:
    query = (query or "").strip()
    if len(query) < 2:
        return []
    return await asyncio.to_thread(_search, query)


# --- fetching -------------------------------------------------------------------------

def _play_details(app_id: str) -> dict:
    from google_play_scraper import app

    d = app(app_id, lang="en", country="in")
    return {"app_id": app_id, "title": d.get("title"), "developer": d.get("developer"), "icon": d.get("icon"),
            "score": round(d["score"], 2) if d.get("score") else None, "ratings": d.get("ratings"),
            "installs": d.get("installs"), "histogram": d.get("histogram")}


def _play_complaints(app_id: str, per_star: int = PER_STAR) -> list[dict]:
    """The newest 1-, 2- and 3-star reviews, reduced to what the report needs.

    No userName, no userImage: a reviewer's identity adds nothing to "what are
    people unhappy about", and personal data not kept cannot leak.
    """
    from google_play_scraper import Sort, reviews

    out = []
    for star in (1, 2, 3):
        rows, _ = reviews(app_id, lang="en", country="in", sort=Sort.NEWEST, count=per_star,
                          filter_score_with=star)
        for r in rows:
            if (r.get("content") or "").strip():
                out.append({"source": "playstore", "id": r["reviewId"], "stars": r["score"],
                            "text": r["content"], "at": _iso(r.get("at")), "replied_at": _iso(r.get("repliedAt")),
                            "app_version": r.get("appVersion")})
    return out


async def _reddit_posts(company: str, limit: int = 50) -> list[dict]:
    if not (os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")):
        return []
    from .connectors.reddit import RedditConnector

    conn = RedditConnector(subreddits=os.getenv("REDDIT_SUBREDDITS", "india"), query=f'"{company}"')
    try:
        found = await conn.fetch(limit=limit)
    finally:
        await conn.close()
    return [{"source": "reddit", "id": c.external_id, "stars": None, "text": c.text,
             "at": c.created_at or None, "replied_at": None, "app_version": None}
            for c in found]


def _iso(value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    return str(value)


# --- analysis --------------------------------------------------------------------------

def themes(texts: list[str], vectors: np.ndarray, threshold: float = CRISIS.similarity) -> list[dict]:
    """Greedy grouping of near-duplicate complaints, biggest first.

    The same cosine floor the pattern agent uses for "the same complaint", so a
    theme here means what a cluster means everywhere else in Sanwaad.
    """
    groups: list[list[int]] = []
    centroids: list[np.ndarray] = []
    for i, v in enumerate(vectors):
        best, score = -1, threshold
        for g, c in enumerate(centroids):
            s = float(np.dot(v, c) / (np.linalg.norm(c) or 1.0))
            if s >= score:
                best, score = g, s
        if best < 0:
            groups.append([i])
            centroids.append(v.copy())
        else:
            groups[best].append(i)
            centroids[best] = centroids[best] + v
    out = []
    for members in sorted(groups, key=len, reverse=True):
        if len(members) < 2:
            break
        centre = centroids[groups.index(members)]
        rep = max(members, key=lambda i: float(np.dot(vectors[i], centre)))
        out.append({"size": len(members), "example": texts[rep][:220],
                    "quotes": [texts[i][:160] for i in members[:3] if i != rep][:2]})
    return out[:8]


def summarise(items: list[dict], triaged: list[dict], now: Optional[datetime] = None) -> dict:
    """The report's numbers, from already-redacted items and their triage."""
    now = now or datetime.now(timezone.utc)
    cats = Counter(t["category"] for t in triaged)
    sev = Counter(str(t["severity"]) for t in triaged)
    severe = sorted(range(len(items)), key=lambda i: (-triaged[i]["severity"], items[i]["at"] or ""))[:6]

    days = [(now - timedelta(days=d)).date() for d in range(TREND_DAYS - 1, -1, -1)]
    index = {d: i for i, d in enumerate(days)}
    per_day, replied_per_day = [0] * TREND_DAYS, [0] * TREND_DAYS
    replies, waits = 0, []
    for item in items:
        at = datetime.fromisoformat(item["at"]) if item.get("at") else None
        if at and at.date() in index:
            per_day[index[at.date()]] += 1
        if item.get("replied_at"):
            replies += 1
            rep = datetime.fromisoformat(item["replied_at"])
            if rep.date() in index:
                replied_per_day[index[rep.date()]] += 1
            if at and rep >= at:
                waits.append((rep - at).total_seconds() / 3600)
    play = [i for i in items if i["source"] == "playstore"]
    return {
        "analysed": len(items),
        "by_source": dict(Counter(i["source"] for i in items)),
        "by_category": dict(cats.most_common()),
        "by_severity": dict(sorted(sev.items())),
        "severe_share": round(sum(1 for t in triaged if t["severity"] >= 4) / len(triaged), 4) if triaged else None,
        "company_replies": {
            "replied": replies, "of": len(play),
            "rate": round(replies / len(play), 4) if play else None,
            "median_hours": round(statistics.median(waits), 1) if waits else None,
        },
        "trend": {"days": [d.isoformat() for d in days], "complaints": per_day, "replied": replied_per_day},
        "severe_examples": [{"text": items[i]["text"][:280], "stars": items[i]["stars"], "source": items[i]["source"],
                             "category": triaged[i]["category"], "severity": triaged[i]["severity"],
                             "at": items[i]["at"]} for i in severe],
    }


async def analyse(app_id: str, *, per_star: int = PER_STAR, progress=None) -> dict:
    """Fetch, triage and summarise one company. Writes the report and returns it."""
    from .graph.nodes import llm_triage
    from .llm import is_offline
    from .rag.embedder import embed

    def step(stage, done=0, total=0):
        if progress:
            progress(stage, done, total)

    step("Reading the Play Store listing")
    details = await asyncio.to_thread(_play_details, app_id)
    company = details["title"] or app_id
    step("Fetching recent 1–3 star reviews")
    items = await asyncio.to_thread(_play_complaints, app_id, per_star)
    step("Searching Reddit")
    items += await _reddit_posts(company)
    for item in items:
        # Same input guardrail as live triage, then redaction before storage.
        item["text"] = redact(check_complaint(item["text"])[0])[0]
    # "good", "ok", "bad app": a star rating with no detail says nothing about
    # what went wrong, and otherwise become the biggest "themes". Counted, not classified.
    short = sum(1 for i in items if len(i["text"].split()) < MIN_WORDS)
    items = [i for i in items if len(i["text"].split()) >= MIN_WORDS]

    gate, triaged, done = asyncio.Semaphore(4), [None] * len(items), 0

    async def one(i: int) -> None:
        nonlocal done
        async with gate:
            result, _ = await llm_triage(items[i]["text"], items[i]["source"], brand=company)
        triaged[i] = {"category": result.category.value, "severity": result.severity}
        done += 1
        step("Classifying complaints", done, len(items))

    await asyncio.gather(*(one(i) for i in range(len(items))))
    step("Grouping into themes")
    vectors = await asyncio.to_thread(embed, [i["text"] for i in items])
    report = {
        "app": details,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": "offline" if is_offline() else "live",
        "reddit": "included" if any(i["source"] == "reddit" for i in items)
                  else ("no matches" if os.getenv("REDDIT_CLIENT_ID") else "not configured"),
        "too_short": short,
        **summarise(items, triaged),
        "themes": themes([i["text"] for i in items], vectors),
    }
    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    _report_path(app_id).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


# --- jobs (one analysis per company at a time) -------------------------------------------

def start(app_id: str) -> dict:
    job = _JOBS.get(app_id)
    if job and job["state"] == "running":
        return job
    job = _JOBS[app_id] = {"app_id": app_id, "state": "running", "stage": "Starting", "done": 0, "total": 0}

    def progress(stage, done, total):
        job.update(stage=stage, done=done, total=total)

    async def run():
        try:
            await analyse(app_id, progress=progress)
            job.update(state="done", stage="Done")
        except Exception as exc:          # a scrape that breaks must say so, not hang
            job.update(state="failed", stage=f"{type(exc).__name__}: {exc}"[:300])

    job["_task"] = asyncio.get_running_loop().create_task(run())
    return job


def job_status(app_id: str) -> Optional[dict]:
    job = _JOBS.get(app_id)
    return {k: v for k, v in job.items() if not k.startswith("_")} if job else None


def list_reports() -> list[dict]:
    if not EXPLORE_DIR.exists():
        return []
    out = []
    for p in sorted(EXPLORE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"app_id": r["app"]["app_id"], "title": r["app"]["title"], "icon": r["app"].get("icon"),
                    "analysed": r["analysed"], "generated_at": r["generated_at"]})
    return out


def load_report(app_id: str) -> Optional[dict]:
    p = _report_path(app_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
