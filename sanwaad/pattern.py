"""The pattern agent: watch the wave, not the drop.

Triage reads one complaint and answers "how bad is this for this person?". That
question has no answer that reaches "the payments rail has been down for
twenty minutes" — no single comment contains that fact. It only exists in the
relationship between comments, so something has to hold the last hour in mind
and keep asking a different question: *is this the ninth of these?*

Two things this deliberately does not do:

- **It does not count comments; it counts people.** One furious customer
  posting six times is one furious customer. Six people posting once is an
  incident. Collapsing to distinct authors before any threshold is checked is
  the difference between an early-warning system and a system that panics at
  whoever is loudest.
- **It does not ask a model.** Similarity is a dot product against embeddings
  we already computed for retrieval, and a threshold. A crisis detector that
  costs a token per comment is a crisis detector that gets sampled, and a
  sampled detector misses the first ten minutes — which is the only ten
  minutes that matter.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Iterable, Optional

from .config import CRISIS, DATA_DIR
from .guardrails import redact
from .models import PatternSignal

MEMORY_PATH = DATA_DIR / "pattern_memory.json"

_lock = threading.Lock()


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Fingerprint:
    """The minimum needed to recognise the same complaint arriving again."""

    case_id: str
    author: str
    category: str
    summary: str
    at: str
    vector: list[float] = field(default_factory=list)
    text: str = ""          # truncated raw text, for the coordination check

    @classmethod
    def from_dict(cls, d: dict) -> "Fingerprint":
        return cls(
            case_id=d.get("case_id", ""),
            author=d.get("author", ""),
            category=d.get("category", ""),
            summary=d.get("summary", ""),
            at=d.get("at", ""),
            vector=list(d.get("vector") or []),
            text=d.get("text", ""),
        )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class PatternStore:
    """A bounded, file-backed window of what has arrived recently.

    Deliberately not the LangGraph checkpointer: that is per-case state, and
    this is the one thing in the system that is explicitly *cross*-case. It is
    also fine to lose — a restart costs us the current window, not any case.
    """

    def __init__(self, path=MEMORY_PATH, max_history: int = CRISIS.max_history):
        self.path = path
        self.max_history = max_history

    def load(self) -> list[Fingerprint]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt window is not worth failing a live case over; the worst
            # outcome is that this comment looks like the first of its kind.
            return []
        return [Fingerprint.from_dict(d) for d in raw]

    def save(self, prints: list[Fingerprint]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trimmed = prints[-self.max_history:]
        self.path.write_text(
            json.dumps([asdict(f) for f in trimmed], ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, fp: Fingerprint) -> list[Fingerprint]:
        with _lock:
            prints = [p for p in self.load() if p.case_id != fp.case_id]
            prints.append(fp)
            self.save(prints)
        return prints

    def author_history(self, author: str, exclude_case_id: str = "") -> int:
        """How many earlier cases this account has raised with us.

        Read over the whole retained window rather than the crisis window: a
        customer who complained last week is still a customer today, which is
        the opposite of what the cluster check wants to know.

        `exclude_case_id` is not optional in practice. The pattern agent writes
        the current case's fingerprint before the judge runs, so a naive count
        finds the case asking the question and reports every first-time poster
        as a returning customer — which silently promoted an hour-old throwaway
        to "customer" in the first end-to-end run.
        """
        if not author:
            return 0
        return sum(1 for fp in self.load()
                   if fp.author == author and fp.case_id != exclude_case_id)

    def recent(self, within_minutes: int = CRISIS.window_minutes,
               now: Optional[datetime] = None) -> list[Fingerprint]:
        now = now or _now()
        floor = now - timedelta(minutes=within_minutes)
        out = []
        for fp in self.load():
            try:
                if _parse(fp.at) >= floor:
                    out.append(fp)
            except ValueError:
                continue
        return out


_store: Optional[PatternStore] = None


def get_store() -> PatternStore:
    global _store
    if _store is None:
        _store = PatternStore()
    return _store


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Both sides come out of `embed_one`, which L2-normalises, so this is a
    plain dot product. Guarded anyway — an empty vector must score 0, not
    raise, because a fingerprint written before an embedding failure should
    degrade to 'not similar' rather than take a live case down."""
    av, bv = list(a), list(b)
    if not av or not bv or len(av) != len(bv):
        return 0.0
    return float(sum(x * y for x, y in zip(av, bv)))


def assess_cluster(
    *,
    vector: list[float],
    author: str,
    at: datetime,
    neighbours: list[Fingerprint],
    similarity: float = CRISIS.similarity,
    watch_cluster: int = CRISIS.watch_cluster,
    crisis_cluster: int = CRISIS.crisis_cluster,
    crisis_velocity: float = CRISIS.crisis_velocity_per_hour,
) -> PatternSignal:
    """The whole decision, as a pure function over vectors and timestamps.

    Kept free of I/O and embedding so the thresholds can be tested directly.
    Every branch here is a product argument someone will want to have.
    """
    similar = [fp for fp in neighbours if cosine(vector, fp.vector) >= similarity]

    # Distinct people, not distinct posts. See the module docstring.
    authors = {fp.author for fp in similar if fp.author}
    authors.add(author)
    cluster_size = len(authors)

    times = []
    for fp in similar:
        try:
            times.append(_parse(fp.at))
        except ValueError:
            continue
    times.append(at)
    span_minutes = max((at - min(times)).total_seconds() / 60.0, 1.0)

    # Rate is computed over the span the cluster actually occupies, not the
    # configured window. Six complaints in four minutes and six spread over
    # ninety are not the same event, and dividing by the window would hide it.
    #
    # A cluster of one has no rate. Dividing by the one-minute floor would
    # report the first complaint of the day as "60 per hour", which is both
    # meaningless and, on a console, alarming.
    velocity = cluster_size / (span_minutes / 60.0) if cluster_size > 1 else 0.0

    if cluster_size >= crisis_cluster or (
        cluster_size >= watch_cluster and velocity >= crisis_velocity
    ):
        level = "crisis"
    elif cluster_size >= watch_cluster:
        level = "watch"
    else:
        level = "none"

    return PatternSignal(
        level=level,
        cluster_size=cluster_size,
        window_minutes=int(span_minutes),
        velocity_per_hour=round(velocity, 2),
        theme=_theme(similar),
        related_case_ids=[fp.case_id for fp in similar][:20],
    )


def _theme(similar: list[Fingerprint]) -> str:
    if not similar:
        return ""
    counts: dict[str, int] = {}
    for fp in similar:
        counts[fp.category] = counts.get(fp.category, 0) + 1
    category = max(counts, key=counts.get)
    earliest = min(similar, key=lambda f: f.at)
    return f"{category}: {earliest.summary[:120]}"


def detect(
    *,
    case_id: str,
    author: str,
    category: str,
    summary: str,
    text: str = "",
    at: Optional[datetime] = None,
    store: Optional[PatternStore] = None,
) -> tuple[PatternSignal, int]:
    """Score this complaint against the recent window, then join it.

    Returns the pattern signal and the number of *other* authors whose wording
    is near-identical — the coordination count the judge uses. Both come out of
    the same pass over the window, because two passes over the same list is
    two chances for them to disagree.
    """
    from .rag.embedder import embed_one

    # `at` is when the comment was POSTED, not when we got to it. Using the
    # wall clock collapses a replayed feed into one instant and reports a
    # velocity of hundreds per hour for complaints that arrived over two days.
    # A backlog being worked through is not an incident.
    at = at or _now()
    store = store or get_store()
    neighbours = [fp for fp in store.recent(now=at) if fp.case_id != case_id]

    vector = list(embed_one(summary or text))
    signal = assess_cluster(vector=vector, author=author, at=at, neighbours=neighbours)

    # The window is cross-case memory that outlives the case, so it holds the
    # redacted text only. Redaction is deterministic, so two copy-pasted
    # comments still match each other after it.
    clean = redact(text or "")[0]
    duplicates = coordinated_authors(clean or summary, author, neighbours)

    store.add(Fingerprint(
        case_id=case_id, author=author, category=category, summary=summary,
        at=at.isoformat(), vector=[round(v, 5) for v in vector], text=clean[:300],
    ))
    return signal, duplicates


# ---------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------

_NORMALISE = re.compile(r"[^\w\s]+", re.UNICODE)


def _normalise(text: str) -> str:
    return _NORMALISE.sub(" ", text.lower()).strip()


def coordinated_authors(text: str, author: str, neighbours: list[Fingerprint],
                        threshold: float = 0.85) -> int:
    """How many *other* accounts posted near-identical wording recently.

    Lexical rather than semantic on purpose. Semantic similarity is what makes
    nine differently-worded outage reports one incident; here we want the
    opposite instrument — one that only fires on copy-paste, because that is
    what a campaign looks like and what an organic wave never does.
    """
    if not text:
        return 0
    norm = _normalise(text)
    if not norm:
        return 0
    seen = set()
    for fp in neighbours:
        if not fp.text or fp.author == author or fp.author in seen:
            continue
        if SequenceMatcher(None, norm, _normalise(fp.text)).ratio() >= threshold:
            seen.add(fp.author)
    return len(seen)
