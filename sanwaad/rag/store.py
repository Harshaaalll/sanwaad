"""Policy index: clause-level chunks with hybrid dense + lexical retrieval.

Chunking on clause markers rather than a fixed token window is deliberate.
A clause is the unit a compliance team actually signs off on, so it is the
unit a citation should point at. It also means a retrieved chunk is never a
half-sentence fragment that the drafting model has to guess the end of.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import INDEX_PATH, POLICY_DIR
from ..models import Citation
from .embedder import embed, embed_one


def _citable(c: dict) -> dict:
    """Clause fields the Citation model accepts — aliases are retrieval-only."""
    return {k: v for k, v in c.items() if k in ("clause_id", "doc", "heading", "text")}


_CLAUSE_RE = re.compile(r"^##\s*\[([A-Z]{2,4}-\d{2})\]\s*(.+)$", re.MULTILINE)


_ALIAS_RE = re.compile(r"^>\s*also:\s*(.+)$", re.MULTILINE)


def parse_clauses(path: Path) -> list[dict]:
    """Split a policy markdown file into `## [ID] Heading` clauses.

    A clause may carry a `> also: ...` line listing the words customers
    actually use for it. This is *document expansion*: the policy says
    "velocity hold", the customer says "everything is on hold after six
    payments", and no retriever bridges that on its own. Rather than hoping a
    bigger embedding model learns our internal vocabulary, we write the
    mapping down once, next to the clause, where a compliance reviewer can see
    and correct it. Aliases are indexed but never shown to the drafting model.
    """
    raw = path.read_text(encoding="utf-8")
    matches = list(_CLAUSE_RE.finditer(raw))
    clauses = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[m.end():end].strip()
        aliases = ""
        am = _ALIAS_RE.search(body)
        if am:
            aliases = am.group(1).strip()
            body = _ALIAS_RE.sub("", body).strip()
        clauses.append({
            "clause_id": m.group(1),
            "heading": m.group(2).strip(),
            "doc": path.stem,
            "text": body,
            "aliases": aliases,
        })
    return clauses


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9ऀ-ॿ]+", text.lower()))


class PolicyStore:
    def __init__(self, clauses: list[dict], vectors: Optional[np.ndarray] = None):
        self.clauses = clauses
        self.vectors = vectors if vectors is not None else np.zeros((0, 384), dtype=np.float32)
        self._token_sets = [
            _tokens(f"{c['heading']} {c['text']} {c.get('aliases', '')}") for c in clauses
        ]
        self._bm25 = None  # built lazily; only the fused path needs it

    # --- build / persist ---------------------------------------------------

    @classmethod
    def build(cls, policy_dir: Path = POLICY_DIR) -> "PolicyStore":
        clauses: list[dict] = []
        for path in sorted(policy_dir.glob("*.md")):
            clauses.extend(parse_clauses(path))
        if not clauses:
            raise ValueError(f"No clauses found under {policy_dir}")
        texts = [
            f"{c['heading']}. {c['text']} {c.get('aliases', '')}".strip()
            for c in clauses
        ]
        return cls(clauses, embed(texts))

    def save(self, path: Path = INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "clauses": self.clauses,
            "vectors": self.vectors.tolist(),
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "PolicyStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["clauses"], np.array(data["vectors"], dtype=np.float32))

    # --- retrieval ---------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        lexical_weight: float = 0.25,
        boost_prefixes: tuple[str, ...] = (),
        boost: float = 0.15,
    ) -> list[Citation]:
        """Hybrid retrieval with optional category boosting.

        Pure dense search misses exact identifiers customers quote verbatim
        ("T+3", "0.5%", "re-KYC"); pure lexical misses paraphrase, which is
        most of what arrives. Blending a Jaccard overlap into the dense score
        recovers both without the operational weight of a real BM25 index.

        `boost_prefixes` carries the triage category into retrieval. Triage
        has already decided this is a refund complaint with far more context
        than the retriever gets; refusing to use that signal and hoping cosine
        similarity rediscovers it is a waste of a classification we paid for.

        Note the query you pass matters more than any of these knobs: pass
        triage's English summary, not the raw comment. Latin-script Hinglish
        ("paise wapas nahi aaye") lands nowhere near English policy text in
        this embedding space — measured, not assumed. See CATEGORY_CLAUSES.
        """
        if not self.clauses:
            return []
        qv = np.array(embed_one(query), dtype=np.float32)
        dense = self.vectors @ qv

        q_tokens = _tokens(query)
        if q_tokens:
            lex = np.array([
                len(q_tokens & ts) / max(1, len(q_tokens)) for ts in self._token_sets
            ], dtype=np.float32)
        else:
            lex = np.zeros_like(dense)

        scores = (1 - lexical_weight) * dense + lexical_weight * lex
        if boost_prefixes:
            hits = np.array([
                c["clause_id"].startswith(boost_prefixes) for c in self.clauses
            ], dtype=np.float32)
            scores = scores + boost * hits
        top = np.argsort(-scores)[:k]
        return [
            Citation(
                clause_id=self.clauses[i]["clause_id"],
                doc=self.clauses[i]["doc"],
                heading=self.clauses[i]["heading"],
                text=self.clauses[i]["text"],
                score=float(scores[i]),
            )
            for i in top
        ]

    def _bm25_index(self):
        from .fusion import BM25

        if self._bm25 is None:
            self._bm25 = BM25([
                f"{c['heading']} {c['text']} {c.get('aliases', '')}" for c in self.clauses
            ])
        return self._bm25

    def search_fused(
        self,
        query: str,
        k: int = 5,
        boost_prefixes: tuple[str, ...] = (),
        weights: tuple[float, float] = (1.0, 1.0),
        candidates: int = 12,
    ) -> list[Citation]:
        """Hybrid retrieval by Reciprocal Rank Fusion of dense + BM25.

        Each retriever proposes its own top-`candidates` ordering and RRF
        merges the two by rank alone, so the embedding model's score scale
        stops being a tuning parameter. The category boost is applied after
        fusion, as a rank nudge rather than a score addition, for the same
        reason: it must not have to be recalibrated when a retriever changes.
        """
        from .fusion import rrf

        if not self.clauses:
            return []

        qv = np.array(embed_one(query), dtype=np.float32)
        dense_scores = self.vectors @ qv
        dense_rank = list(np.argsort(-dense_scores)[:candidates])
        lex_rank = self._bm25_index().rank(query, k=candidates)

        fused = rrf([dense_rank, lex_rank], weights=list(weights))

        if boost_prefixes:
            # Promote in-category clauses by a fixed fraction of one rank step,
            # which is meaningful against RRF scores of ~1/60 but cannot let a
            # wholly irrelevant clause outrank a strong double hit.
            bump = 1.0 / (60 + 1) * 0.5
            fused = sorted(
                ((i, sc + (bump if self.clauses[i]["clause_id"].startswith(boost_prefixes) else 0.0))
                 for i, sc in fused),
                key=lambda kv: -kv[1],
            )

        return [
            Citation(
                clause_id=self.clauses[i]["clause_id"],
                doc=self.clauses[i]["doc"],
                heading=self.clauses[i]["heading"],
                text=self.clauses[i]["text"],
                score=float(sc),
            )
            for i, sc in fused[:k]
        ]

    def get(self, clause_id: str) -> Optional[Citation]:
        for c in self.clauses:
            if c["clause_id"] == clause_id:
                return Citation(**_citable(c), score=1.0)
        return None

    def always_on(self, prefixes: tuple[str, ...] = ("BV-", "PRV-")) -> list[Citation]:
        """Clauses that apply to every reply regardless of the query.

        Brand voice and privacy rules are not retrieved — they are constitutional.
        Leaving them to semantic search means they drop out exactly when an
        unusual complaint makes them matter most.
        """
        return [
            Citation(**_citable(c), score=1.0)
            for c in self.clauses
            if c["clause_id"].startswith(prefixes)
        ]


_store: Optional[PolicyStore] = None
_store_lock = threading.Lock()


def get_store(rebuild: bool = False) -> PolicyStore:
    """The one policy index, built once.

    Double-checked locking because the server warms this on a worker thread at
    boot while requests may already be arriving: without the lock two callers
    can both miss the cache, build the index twice and race each other writing
    `policy_index.json`. The fast path stays a bare attribute read.
    """
    global _store
    if _store is not None and not rebuild:
        return _store
    with _store_lock:
        if _store is not None and not rebuild:
            return _store
        if INDEX_PATH.exists() and not rebuild:
            _store = PolicyStore.load()
        else:
            built = PolicyStore.build()
            built.save()
            _store = built
        return _store


# Which clause families a triage category should pull toward. Deterministic,
# auditable, and free — the kind of prior that is cheaper to encode than to
# make an embedding model learn.
CATEGORY_CLAUSES: dict[str, tuple[str, ...]] = {
    "billing": ("BIL-",),
    "refund": ("RFD-",),
    "service_outage": ("ESC-",),
    "delivery": ("RFD-",),
    "account_access": ("KYC-",),
    "agent_behaviour": ("BV-", "ESC-"),
    "data_privacy": ("PRV-", "KYC-"),
    "praise": ("BV-",),
    "off_topic": ("BV-",),
}
