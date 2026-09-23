"""Caching, in the three forms that matter for an LLM pipeline.

1. PROVIDER PREFIX CACHE. Providers cache a *prefix* of the prompt and charge
   a fraction to re-read it (Gemini ~10x cheaper, Muse Spark 1.3 50x). The
   catch is that it is a strict prefix: one changed byte near the front voids
   everything after it. So prompt assembly order is a cost decision, not a
   style one — stable content first, volatile content last. `build_prompt`
   below enforces that ordering rather than trusting each call site.

2. EXACT CACHE. The same text arriving twice. Common in this system: a viral
   complaint gets copy-pasted by fifty people, a retry re-submits the same
   comment. Keyed on a hash of (model, prompt) — a different model is a
   different answer, so it must be in the key.

3. SEMANTIC CACHE. "paise wapas nahi aaye" and "refund not received yet" are
   different strings and the same question. Embed the input, and reuse a prior
   answer when cosine similarity clears a threshold. Powerful and dangerous:
   too low a threshold and you serve a confidently wrong answer that never
   appears in your logs as an error. Used here ONLY for triage, where the
   output is a small fixed-shape label — never for drafting, where two similar
   complaints can still need different replies.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 1. Cache-friendly prompt assembly
# ---------------------------------------------------------------------------

def build_prompt(*, stable: list[str], volatile: list[str]) -> tuple[str, str]:
    """Return (cacheable_prefix, volatile_suffix).

    `stable` is everything identical across requests of this kind — the system
    instruction, the policy clauses, the few-shot examples. `volatile` is this
    request. Keep the caller honest by returning them separately, so nobody
    can accidentally interpolate a case id into the prefix and silently halve
    the cache hit rate.
    """
    return "\n\n".join(s for s in stable if s), "\n\n".join(v for v in volatile if v)


def prefix_fingerprint(prefix: str) -> str:
    """Identifies a cacheable prefix. Log it: if this changes between requests
    you expected to share a cache, that is the bug."""
    return hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 2 + 3. Response caches
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    value: Any
    created: float
    hits: int = 0
    vector: Optional[np.ndarray] = None
    source_text: str = ""
    model: str = ""
    namespace: str = ""


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.exact_hits + self.semantic_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if not self.total else (self.exact_hits + self.semantic_hits) / self.total

    def as_dict(self) -> dict:
        return {"exact_hits": self.exact_hits, "semantic_hits": self.semantic_hits,
                "misses": self.misses, "hit_rate": round(self.hit_rate, 3)}


class ResponseCache:
    """Exact cache, with an optional semantic layer behind it.

    Order matters: exact first (free, no false positives), semantic only on a
    miss (costs an embedding, can be wrong). TTL exists because policy changes
    — a cached answer citing a superseded clause is worse than no cache.
    """

    def __init__(self, *, ttl_s: float = 3600, semantic: bool = False,
                 threshold: float = 0.93, max_entries: int = 2000):
        self.ttl_s = ttl_s
        self.semantic = semantic
        self.threshold = threshold
        self.max_entries = max_entries
        self._store: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    @staticmethod
    def key(model: str, text: str, *, namespace: str = "") -> str:
        h = hashlib.sha256(f"{namespace}\x00{model}\x00{text}".encode("utf-8"))
        return h.hexdigest()[:24]

    def _fresh(self, e: CacheEntry) -> bool:
        return (time.time() - e.created) < self.ttl_s

    def get(self, model: str, text: str, *, namespace: str = "") -> Optional[Any]:
        k = self.key(model, text, namespace=namespace)
        e = self._store.get(k)
        if e and self._fresh(e):
            e.hits += 1
            self.stats.exact_hits += 1
            return e.value

        if self.semantic:
            hit = self._semantic_get(model, text, namespace)
            if hit is not None:
                self.stats.semantic_hits += 1
                return hit

        self.stats.misses += 1
        return None

    def _semantic_get(self, model: str, text: str, namespace: str) -> Optional[Any]:
        from .rag.embedder import embed_one

        # Scope the neighbour search to the same model AND namespace. Without
        # this the semantic layer silently answers a flash request from a
        # flash-lite entry — a different model is a different answer.
        live = [(k, e) for k, e in self._store.items()
                if e.vector is not None and self._fresh(e)
                and e.model == model and e.namespace == namespace]
        if not live:
            return None
        q = np.array(embed_one(text), dtype=np.float32)
        mat = np.stack([e.vector for _, e in live])
        sims = mat @ q
        i = int(np.argmax(sims))
        if float(sims[i]) >= self.threshold:
            entry = live[i][1]
            entry.hits += 1
            return entry.value
        return None

    def put(self, model: str, text: str, value: Any, *, namespace: str = "") -> None:
        if len(self._store) >= self.max_entries:
            # Evict the oldest. An LRU would be better under real traffic;
            # this is enough while the working set is a day of complaints.
            oldest = min(self._store.items(), key=lambda kv: kv[1].created)[0]
            del self._store[oldest]

        vec = None
        if self.semantic:
            from .rag.embedder import embed_one

            vec = np.array(embed_one(text), dtype=np.float32)
        self._store[self.key(model, text, namespace=namespace)] = CacheEntry(
            value=value, created=time.time(), vector=vec, source_text=text,
            model=model, namespace=namespace)

    def __len__(self) -> int:
        return len(self._store)


# Triage is the only safe place for the semantic layer: fixed-shape output,
# highest call volume, and a near-duplicate complaint really does have the
# same category and severity.
TRIAGE_CACHE = ResponseCache(ttl_s=6 * 3600, semantic=True, threshold=0.94)
