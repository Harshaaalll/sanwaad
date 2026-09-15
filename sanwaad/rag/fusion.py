"""BM25 + Reciprocal Rank Fusion.

Why fuse instead of blending scores. The old `search()` added a dense cosine
score to a lexical overlap score. That is comparing two different currencies:
cosine lives in roughly 0.2-0.6 for this model, Jaccard overlap in 0.0-0.3, so
the weight was really just "how much do I mute the lexical signal", and it had
to be retuned whenever the embedding model changed.

RRF throws the scores away and keeps only the ranks:

    score(d) = sum over retrievers of  1 / (k + rank(d))

A document ranked 1st by either retriever gets 1/61; ranked 2nd, 1/62. The
constant k (60 by convention) flattens the curve so one retriever cannot
dominate on a single confident hit. Because only ordering matters, the two
retrievers never need to agree on scale — which is the entire point, and why
this survives swapping the embedding model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Optional

RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Split Latin and Devanagari alike. Kept deliberately dumb: a stemmer
    tuned for English would mangle Hinglish, and BM25 here is a recall net,
    not the precision layer."""
    return re.findall(r"[a-z0-9]+|[ऀ-ॿ]+", text.lower())


class BM25:
    """Okapi BM25 over the clause set.

    BM25 earns its place next to embeddings because customers quote strings
    verbatim — "T+3", "0.5%", "re-KYC", a clause id — and a dense model
    happily maps those onto something merely similar. Exact terms want an
    exact-term retriever.
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = (sum(self.lengths) / self.N) if self.N else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]

        df = Counter()
        for d in self.docs:
            df.update(set(d))
        # Standard BM25 IDF with the +0.5 smoothing, floored at a small
        # positive value so a term appearing in most clauses contributes
        # nothing rather than going negative and actively penalising a match.
        self.idf = {
            term: max(1e-6, math.log((self.N - n + 0.5) / (n + 0.5) + 1.0))
            for term, n in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = [0.0] * self.N
        for i in range(self.N):
            tf, dl = self.tf[i], self.lengths[i]
            if not dl:
                continue
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            out[i] = s
        return out

    def rank(self, query: str, k: Optional[int] = None) -> list[int]:
        s = self.scores(query)
        order = sorted(range(self.N), key=lambda i: -s[i])
        order = [i for i in order if s[i] > 0]
        return order[:k] if k else order


def rrf(rankings: Iterable[list[int]], k: int = RRF_K,
        weights: Optional[list[float]] = None) -> list[tuple[int, float]]:
    """Fuse ranked id lists. Returns [(doc_index, fused_score)], best first.

    `weights` lets one retriever count for more without reintroducing the
    scale problem — it scales the reciprocal, not the underlying score.
    """
    rankings = list(rankings)
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])
