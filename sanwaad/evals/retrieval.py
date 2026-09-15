"""Retrieval evaluation: does the right clause reach the drafting model?

Everything downstream is capped by this. A grounded draft citing the wrong
clause is still a wrong answer, and the grounding checker will happily wave it
through because the claim *is* supported — just by a clause that does not
apply. So retrieval is measured on its own, before any generation.

Metrics:
  recall@k   fraction of `must_retrieve` clauses present in the top k
  strict@k   1.0 only when EVERY must-clause is present — the honest one,
             because a reply needs all the governing clauses, not one of them
  MRR        1/rank of the first must-clause; how far down the list the
             model has to read before it finds something that matters
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable

from ..rag.store import CATEGORY_CLAUSES, get_store
from .golden import RETRIEVAL_CASES, GoldenCase

Retriever = Callable[[str, tuple[str, ...], int], list[str]]


@dataclass
class Result:
    name: str
    recall: float
    strict: float
    mrr: float
    per_lang: dict[str, float]
    failures: list[tuple[str, list[str], list[str]]]


def _metrics(got: list[str], want: list[str]) -> tuple[float, float, float]:
    hits = [c for c in want if c in got]
    recall = len(hits) / len(want)
    strict = 1.0 if len(hits) == len(want) else 0.0
    mrr = 0.0
    for i, c in enumerate(got):
        if c in want:
            mrr = 1.0 / (i + 1)
            break
    return recall, strict, mrr


def evaluate(name: str, retrieve: Retriever, k: int = 5,
             cases: list[GoldenCase] | None = None) -> Result:
    cases = cases or RETRIEVAL_CASES
    rec, st, mr = [], [], []
    lang: dict[str, list[float]] = {}
    failures = []

    for case in cases:
        boost = CATEGORY_CLAUSES.get(case.category or "", ())
        got = retrieve(case.text, boost, k)
        r, s, m = _metrics(got, case.must_retrieve)
        rec.append(r); st.append(s); mr.append(m)
        lang.setdefault(case.lang, []).append(s)
        if s < 1.0:
            failures.append((case.id, case.must_retrieve, got))

    return Result(
        name=name,
        recall=mean(rec), strict=mean(st), mrr=mean(mr),
        per_lang={k2: mean(v) for k2, v in sorted(lang.items())},
        failures=failures,
    )


# --- the strategies under test --------------------------------------------

def make_blend() -> Retriever:
    store = get_store()
    return lambda q, boost, k: [c.clause_id for c in store.search(q, k=k, boost_prefixes=boost)]


def make_rrf(weights=(1.0, 1.0)) -> Retriever:
    store = get_store()
    return lambda q, boost, k: [
        c.clause_id for c in store.search_fused(q, k=k, boost_prefixes=boost, weights=weights)
    ]


def make_dense_only() -> Retriever:
    store = get_store()
    return lambda q, boost, k: [
        c.clause_id for c in store.search(q, k=k, lexical_weight=0.0, boost_prefixes=boost)
    ]


def make_no_boost(inner: Retriever) -> Retriever:
    """Ablation: how much of the score is the category prior doing?"""
    return lambda q, boost, k: inner(q, (), k)


def report(results: list[Result], k: int) -> None:
    w = max(len(r.name) for r in results) + 2
    print(f"\n{'strategy'.ljust(w)} strict@{k}  recall@{k}   MRR    en    hi   hinglish")
    print("-" * (w + 48))
    for r in results:
        pl = r.per_lang
        print(f"{r.name.ljust(w)}  {r.strict:5.2f}    {r.recall:5.2f}  {r.mrr:5.2f}"
              f"  {pl.get('en', 0):4.2f}  {pl.get('hi', 0):4.2f}  {pl.get('hinglish', 0):6.2f}")


if __name__ == "__main__":
    K = 5
    results = [
        evaluate("dense only", make_dense_only(), K),
        evaluate("dense+lexical blend", make_blend(), K),
        evaluate("RRF (dense+BM25)", make_rrf(), K),
        evaluate("RRF dense-weighted 2:1", make_rrf(weights=(2.0, 1.0)), K),
        evaluate("blend, no category boost", make_no_boost(make_blend()), K),
    ]
    report(results, K)

    best = max(results, key=lambda r: (r.strict, r.mrr))
    print(f"\nbest: {best.name}")
    if best.failures:
        print(f"\nstill failing ({len(best.failures)}):")
        for cid, want, got in best.failures:
            print(f"  {cid:<18} want {want}  got {got[:4]}")
