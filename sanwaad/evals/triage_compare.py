"""Which triage model is best for us, measured on our own complaints.

    python -m sanwaad.evals.triage_compare --data complaints.csv
    python -m sanwaad.evals.triage_compare --data complaints.csv --backends gemini,laya --limit 50

The CSV needs a `text` column and, for every row that is scored, a `category`
column (see sanwaad/evals/data/triage_template.csv). `id` and `channel` are
optional. A row with an empty category is still run, and counts towards how
often the backends agree, but not towards accuracy.

What it reports, per backend:

- accuracy and macro-F1 on category, plus precision and recall per category,
  because a model can reach 80% accuracy by never saying `data_privacy`, and
  the category it never says is the one that escalates;
- a confusion matrix, to show *which* categories it mixes up;
- calibration (expected calibration error) and accuracy at confidence
  thresholds, for backends that report confidence. This is the number that
  sets SANWAAD_TRIAGE_MIN_CONFIDENCE: the lowest threshold whose accuracy you
  can live with, and the coverage it costs;
- p50/p95 latency and cost per 1,000 comments;
- on the fields nobody labelled (severity, sentiment, is_complaint,
  needs_private_data): how often each pair of backends agrees. Agreement is
  not accuracy — two models can agree and both be wrong — so it is reported
  as agreement and nothing more.

Results go to sanwaad/data/triage_compare.json, which the console's Model
comparison tab reads. Example texts in that file are redacted first: this runs
on real complaints, and the file is served to anyone who opens the console.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sanwaad.config import DATA_DIR  # noqa: E402
from sanwaad.guardrails import check_complaint, redact  # noqa: E402
from sanwaad.models import Category  # noqa: E402

RESULTS_PATH = DATA_DIR / "triage_compare.json"
CATEGORIES = [c.value for c in Category]
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
UNLABELLED_FIELDS = ("severity", "sentiment", "is_complaint", "needs_private_data")
MAX_EXAMPLES = 40


class DatasetError(ValueError):
    """The CSV cannot be scored as it stands, and the message says what to fix."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _normalise_label(raw: str) -> str:
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def load_rows(path: Path) -> list[dict]:
    """Read and check the CSV before any model is paid to read it."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = {f.strip().lower() for f in (reader.fieldnames or [])}
        if "text" not in fields:
            raise DatasetError(f"{path.name} has no `text` column (found: {sorted(fields)})")
        rows, unknown = [], Counter()
        for n, raw in enumerate(reader, start=2):      # line 1 is the header
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            if not row.get("text"):
                continue
            label = _normalise_label(row.get("category", ""))
            if label and label not in CATEGORIES:
                unknown[label] += 1
            rows.append({
                "id": row.get("id") or f"row{n}",
                "text": row["text"],
                "channel": row.get("channel") or "reddit",
                "gold": label or None,
            })
    if unknown:
        listed = ", ".join(f"{k} ({v})" for k, v in unknown.most_common())
        raise DatasetError(
            f"categories not in the taxonomy: {listed}. Allowed: {', '.join(CATEGORIES)}. "
            "Relabel those rows, or add the category to sanwaad/models.py:Category "
            "and to CATEGORY_CRITERIA in sanwaad/triage_backends.py.")
    if not rows:
        raise DatasetError(f"{path.name} has no rows with text")
    return rows


# ---------------------------------------------------------------------------
# Metrics — plain Python, so the numbers are checkable by hand
# ---------------------------------------------------------------------------

def classification_report(pairs: list[tuple[str, str]]) -> dict:
    """Accuracy, macro-F1, per-class precision/recall and a confusion matrix.

    `pairs` is (gold, predicted). Macro-F1 averages over the classes that
    appear in gold, so a category absent from the data neither helps nor hurts.
    """
    if not pairs:
        return {"n": 0, "accuracy": None, "macro_f1": None, "per_class": {}, "confusion": {}}
    confusion: dict[str, Counter] = {}
    for gold, pred in pairs:
        confusion.setdefault(gold, Counter())[pred] += 1
    per_class, f1s = {}, []
    for label in sorted({g for g, _ in pairs}):
        tp = confusion.get(label, Counter())[label]
        support = sum(confusion.get(label, Counter()).values())
        predicted = sum(c[label] for c in confusion.values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": support}
        f1s.append(f1)
    correct = sum(1 for g, p in pairs if g == p)
    return {
        "n": len(pairs),
        "accuracy": round(correct / len(pairs), 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class": per_class,
        "confusion": {g: dict(c) for g, c in confusion.items()},
    }


def expected_calibration_error(scored: list[tuple[float, bool]], bins: int = 10) -> Optional[float]:
    """Mean gap between stated confidence and observed accuracy, weighted by bin size.

    0 means "70% sure" is right 70% of the time. A model that says 0.95 and is
    right 60% of the time cannot have its confidence used as a gate at all.
    """
    if not scored:
        return None
    total, err = len(scored), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [(c, ok) for c, ok in scored if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if not in_bin:
            continue
        conf = sum(c for c, _ in in_bin) / len(in_bin)
        acc = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        err += abs(acc - conf) * len(in_bin) / total
    return round(err, 4)


def selective_accuracy(scored: list[tuple[float, bool]],
                       thresholds: tuple[float, ...] = THRESHOLDS) -> list[dict]:
    """At each threshold: what share of comments the model keeps, and how right it is on those.

    The rest fall back to Gemini on the live path, so coverage is the share of
    traffic the cheaper model actually handles.
    """
    out = []
    for t in thresholds:
        kept = [ok for c, ok in scored if c >= t]
        out.append({
            "threshold": t,
            "coverage": round(len(kept) / len(scored), 4) if scored else None,
            "accuracy": round(sum(kept) / len(kept), 4) if kept else None,
        })
    return out


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


def agreement(preds: dict[str, dict[str, dict]], row_ids: list[str]) -> list[dict]:
    """Pairwise agreement between backends on the fields nobody labelled."""
    names = sorted(preds)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [r for r in row_ids if r in preds[a] and r in preds[b]]
            if not both:
                continue
            entry = {"a": a, "b": b, "n": len(both)}
            for fld in UNLABELLED_FIELDS:
                same = sum(1 for r in both if preds[a][r][fld] == preds[b][r][fld])
                entry[fld] = round(same / len(both), 4)
            near = sum(1 for r in both
                       if abs(int(preds[a][r]["severity"]) - int(preds[b][r]["severity"])) <= 1)
            entry["severity_within_1"] = round(near / len(both), 4)
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

async def _run_backend(backend, rows: list[dict], concurrency: int) -> tuple[dict, list[str]]:
    """Classify every row. Returns ({row_id: labels}, errors)."""
    gate = asyncio.Semaphore(concurrency)
    out: dict[str, dict] = {}
    errors: list[str] = []

    async def one(row: dict) -> None:
        async with gate:
            # The same input guardrail live triage applies, so the model sees
            # the text it would see in production.
            text, _ = check_complaint(row["text"])
            try:
                out[row["id"]] = (await backend.classify(text, row["channel"])).to_dict()
            except Exception as exc:
                errors.append(f"{row['id']}: {type(exc).__name__}: {exc}")

    await asyncio.gather(*(one(r) for r in rows))
    return out, errors


def _summarise_backend(name: str, detail: str, preds: dict[str, dict], errors: list[str],
                       rows: list[dict], wall_s: float) -> dict:
    labelled = [r for r in rows if r["gold"] and r["id"] in preds]
    pairs = [(r["gold"], preds[r["id"]]["category"]) for r in labelled]
    scored = [(float(preds[r["id"]]["confidence"]), preds[r["id"]]["category"] == r["gold"])
              for r in labelled if preds[r["id"]].get("confidence") is not None]
    latencies = [p["latency_ms"] for p in preds.values()]
    cost_inr = sum(float((p.get("cost") or {}).get("inr", 0.0)) for p in preds.values())
    models = Counter(p["model"] for p in preds.values())
    return {
        "backend": name,
        "detail": detail,
        "models": dict(models),
        "rows_run": len(preds),
        "errors": errors[:20],
        "error_count": len(errors),
        "category": classification_report(pairs),
        "calibration": {
            "reports_confidence": bool(scored),
            "ece": expected_calibration_error(scored),
            "selective": selective_accuracy(scored) if scored else [],
        },
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95),
                       "mean": round(statistics.fmean(latencies), 1) if latencies else None},
        "cost_inr": {"total": round(cost_inr, 6),
                     "per_1k": round(cost_inr / len(preds) * 1000, 4) if preds else None},
        "wall_s": round(wall_s, 1),
    }


def _examples(rows: list[dict], preds: dict[str, dict[str, dict]]) -> list[dict]:
    """The rows worth a person's eyes: any backend wrong, or backends disagreeing."""
    out = []
    for r in rows:
        got = {name: p[r["id"]] for name, p in preds.items() if r["id"] in p}
        if not got:
            continue
        cats = {g["category"] for g in got.values()}
        wrong = r["gold"] is not None and any(g["category"] != r["gold"] for g in got.values())
        if not wrong and len(cats) <= 1:
            continue
        text, _ = redact(r["text"])
        out.append({
            "id": r["id"],
            "text": text[:280],
            "gold": r["gold"],
            "predictions": {name: {"category": g["category"], "confidence": g.get("confidence"),
                                   "severity": g["severity"]} for name, g in got.items()},
        })
        if len(out) >= MAX_EXAMPLES:
            break
    return out


async def compare(rows: list[dict], backend_names: list[str], *, dataset: str,
                  concurrency: int = 4) -> dict:
    from sanwaad.triage_backends import get_backend

    summaries, skipped, preds = [], [], {}
    for name in backend_names:
        backend = get_backend(name)
        ok, detail = backend.available()
        if not ok:
            skipped.append({"backend": name, "reason": detail})
            print(f"  skip {name}: {detail}")
            continue
        print(f"  run  {name} ({detail}) on {len(rows)} rows…", flush=True)
        try:
            # A cold start (download, weight load) is a one-off, not latency.
            await asyncio.to_thread(backend.warm)
        except Exception as exc:
            skipped.append({"backend": name, "reason": f"failed to load: {type(exc).__name__}: {exc}"})
            print(f"  skip {name}: failed to load: {exc}")
            continue
        started = time.perf_counter()
        # Local models are CPU-bound: one at a time is faster than contention.
        got, errors = await _run_backend(backend, rows, 1 if name == "laya" else concurrency)
        preds[name] = got
        summaries.append(_summarise_backend(name, detail, got, errors, rows,
                                            time.perf_counter() - started))

    gold = Counter(r["gold"] for r in rows if r["gold"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": dataset, "rows": len(rows),
                    "labelled": sum(gold.values()), "class_counts": dict(gold.most_common())},
        "backends": summaries,
        "skipped": skipped,
        "agreement": agreement(preds, [r["id"] for r in rows]),
        "examples": _examples(rows, preds),
    }


def _print(result: dict) -> None:
    print(f"\n{result['dataset']['name']}: {result['dataset']['rows']} rows, "
          f"{result['dataset']['labelled']} labelled\n")
    head = f"{'backend':<10}{'accuracy':>10}{'macro-F1':>10}{'ECE':>8}{'p50 ms':>9}{'p95 ms':>9}{'₹/1k':>10}"
    print(head)
    print("-" * len(head))
    for b in result["backends"]:
        cat, cal = b["category"], b["calibration"]

        def fmt(v, spec):
            # A missing number keeps its column: ">10.1%" pads "—" to 10 too.
            width = "".join(ch for ch in spec.split(".")[0] if ch.isdigit())
            return format(v, spec) if v is not None else format("—", f">{width or 1}")
        print(f"{b['backend']:<10}{fmt(cat['accuracy'], '>10.1%')}{fmt(cat['macro_f1'], '>10.3f')}"
              f"{fmt(cal['ece'], '>8.3f')}{fmt(b['latency_ms']['p50'], '>9.0f')}"
              f"{fmt(b['latency_ms']['p95'], '>9.0f')}{fmt(b['cost_inr']['per_1k'], '>10.4f')}")
        if b["error_count"]:
            print(f"  {b['error_count']} rows failed, e.g. {b['errors'][0]}")
    for s in result["skipped"]:
        print(f"{s['backend']:<10}skipped — {s['reason']}")
    print(f"\nwritten to {RESULTS_PATH}; open the console's Model comparison tab")


def main(argv: Optional[list[str]] = None) -> int:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path,
                    default=Path(__file__).parent / "data" / "triage_template.csv")
    ap.add_argument("--backends", default="gemini,laya,jev")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)

    try:
        rows = load_rows(args.data)
    except (DatasetError, FileNotFoundError) as exc:
        print(f"cannot use {args.data}: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        rows = rows[: args.limit]
    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    result = asyncio.run(compare(rows, names, dataset=args.data.name,
                                 concurrency=args.concurrency))
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    _print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
