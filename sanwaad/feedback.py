"""The feedback loop: turn human corrections into training signal.

Every time a reviewer edits a draft before posting, they produce something
expensive and perishable — a labelled pair of (what the model wrote, what a
competent human actually sends). Most systems throw this away. Capturing it is
the cheapest data collection available, because the work was going to happen
anyway.

What the captured pairs are worth, in increasing order of effort:

  1. DIAGNOSIS   which failure modes recur — the fastest route to a prompt fix
  2. EVAL CASES  a reviewer rejection is, by definition, a case worth testing
  3. FEW-SHOTS   approved edits, retrieved by similarity, steer the next draft
  4. FINE-TUNING once there are thousands of pairs, train the behaviour in

Most teams reach for (4) first. (1) through (3) need no GPUs, ship the same
week, and usually close most of the gap — and (4) is only viable once you have
the corpus that (1) through (3) accumulate for free.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from .config import DATA_DIR
from .guardrails import redact

FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"


@dataclass
class FeedbackRecord:
    case_id: str
    decision: Literal["approve", "edit", "reject"]
    complaint: str
    draft: str
    final: str
    category: str
    severity: int
    language: str
    citations: list[str] = field(default_factory=list)
    reviewer: str = "human"
    note: str = ""
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def was_edited(self) -> bool:
        return self.decision == "edit" and self.norm(self.draft) != self.norm(self.final)

    @staticmethod
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "").strip().lower())

    def diff(self) -> list[str]:
        """Human-readable edit script, for the console and for pattern mining."""
        a, b = (self.draft or "").split(), (self.final or "").split()
        out = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
            if tag == "equal":
                continue
            removed, added = " ".join(a[i1:i2]), " ".join(b[j1:j2])
            if tag == "delete":
                out.append(f"- {removed}")
            elif tag == "insert":
                out.append(f"+ {added}")
            else:
                out.append(f"~ {removed} -> {added}")
        return out

    def similarity(self) -> float:
        return difflib.SequenceMatcher(a=self.norm(self.draft), b=self.norm(self.final)).ratio()


def record(fb: FeedbackRecord, path: Optional[Path] = None) -> None:
    """Append one correction, with identifiers redacted.

    This file is kept for a year and mined for few-shot examples, so it is
    long-term memory — and long-term memory is the wrong place for a phone
    number. The diff a reviewer made survives redaction; the identifier does not.
    """
    path = path or FEEDBACK_PATH
    row = asdict(fb)
    for key in ("complaint", "draft", "final"):
        row[key] = redact(row.get(key) or "")[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load(path: Optional[Path] = None) -> list[FeedbackRecord]:
    path = path or FEEDBACK_PATH
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(FeedbackRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# 1. Diagnosis
# ---------------------------------------------------------------------------

def summarise(records: list[FeedbackRecord]) -> dict:
    """What is the reviewer actually fixing?

    `edit_rate` is the headline health metric: the fraction of drafts a human
    had to touch. It is the number to watch after any prompt change, and it is
    the one that tells you when the model is ready for a wider auto-post band.
    """
    if not records:
        return {"n": 0}

    decisions = Counter(r.decision for r in records)
    edited = [r for r in records if r.was_edited]

    # Words the reviewer consistently deletes are prompt-fix candidates.
    removed, added = Counter(), Counter()

    def words_of(text: str) -> list[str]:
        return re.findall(r"[a-z']{4,}", text.lower())[:12]

    for r in edited:
        for line in r.diff():
            # A replace line carries BOTH sides ("~ old -> new") and must be
            # split, or every deleted word is counted as an addition and the
            # diagnosis comes out exactly backwards.
            if line.startswith("~ "):
                before, _, after = line[2:].partition(" -> ")
                removed.update(words_of(before))
                added.update(words_of(after))
            elif line.startswith("- "):
                removed.update(words_of(line[2:]))
            elif line.startswith("+ "):
                added.update(words_of(line[2:]))

    by_cat = Counter(r.category for r in records if r.decision != "approve")

    return {
        "n": len(records),
        "decisions": dict(decisions),
        "edit_rate": round(len(edited) / len(records), 3),
        "reject_rate": round(decisions["reject"] / len(records), 3),
        "mean_similarity_when_edited": (
            round(sum(r.similarity() for r in edited) / len(edited), 3) if edited else None),
        "top_removed": removed.most_common(8),
        "top_added": added.most_common(8),
        "problem_categories": by_cat.most_common(5),
    }


# ---------------------------------------------------------------------------
# 2. Eval cases from rejections
# ---------------------------------------------------------------------------

def to_eval_cases(records: list[FeedbackRecord]) -> list[dict]:
    """A rejected or heavily-edited draft is a regression test waiting to be
    written. Exported in the golden-set shape so they can be reviewed and
    merged by hand — never merged automatically, because a reviewer's edit can
    itself be wrong, and a golden set that absorbs mistakes is worse than none.
    """
    out = []
    for r in records:
        if r.decision == "reject" or (r.was_edited and r.similarity() < 0.75):
            out.append({
                "id": f"fb-{r.case_id[-6:]}",
                "text": r.complaint,
                "lang": "hi" if re.search(r"[ऀ-ॿ]", r.complaint) else "en",
                "is_complaint": True,
                "category": r.category,
                "severity_band": [max(1, r.severity - 1), min(5, r.severity + 1)],
                "must_retrieve": r.citations[:2],
                "note": f"from reviewer {r.decision}: {r.note or 'no note'}",
                "_human_reference": r.final,
            })
    return out


# ---------------------------------------------------------------------------
# 3. Few-shot mining
# ---------------------------------------------------------------------------

def few_shots(records: list[FeedbackRecord], category: str, n: int = 3) -> list[dict]:
    """Best human-approved examples for a category, newest first.

    Approved-as-written drafts are excluded on purpose: they teach the model
    what it already does. The signal lives in what a human *changed*.
    """
    pool = [r for r in records if r.category == category and r.was_edited]
    pool.sort(key=lambda r: r.at, reverse=True)
    return [{"complaint": r.complaint, "reply": r.final} for r in pool[:n]]


def render_few_shots(shots: list[dict]) -> str:
    if not shots:
        return ""
    body = "\n\n".join(
        f"Customer: {s['complaint'][:220]}\nGood reply: {s['reply']}" for s in shots)
    return ("Examples of replies a human reviewer approved for this kind of "
            f"complaint:\n\n{body}")


# ---------------------------------------------------------------------------
# 4. Fine-tuning export
# ---------------------------------------------------------------------------

def to_sft_jsonl(records: list[FeedbackRecord], system_prompt: str,
                 out_path: Optional[Path] = None, min_pairs: int = 500) -> dict:
    """Export approved replies as supervised fine-tuning pairs.

    Deliberately refuses to write a tiny file. Fine-tuning on a few dozen
    examples reliably makes a model worse in ways that do not show up until
    production: it overfits the phrasing of whoever reviewed that week and
    loses the general instruction-following you were relying on. The gate is a
    feature, not an inconvenience.
    """
    usable = [r for r in records if r.decision in ("approve", "edit") and r.final]
    ready = len(usable) >= min_pairs
    report = {"usable_pairs": len(usable), "min_pairs": min_pairs, "ready": ready,
              "written": 0, "path": None}
    if not ready or out_path is None:
        report["advice"] = (
            f"{len(usable)} pairs. Below {min_pairs}, prompt fixes and retrieved "
            f"few-shots beat fine-tuning — and cost nothing to undo.")
        return report

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in usable:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": r.complaint},
                {"role": "assistant", "content": r.final},
            ]}, ensure_ascii=False) + "\n")
    report["written"] = len(usable)
    report["path"] = str(out_path)
    return report
