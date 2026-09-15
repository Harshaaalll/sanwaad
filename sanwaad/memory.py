"""Memory design is data architecture.

"Memory" is not one thing, and the most common mistake is to treat it as one —
usually by pouring all of it into a vector database. Sanwaad keeps eight kinds
of remembered data, and each lives where its ACCESS PATTERN says it should:

  state              what one case in flight knows right now
  working            short-lived, bounded, cross-case scratch space
  knowledge          retrieved per step, never stuffed into every prompt
  episodic           what happened before that should shape what happens next
  audit              what happened, for people asking afterwards
  system of record   business truth, owned by the application, reached by tools

For every kind this module states: where it is stored, what it holds, how long
it is kept, whether it may reach a model, and what happens to personal data.
Retention is enforced by `prune()` for the file-backed stores; the checkpoint
database's retention is declared but not yet enforced (see DESIGN.md, exercise 3).

    python -m sanwaad.memory           the map, and what pruning would remove
    python -m sanwaad.memory --apply   actually prune
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MemoryTier:
    name: str
    kind: str
    store: str
    path_ref: tuple[str, str]          # (module, attribute) — resolved when used
    holds: str
    retention_days: Optional[int]      # None = kept until superseded or bounded by size
    prunable: bool
    reaches_model: str
    personal_data: str
    why: str

    def path(self) -> Path:
        module, attr = self.path_ref
        return getattr(importlib.import_module(module), attr)


MEMORY_MAP: list[MemoryTier] = [
    MemoryTier(
        "Workflow state", "state", "LangGraph checkpointer (SQLite)",
        ("sanwaad.config", "CHECKPOINT_PATH"),
        "One case in flight: every agent's conclusion, the pending interrupt, costs",
        90, False,
        "Selected fields per step — never the whole state",
        "Raw complaint text, because the reply depends on it",
        "A case parked on a human for two days must survive a restart. That is "
        "state, not memory, and it belongs in a durable low-latency store."),
    MemoryTier(
        "Cross-case window", "working", "JSON file, bounded to 500 entries",
        ("sanwaad.pattern", "MEMORY_PATH"),
        "Fingerprints of recent complaints: handle, category, summary vector, text",
        7, True,
        "Never — the pattern agent uses dot products, not prompts",
        "Identifiers redacted before storage",
        "The only memory about many cases at once. Losing it costs the current "
        "window, never a case, so a file is enough."),
    MemoryTier(
        "Policy knowledge", "knowledge", "Local embeddings + BM25 index",
        ("sanwaad.config", "INDEX_PATH"),
        "33 policy clauses and their vectors",
        None, False,
        "Yes — retrieved clauses are the only facts a reply may state",
        "None",
        "Retrieved per step with hybrid search. The smallest useful context, "
        "not the whole handbook in every prompt."),
    MemoryTier(
        "Dedupe memory", "working", "JSON file, bounded to 5,000 ids",
        ("sanwaad.listener", "SEEN_PATH"),
        "Ids of comments already handed downstream",
        None, False,
        "Never",
        "Ids only",
        "Forgetting this means replying twice in public."),
    MemoryTier(
        "Human corrections", "episodic", "JSONL",
        ("sanwaad.feedback", "FEEDBACK_PATH"),
        "What the model drafted vs what a reviewer actually sent",
        365, True,
        "Selected approved edits, retrieved as few-shot examples",
        "Identifiers redacted at write time",
        "The cheapest training data available: the work was happening anyway."),
    MemoryTier(
        "Traces", "audit", "JSONL",
        ("sanwaad.obs", "TRACE_PATH"),
        "Per-step timing, model, prompt version, attempts, fallbacks, cost",
        30, True,
        "Never",
        "No customer text is written to spans",
        "Answers 'which step failed, and why' after the fact."),
    MemoryTier(
        "Tool audit log", "audit", "JSONL",
        ("sanwaad.tools.registry", "AUDIT_PATH"),
        "Every tool call: agent, arguments, approval, outcome",
        180, True,
        "Never",
        "String arguments redacted",
        "Who moved money, on whose approval, is not an optional question."),
    MemoryTier(
        "Conversation memory", "episodic", "JSON file, 5 turns per customer",
        ("sanwaad.loop.support", "CONVERSATIONS_PATH"),
        "What earlier support turns decided: intent, outcome, the open ticket",
        30, False,
        "Yes — one line per earlier turn, at the start of the next turn",
        "Decisions only, redacted; no live facts",
        "A returning customer should not start again or get a second ticket. Live "
        "facts go stale, so only decisions are kept; turns older than 30 days are "
        "ignored at recall."),
    MemoryTier(
        "Loop run log", "audit", "JSONL",
        ("sanwaad.loop.kernel", "LOOP_RUNS_PATH"),
        "Every support-loop run: steps, verdicts, stop reason, spend, context stats",
        30, True,
        "Never — the outer loop and the trace evals read it",
        "Messages and answers redacted",
        "The external feedback loop learns from real runs, not only from the "
        "scenarios someone thought to write."),
    MemoryTier(
        "Payments and tickets", "system of record", "Application backend (mock: JSON)",
        ("sanwaad.tools.ledger", "BACKEND_PATH"),
        "Transactions, reversals, tickets",
        None, False,
        "Minimal views, through tools only",
        "Account ids never leave the backend",
        "Business truth lives in the application. An agent that remembers "
        "'already reversed' instead of asking the ledger reverses twice."),
]


def _parse(ts: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _prune_jsonl(path: Path, floor: datetime, dry_run: bool) -> tuple[int, int, int]:
    """Returns (kept, removed, undated). Undated rows are kept: when in doubt
    about when something happened, deleting it is the irreversible choice."""
    if not path.exists():
        return 0, 0, 0
    kept_lines, removed, undated = [], 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            at = _parse(json.loads(line).get("at"))
        except (json.JSONDecodeError, AttributeError):
            at = None
        if at is None:
            undated += 1
            kept_lines.append(line)
        elif at < floor:
            removed += 1
        else:
            kept_lines.append(line)
    if removed and not dry_run:
        path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return len(kept_lines), removed, undated


def _prune_json_list(path: Path, floor: datetime, dry_run: bool) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0, 0, 0
    kept, removed, undated = [], 0, 0
    for row in rows:
        at = _parse(row.get("at")) if isinstance(row, dict) else None
        if at is None:
            undated += 1
            kept.append(row)
        elif at < floor:
            removed += 1
        else:
            kept.append(row)
    if removed and not dry_run:
        path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    return len(kept), removed, undated


def prune(now: Optional[datetime] = None, dry_run: bool = False) -> list[dict]:
    """Apply each prunable tier's retention. Returns one report row per tier."""
    now = now or datetime.now(timezone.utc)
    report = []
    for tier in MEMORY_MAP:
        if not tier.prunable or tier.retention_days is None:
            continue
        path = tier.path()
        floor = now - timedelta(days=tier.retention_days)
        pruner = _prune_json_list if path.suffix == ".json" else _prune_jsonl
        kept, removed, undated = pruner(path, floor, dry_run)
        report.append({"tier": tier.name, "path": str(path), "retention_days": tier.retention_days,
                       "kept": kept, "removed": removed, "undated_kept": undated,
                       "dry_run": dry_run})
    return report


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    print(f"\n{'memory':<22}{'kind':<18}{'kept':<10}reaches the model")
    print("-" * 86)
    for t in MEMORY_MAP:
        keep = f"{t.retention_days}d" if t.retention_days else "bounded"
        print(f"{t.name:<22}{t.kind:<18}{keep:<10}{t.reaches_model}")
    print()
    for row in prune(dry_run=not apply):
        verb = "removed" if apply else "would remove"
        print(f"  {row['tier']:<22} {verb} {row['removed']}, keeping {row['kept']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
