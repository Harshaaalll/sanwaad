#!/usr/bin/env python3
"""Test gates for Claude Code in this repo. Standard library only.

    gate.py stop         Stop hook: ruff + the full pytest suite, if Python
                         changed since the last green run. Blocks finishing on
                         a failure, at most MAX_BLOCKS times in a row, then lets
                         the turn end with a warning instead of looping.
    gate.py post-edit    PostToolUse hook on Edit|Write: ruff on the edited
                         file, and the test files that mention its module.
    gate.py pre-commit   git pre-commit: ruff + full pytest; non-zero blocks.

Hook input arrives as JSON on stdin; decisions go out as JSON on stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
STATE = ROOT / ".claude" / ".gate"
MAX_BLOCKS = 3
TAIL = 3500          # characters of failure output handed back to Claude


def _run(args: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {' '.join(args)}"


def _ruff(paths: list[str]) -> tuple[int, str]:
    return _run([str(PY), "-m", "ruff", "check", *paths], 60)


def _pytest(targets: list[str], timeout: int) -> tuple[int, str]:
    return _run([str(PY), "-m", "pytest", *targets, "-q", "-x", "--no-header",
                 "-p", "no:cacheprovider"], timeout)


def _python_fingerprint() -> str:
    """A hash of every Python change against HEAD, tracked and untracked."""
    diff = _run(["git", "diff", "HEAD", "--", "*.py"], 30)[1]
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"], 30)[1]
    h = hashlib.sha256(diff.encode())
    for name in sorted(untracked.split()):
        try:
            h.update(name.encode() + (ROOT / name).read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state))


def full_check() -> tuple[bool, str]:
    code, out = _ruff(["sanwaad", "tests"])
    if code:
        return False, "ruff failed:\n" + out
    code, out = _pytest(["tests"], 900)
    if code:
        return False, "pytest failed:\n" + out
    return True, out.strip().splitlines()[-1] if out.strip() else "ok"


def stop(payload: dict) -> int:
    fingerprint = _python_fingerprint()
    state = _state()
    if state.get("green") == fingerprint:
        return 0                                  # nothing changed since the last pass
    ok, detail = full_check()
    if ok:
        _save({"green": fingerprint, "blocks": 0})
        return 0
    blocks = int(state.get("blocks", 0)) + 1
    _save({"green": state.get("green"), "blocks": blocks})
    if blocks > MAX_BLOCKS:
        _save({"green": state.get("green"), "blocks": 0})
        print(json.dumps({"systemMessage": f"Test gate: still failing after {MAX_BLOCKS} "
                          "attempts; stopping so a person can look. Last output:\n"
                          + detail[-800:]}))
        return 0
    print(json.dumps({
        "decision": "block",
        "reason": (f"Test gate ({blocks}/{MAX_BLOCKS}): the change is not done until ruff and "
                   "pytest pass. Fix the failure below, or tell the user why it cannot be "
                   "fixed.\n\n" + detail[-TAIL:]),
    }))
    return 0


def _tests_for(path: Path) -> list[str]:
    rel = path.relative_to(ROOT)
    if rel.parts[0] == "tests":
        return [str(rel)] if rel.name.startswith("test_") else []
    if rel.parts[0] != "sanwaad" or rel.suffix != ".py":
        return []
    stem = rel.stem if rel.stem != "__init__" else rel.parent.name
    dotted = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
    pattern = re.compile(rf"\b({re.escape(dotted)}|{re.escape(stem)})\b")
    hits = []
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        if test.stem == f"test_{stem}":
            hits.insert(0, str(test.relative_to(ROOT)))
        elif pattern.search(test.read_text(encoding="utf-8", errors="ignore")):
            hits.append(str(test.relative_to(ROOT)))
    return hits[:4]


def post_edit(payload: dict) -> int:
    raw = (payload.get("tool_input") or {}).get("file_path") or \
          (payload.get("tool_response") or {}).get("filePath") or ""
    if not raw.endswith(".py"):
        return 0
    path = Path(raw).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return 0
    problems = []
    code, out = _ruff([str(path)])
    if code:
        problems.append("ruff:\n" + out)
    tests = _tests_for(path)
    if tests:
        code, out = _pytest(tests, 240)
        if code:
            problems.append(f"pytest {' '.join(tests)}:\n" + out)
    if problems:
        print(json.dumps({
            "decision": "block",
            "reason": "Quick check after this edit found problems:\n\n"
                      + "\n\n".join(problems)[-TAIL:],
        }))
    return 0


def pre_commit() -> int:
    ok, detail = full_check()
    if not ok:
        sys.stderr.write("pre-commit: tests must pass before committing.\n\n" + detail[-TAIL:] + "\n")
        return 1
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pre-commit":
        return pre_commit()
    if os.environ.get("SANWAAD_SKIP_GATE") == "1":
        return 0
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        payload = {}
    if mode == "stop":
        return stop(payload)
    if mode == "post-edit":
        return post_edit(payload)
    sys.stderr.write(f"unknown mode {mode!r}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
