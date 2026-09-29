"""The checkpoint database under a burst, and at exit.

Two failures pinned here, both from the same change of shape:

- Concurrent cases each opened their own connection, and under WAL a writer
  holding a stale read snapshot gets "database is locked" at once — the busy
  timeout cannot wait its way out of it. A burst of complaints, the exact load
  bounded concurrency exists for, turned some of them into delivery failures.
- The fix shares one connection, whose worker thread is not a daemon by
  default, so every CLI that ran a case finished its work and never exited.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sanwaad import pipeline
from sanwaad.connectors import get_connector


@pytest.mark.asyncio
async def test_a_burst_of_cases_all_reach_the_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "CHECKPOINT_PATH", tmp_path / "burst.sqlite")
    complaints = await get_connector("mock").fetch(limit=15)

    results = await asyncio.gather(*(pipeline.run_case(c) for c in complaints),
                                   return_exceptions=True)

    errors = [f"{type(r).__name__}: {r}" for r in results if isinstance(r, BaseException)]
    assert not errors, errors
    assert len(await pipeline.list_cases()) == len(complaints)
    await pipeline.close_sessions()


def test_a_script_that_ran_a_case_exits(tmp_path):
    script = f"""
import asyncio, sys, pathlib
sys.path.insert(0, {str(ROOT)!r})
from sanwaad import pipeline
from sanwaad.connectors import get_connector
pipeline.CHECKPOINT_PATH = pathlib.Path({str(tmp_path / 'exit.sqlite')!r})

async def main():
    (item,) = await get_connector("mock").fetch(limit=1)
    await pipeline.run_case(item)

asyncio.run(main())
print("done")
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=180)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("done")
