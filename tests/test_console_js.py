"""Run the console's JavaScript tests (tests/js/) as part of the Python suite.

The console's logic decides what a reviewer sees and what an approval sends,
so a regression there is as real as one in the pipeline. Running it from
pytest puts it in CI and in the commit gate without a second test runner to
remember. Skipped, loudly, where Node is not installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = Path(__file__).parent / "js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_console_logic():
    proc = subprocess.run(["node", "--test", str(JS_TESTS)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
