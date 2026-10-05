"""Shared test setup: nothing a test does may reach a developer's real data.

- Every test gets its own empty accounts database. Without this, a developer
  who has created a console account would see API tests fail with 401.
- Every test gets its own policy audit log, and every policy value is put back
  afterwards. The role-matrix test really applies a policy change as an admin;
  before this fixture it wrote that change to the real audit log, which the
  server replays on start, so the next console launch would silently have run
  with a lowered auto-post threshold.
"""

import sys
from pathlib import Path

import pytest

# Plain `pytest` (as CI runs it) loads this before any test file adds the repo to
# sys.path the way each of them does, so do the same here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import auth, policy_store  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_accounts(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "USERS_PATH", tmp_path / "users.sqlite")
    monkeypatch.setattr(auth, "_failures", {})


@pytest.fixture(autouse=True)
def _isolated_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(policy_store, "AUDIT_PATH", tmp_path / "policy_audit.jsonl")
    yield
    for name, values in policy_store.DEFAULTS.items():
        for field, value in values.items():
            policy_store._set(name, field, value)
