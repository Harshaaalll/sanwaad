"""Accounts, roles and sessions.

The properties that matter: a password is never stored or compared in a way
that leaks it; once any account exists nothing is reachable without a session;
each role reaches exactly its endpoints; and decisions carry the signed-in
person's name rather than one the browser supplied.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanwaad import auth
from sanwaad.api import server

PW = "a-long-password"


def _client():
    return TestClient(server.app)


def _signed_in(role, email=None):
    email = email or f"{role}@example.com"
    auth.add_user(email, f"{role.title()} Person", role, PW)
    c = _client()
    assert c.post("/api/login", json={"email": email, "password": PW}).status_code == 200
    return c


# --- the account store ------------------------------------------------------------

def test_a_password_is_stored_only_as_a_salted_hash():
    user = auth.add_user("A@Example.com ", "Asha", "agent", PW)
    assert user.email == "a@example.com"
    stored = auth._db().execute("SELECT pw_hash FROM users").fetchone()[0]
    assert PW not in stored and stored.startswith("scrypt$")
    assert auth.hash_password(PW) != auth.hash_password(PW), "a fresh salt every time"
    assert auth.verify_password(PW, stored) and not auth.verify_password("wrong-password", stored)


@pytest.mark.parametrize("email,name,role,password,message", [
    ("not-an-email", "A", "agent", PW, "valid email"),
    ("a@x.com", "", "agent", PW, "name"),
    ("a@x.com", "A", "owner", PW, "role must be"),
    ("a@x.com", "A", "agent", "short", "at least 10"),
])
def test_bad_accounts_are_refused(email, name, role, password, message):
    with pytest.raises(auth.AuthError, match=message):
        auth.add_user(email, name, role, password)


def test_one_account_per_email():
    auth.add_user("a@x.com", "A", "agent", PW)
    with pytest.raises(auth.AuthError, match="already has an account"):
        auth.add_user("A@X.COM", "B", "agent", PW)


def test_the_last_admin_cannot_be_demoted_or_disabled():
    admin = auth.add_user("boss@x.com", "Boss", "admin", PW)
    with pytest.raises(auth.AuthError, match="last active admin"):
        auth.update_user(admin.id, role="lead")
    with pytest.raises(auth.AuthError, match="last active admin"):
        auth.update_user(admin.id, disabled=True)
    auth.add_user("second@x.com", "Second", "admin", PW)
    assert auth.update_user(admin.id, role="lead").role == "lead"


def test_a_wrong_email_and_a_wrong_password_look_the_same():
    auth.add_user("a@x.com", "A", "agent", PW)
    for email, password in (("nobody@x.com", PW), ("a@x.com", "wrong-password")):
        with pytest.raises(auth.AuthError) as err:
            auth.authenticate(email, password)
        assert str(err.value) == "wrong email or password"


def test_repeated_wrong_passwords_lock_the_email_out():
    auth.add_user("a@x.com", "A", "agent", PW)
    for _ in range(auth.MAX_FAILURES):
        with pytest.raises(auth.AuthError):
            auth.authenticate("a@x.com", "wrong-password")
    with pytest.raises(auth.AuthError, match="too many failed attempts"):
        auth.authenticate("a@x.com", PW)       # even the right password, until the window passes


def test_disabling_an_account_or_changing_its_password_ends_its_sessions():
    auth.add_user("boss@x.com", "Boss", "admin", PW)
    user = auth.add_user("a@x.com", "A", "agent", PW)
    token = auth.start_session(user)
    assert auth.user_for(token) == user
    auth.update_user(user.id, password="another-long-one")
    assert auth.user_for(token) is None
    token = auth.start_session(auth.authenticate("a@x.com", "another-long-one"))
    auth.update_user(user.id, disabled=True)
    assert auth.user_for(token) is None
    with pytest.raises(auth.AuthError):
        auth.authenticate("a@x.com", "another-long-one")


def test_only_a_hash_of_the_session_token_is_stored():
    token = auth.start_session(auth.add_user("a@x.com", "A", "agent", PW))
    stored = auth._db().execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert token not in stored and len(stored) == 64


# --- the API ------------------------------------------------------------------------

# Every protected endpoint and the lowest role that may use it. A new endpoint
# that forgets its guard fails test_every_endpoint_refuses_anyone_signed_out.
ENDPOINTS = [
    ("get", "/api/cases", None, "agent"),
    ("get", "/api/cases/case_missing", None, "agent"),
    ("get", "/api/delivery", None, "agent"),
    ("get", "/api/policy/search?q=refund", None, "agent"),
    ("get", "/api/overview", None, "lead"),
    ("get", "/api/settings", None, "lead"),
    ("get", "/api/evals/triage", None, "lead"),
    ("post", "/api/settings/preview", {"policy": "review", "field": "auto_post_max_severity", "value": 1}, "lead"),
    ("post", "/api/settings/change", {"policy": "review", "field": "auto_post_max_severity", "value": 1,
                                      "reason": "tighten it"}, "admin"),
    ("get", "/api/users", None, "admin"),
]


def test_with_no_accounts_the_console_runs_open():
    assert _client().get("/api/cases").status_code == 200
    assert _client().get("/api/me").json() == {"mode": "open", "user": None}
    assert _client().get("/ready").json()["auth"] == "open"


@pytest.mark.parametrize("method,path,body,_", ENDPOINTS)
def test_every_endpoint_refuses_anyone_signed_out(method, path, body, _):
    auth.add_user("boss@x.com", "Boss", "admin", PW)
    r = getattr(_client(), method)(path, **({"json": body} if body else {}))
    assert r.status_code == 401, f"{method.upper()} {path} answered {r.status_code} without a session"


@pytest.mark.parametrize("role", auth.ROLES)
def test_each_role_reaches_exactly_its_endpoints(role):
    if role != "admin":
        auth.add_user("boss@x.com", "Boss", "admin", PW)     # so the console is not open
    client = _signed_in(role)
    for method, path, body, needs in ENDPOINTS:
        status = getattr(client, method)(path, **({"json": body} if body else {})).status_code
        allowed = auth.ROLES.index(role) >= auth.ROLES.index(needs)
        assert (status != 403) == allowed, f"{role} on {method.upper()} {path}: {status}"


def test_the_session_cookie_cannot_be_read_by_scripts_or_sent_cross_site():
    auth.add_user("a@x.com", "A", "agent", PW)
    r = _client().post("/api/login", json={"email": "a@x.com", "password": PW})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert r.json()["user"]["role"] == "agent"


def test_signing_out_ends_the_session():
    client = _signed_in("agent", "a@x.com")
    assert client.get("/api/cases").status_code == 200
    client.post("/api/logout")
    assert client.get("/api/cases").status_code == 401


def test_a_review_is_recorded_under_the_signed_in_person(monkeypatch):
    seen = {}

    async def resume(case_id, payload):
        seen.update(payload)
        return {"pending": None, "state": {}}

    monkeypatch.setattr(server, "resume_case", resume)
    client = _signed_in("agent", "asha@x.com")
    r = client.post("/api/cases/c1/review", json={"decision": "approve", "reviewer": "the CEO"})
    assert r.status_code == 200
    assert seen["reviewer"] == "Agent Person"


def test_an_admin_manages_accounts_and_cannot_lock_everyone_out():
    admin = _signed_in("admin", "boss@x.com")
    r = admin.post("/api/users", json={"email": "new@x.com", "name": "New", "role": "agent", "password": PW})
    assert r.status_code == 200
    new_id = r.json()["id"]
    assert admin.patch(f"/api/users/{new_id}", json={"role": "lead"}).json()["role"] == "lead"
    me = admin.get("/api/me").json()["user"]["id"]
    r = admin.patch(f"/api/users/{me}", json={"disabled": True})
    assert r.status_code == 400 and "last active admin" in r.json()["detail"]


def test_no_cross_origin_requests_are_invited():
    r = _client().get("/api/me", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
