"""Accounts, roles and sessions for the review console.

Three roles, each a superset of the one before:

  agent  works the case queue: reads cases, approves, edits, rejects, calls back
  lead   + the overview, model comparison, loading complaints, viewing policy
  admin  + changing policy and managing accounts

Built-in accounts first, single sign-on later: `authenticate` is the one place
a person is matched to a user row, so an SSO login only has to end by calling
`start_session` with the same user.

With no accounts at all the console runs open, as it always has, so the demo
and a first install work; the moment the first account exists, every API call
needs a session. The first admin is created from the command line, so no
password ever sits in an environment file:

    python -m sanwaad.auth add-user --email you@company.com --name "Your Name" --role admin

Passwords are scrypt hashes with a per-user salt. A session is a random token
in an HttpOnly cookie; only its SHA-256 is stored, so a copied database cannot
be used to log in.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import secrets
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .config import DATA_DIR

USERS_PATH = DATA_DIR / "users.sqlite"
ROLES = ("agent", "lead", "admin")
SESSION_HOURS = 12
COOKIE = "sanwaad_session"
# Lockout: this many wrong passwords for one email inside the window.
MAX_FAILURES, FAILURE_WINDOW_S = 5, 15 * 60
MIN_PASSWORD = 10

_failures: dict[str, list[float]] = {}


class AuthError(ValueError):
    """A refused account operation; the message is safe to show the person."""


@dataclass(frozen=True)
class User:
    id: int
    email: str
    name: str
    role: str
    disabled: bool = False

    def can(self, role: str) -> bool:
        return ROLES.index(self.role) >= ROLES.index(role)


def _db() -> sqlite3.Connection:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(USERS_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('agent', 'lead', 'admin')),
        pw_hash TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
        expires_at REAL NOT NULL)""")
    return conn


def _user(row) -> User:
    return User(id=row[0], email=row[1], name=row[2], role=row[3], disabled=bool(row[4]))


# --- passwords -------------------------------------------------------------------

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1)
    return f"scrypt$16384$8$1${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), digest)


# A real hash to compare against when the email does not exist, so a wrong
# email and a wrong password take the same time and reveal nothing.
_DUMMY = hash_password(secrets.token_hex(8))


# --- accounts ----------------------------------------------------------------------

def has_users() -> bool:
    if not USERS_PATH.exists():
        return False
    with _db() as conn:
        return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def list_users() -> list[User]:
    with _db() as conn:
        return [_user(r) for r in conn.execute(
            "SELECT id, email, name, role, disabled FROM users ORDER BY role DESC, name")]


def add_user(email: str, name: str, role: str, password: str) -> User:
    email, name = (email or "").strip().lower(), (name or "").strip()
    if "@" not in email:
        raise AuthError("enter a valid email address")
    if not name:
        raise AuthError("enter the person's name")
    if role not in ROLES:
        raise AuthError(f"role must be one of {', '.join(ROLES)}")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"use a password of at least {MIN_PASSWORD} characters")
    with _db() as conn:
        try:
            cur = conn.execute("INSERT INTO users (email, name, role, pw_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                               (email, name, role, hash_password(password), datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError:
            raise AuthError(f"{email} already has an account") from None
        return User(id=cur.lastrowid, email=email, name=name, role=role)


def update_user(user_id: int, *, role: Optional[str] = None, disabled: Optional[bool] = None,
                password: Optional[str] = None) -> User:
    with _db() as conn:
        row = conn.execute("SELECT id, email, name, role, disabled FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise AuthError("no such account")
        user = _user(row)
        if role is not None and role not in ROLES:
            raise AuthError(f"role must be one of {', '.join(ROLES)}")
        losing_admin = user.role == "admin" and not user.disabled and (
            (role is not None and role != "admin") or disabled)
        if losing_admin:
            admins = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND disabled = 0").fetchone()[0]
            if admins <= 1:
                raise AuthError("this is the last active admin; make someone else admin first")
        if password is not None:
            if len(password) < MIN_PASSWORD:
                raise AuthError(f"use a password of at least {MIN_PASSWORD} characters")
            conn.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (hash_password(password), user_id))
        if role is not None:
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        if disabled is not None:
            conn.execute("UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), user_id))
        if disabled or password is not None:
            # A disabled account or a changed password ends every open session.
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        row = conn.execute("SELECT id, email, name, role, disabled FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user(row)


# --- sessions -------------------------------------------------------------------

def authenticate(email: str, password: str) -> User:
    """The one place a person is matched to an account."""
    email = (email or "").strip().lower()
    now = time.time()
    recent = [t for t in _failures.get(email, []) if now - t < FAILURE_WINDOW_S]
    _failures[email] = recent
    if len(recent) >= MAX_FAILURES:
        raise AuthError("too many failed attempts; try again in 15 minutes")
    with _db() as conn:
        row = conn.execute("SELECT id, email, name, role, disabled, pw_hash FROM users WHERE email = ?",
                           (email,)).fetchone()
    ok = verify_password(password or "", row[5] if row else _DUMMY)
    if not row or not ok or row[4]:
        _failures[email].append(now)
        raise AuthError("wrong email or password")
    _failures.pop(email, None)
    return _user(row)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(user: User) -> str:
    token = secrets.token_urlsafe(32)
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                     (_token_hash(token), user.id, time.time() + SESSION_HOURS * 3600))
    return token


def user_for(token: Optional[str]) -> Optional[User]:
    if not token or not USERS_PATH.exists():
        return None
    with _db() as conn:
        row = conn.execute(
            """SELECT u.id, u.email, u.name, u.role, u.disabled FROM sessions s JOIN users u ON u.id = s.user_id
               WHERE s.token_hash = ? AND s.expires_at > ? AND u.disabled = 0""",
            (_token_hash(token), time.time())).fetchone()
    return _user(row) if row else None


def end_session(token: Optional[str]) -> None:
    if token and USERS_PATH.exists():
        with _db() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def bootstrap_admin_from_env() -> Optional[User]:
    """Create the first admin from SANWAAD_ADMIN_EMAIL / SANWAAD_ADMIN_PASSWORD.

    For hosts with no shell and no persistent disk (a free Hugging Face Space
    is wiped on every restart), where the CLI cannot run. Set both in the
    host's secret store, never in a committed file. Does nothing once any
    account exists, so it can never overwrite or add to a real team.
    """
    import os

    email, password = os.getenv("SANWAAD_ADMIN_EMAIL"), os.getenv("SANWAAD_ADMIN_PASSWORD")
    if not (email and password) or has_users():
        return None
    return add_user(email, os.getenv("SANWAAD_ADMIN_NAME") or "Admin", "admin", password)


# --- command line -------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Manage Sanwaad console accounts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add-user", help="create an account (prompts for the password)")
    add.add_argument("--email", required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--role", choices=ROLES, default="agent")
    sub.add_parser("list", help="list accounts")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for u in list_users():
            print(f"{u.role:6} {u.email:32} {u.name}{'  (disabled)' if u.disabled else ''}")
        return 0
    password = getpass.getpass("Password (10+ characters): ")
    if password != getpass.getpass("Repeat it: "):
        print("passwords did not match", file=sys.stderr)
        return 2
    try:
        user = add_user(args.email, args.name, args.role, password)
    except AuthError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"created {user.role} {user.email}. Sign in at the console.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
