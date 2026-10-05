"""Deploy the review console to a free Hugging Face Space (Docker).

    hf auth login                                   # once, on this machine
    python scripts/deploy_hf_space.py --space <you>/sanwaad --set-admin
    python scripts/deploy_hf_space.py --space <you>/sanwaad   # later updates

Hugging Face builds the image on its own machines, so nothing is built
locally; only the source (a few MB) is uploaded. The free tier's disk is wiped
on every restart, so cases and accounts do not persist there. --set-admin
stores SANWAAD_ADMIN_EMAIL / _PASSWORD / _NAME as Space *secrets* (prompted,
never echoed or written to a file); the server creates that admin at start
whenever no account exists, so sign-in stays required across restarts.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only what the image needs. Never the venv, git history, local runtime data,
# the developer's .env, or Claude Code settings.
IGNORE = [".venv/*", ".git/*", "**/__pycache__/*", "*.pyc", ".env", ".claude/*", "sanwaad/data/*",
          ".pytest_cache/*", ".ruff_cache/*", "tests/*", "HANDOFF.md"]

SPACE_README = """---
title: Sanwaad
emoji: 🗣️
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7870
pinned: false
short_description: Multi-agent complaint resolution with a human in the loop
---

# संवाद Sanwaad — review console

A multi-agent system that turns public complaints into resolved cases: triage,
grounded replies, money moves a person approves, and a console that explains
every decision. This Space runs the console with offline model stubs (no API
key) on sample complaints. Sign-in is required.

Source, docs and tests: https://github.com/Harshaaalll/sanwaad
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--space", required=True, help="<user or org>/<space name>")
    ap.add_argument("--set-admin", action="store_true", help="prompt for the admin sign-in and store it as Space secrets")
    args = ap.parse_args(argv)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("needs huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    if args.set_admin:
        email = input("Admin email: ").strip()
        name = input("Admin name: ").strip() or "Admin"
        password = getpass.getpass("Admin password (10+ characters): ")
        if len(password) < 10 or password != getpass.getpass("Repeat it: "):
            print("password too short or did not match; nothing stored", file=sys.stderr)
            return 2
        for key, value in (("SANWAAD_ADMIN_EMAIL", email), ("SANWAAD_ADMIN_NAME", name),
                           ("SANWAAD_ADMIN_PASSWORD", password)):
            api.add_space_secret(args.space, key, value)
        print("admin sign-in stored as Space secrets")

    api.upload_folder(folder_path=str(ROOT), repo_id=args.space, repo_type="space",
                      ignore_patterns=IGNORE, commit_message="Deploy Sanwaad console")
    # The data directory must exist in the image; only its placeholder ships.
    api.upload_file(path_or_fileobj=str(ROOT / "sanwaad/data/.gitkeep"), path_in_repo="sanwaad/data/.gitkeep",
                    repo_id=args.space, repo_type="space")
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=args.space, repo_type="space", commit_message="Space README")
    owner, name = args.space.split("/", 1)
    print(f"Deployed. Building on Hugging Face now (a few minutes):\n"
          f"  page: https://huggingface.co/spaces/{args.space}\n"
          f"  app:  https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
