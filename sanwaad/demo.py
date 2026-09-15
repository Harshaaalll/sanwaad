"""End-to-end walkthrough on the mock feed.

    python -m sanwaad.demo

Runs without any API key. With GOOGLE_API_KEY set it uses real Gemini for
triage, drafting and grounding, and the printed cost stops being zero.

The demo resets the listener's seen-store and the pattern agent's window
before it starts, so a second run shows the same thing as the first. Both are
memory that accumulates on purpose in production and would make a demo lie.
"""

from __future__ import annotations

import asyncio
import sys

import json

from .listener import SEEN_PATH, Listener
from .llm import is_offline
from .pattern import MEMORY_PATH
from .pipeline import resume_case, run_case
from .tools import registry as tool_registry
from .tools.ledger import BACKEND

BOLD, DIM, GREEN, YELLOW, RED, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m",
)

TIER_COLOUR = {"crisis": RED, "priority": YELLOW, "routine": GREEN, "ignore": DIM}


def rule(title: str = "") -> None:
    print(f"\n{DIM}{'─' * 78}{RESET}")
    if title:
        print(f"{BOLD}{title}{RESET}")


def _reset_memory() -> None:
    for path in (MEMORY_PATH, SEEN_PATH, tool_registry.AUDIT_PATH):
        path.unlink(missing_ok=True)
    BACKEND.reset()


async def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # a GOOGLE_API_KEY in .env switches the demo to real models
    except ImportError:
        pass
    if is_offline():
        print(f"{YELLOW}No GOOGLE_API_KEY — running the deterministic offline stub. "
              f"The listener, the judge, the pattern agent, retrieval, gating and "
              f"the receipts are all real; only the model calls are faked.{RESET}")

    _reset_memory()

    # 1 -----------------------------------------------------------------
    rule("1. LISTEN — every comment, tagged or not")
    heard = await Listener(channels=["mock"]).poll()
    print(f"  {len(heard.complaints)} new items, {heard.untagged} of which never "
          f"tagged the brand")
    print(f"  {DIM}A mentions-based queue would have seen {len(heard.complaints) - heard.untagged}"
          f" of these.{RESET}")

    # 2 -----------------------------------------------------------------
    rule("2. JUDGE + PATTERN — who is speaking, and how many of them")
    print(f"  {DIM}{'author':<20}{'reads as':<10}{'auth':>5}{'reach':>8}"
          f"{'pattern':>9}{'n':>3}  decision{RESET}")

    results = []
    for complaint in heard.complaints:
        out = await run_case(complaint)
        state = out["state"]
        verdict = state.get("verdict") or {}
        pattern = state.get("pattern") or {}
        priority = state.get("priority") or {}
        tier = priority.get("tier", "?")
        colour = TIER_COLOUR.get(tier, "")
        level = pattern.get("level", "none")
        # Pad before colouring: an ANSI escape counts toward an f-string's
        # field width but occupies no columns, so a coloured cell silently
        # knocks the rest of the row out of alignment.
        level_cell = f"{level:>9}"
        if level == "crisis":
            level_cell = f"{RED}{level_cell}{RESET}"
        print(f"  {complaint.author:<20}{verdict.get('author_class', '?'):<10}"
              f"{verdict.get('authenticity', 0):>5.2f}{verdict.get('reach', 0):>8,}"
              f"{level_cell}{pattern.get('cluster_size', 1):>3}  "
              f"{colour}{tier:<8}{RESET}"
              f"{DIM}{(priority.get('reasons') or [''])[0][:34]}{RESET}")
        results.append((complaint, out))

    ignored = [r for r in results if (r[1]["state"].get("priority") or {}).get("tier") == "ignore"]
    crisis = [r for r in results if (r[1]["state"].get("pattern") or {}).get("level") == "crisis"]
    print(f"\n  {len(ignored)} logged without a drafted reply; "
          f"{len(crisis)} case(s) inside a detected incident.")
    if crisis:
        pattern = crisis[-1][1]["state"]["pattern"]
        print(f"  {RED}CRISIS{RESET} — {pattern['cluster_size']} distinct people reported "
              f"the same thing within {pattern['window_minutes']} minutes "
              f"({pattern['velocity_per_hour']}/hr).")
        print(f"  {DIM}No single one of those comments says 'there is an outage'. "
              f"Only the set does.{RESET}")

    # 3 -----------------------------------------------------------------
    target, out = results[1]   # the Devanagari frozen-wallet case
    case_id, state = out["case_id"], out["state"]
    triage = state["triage"]

    rule(f"3. GHOSTWRITER — {target.author}")
    print(f"  {target.text}\n")
    print(f"  judge     → {state['verdict']['author_class']}, "
          f"authenticity {state['verdict']['authenticity']}")
    print(f"  triage    → {triage['category']}, severity {triage['severity']}, "
          f"reply in {triage['language']}")
    print(f"  retrieved → {[c['clause_id'] for c in state['citations'][:5]]}")
    print(f"\n  draft:\n    {state['draft']['text']}")
    print(f"  cites: {state['draft']['citations']}")

    rule("4. REVIEW GATE")
    if out["pending"]:
        print(f"  {RED}HELD{RESET} — {out['pending']['reason']}")
        print(f"  {DIM}A human approves in the console; the graph stays parked in "
              f"SQLite until they do.{RESET}")
        out = await resume_case(case_id, {
            "decision": "approve", "reviewer": "demo", "note": "approved in demo",
        })
        state = out["state"]
    else:
        print(f"  {GREEN}AUTO-APPROVED{RESET} — {state['review']['note']}")

    rule("5. ESCALATION → VOICE")
    esc = state.get("escalation") or {}
    if esc.get("needed"):
        for r in esc["reasons"]:
            print(f"  → {r}")
        print(f"  {DIM}Browser callback at /call/{case_id}. Simulating the call…{RESET}")
        out = await resume_case(case_id, {
            "happened": True, "channel": "webrtc", "duration_s": 186.0, "resolved": True,
            "summary": "Explained the freeze reason and the re-KYC path.",
            "citations_used": ["KYC-01", "KYC-02", "BV-03"],
        })
        state = out["state"]
    else:
        print("  none needed")

    rule("6. CLOSURE")
    closure = state["closure"]
    con = closure["consistency"]
    print(f"  resolved       {closure['resolved']}")
    print(f"  cost           ₹{closure['total_cost_inr']:.4f}  "
          f"({closure['llm_calls']} model calls)")
    print(f"  human baseline ₹18–85 for the same work")
    print(f"\n  consistency    "
          f"{GREEN + 'channels agree' + RESET if con['consistent'] else RED + 'DIVERGED' + RESET}")
    print(f"    shared     {con['shared_clauses']}")
    print(f"    text only  {con['text_only']}")
    print(f"    voice only {con['voice_only']}")

    # 7 -----------------------------------------------------------------
    k_complaint, k_out = next((c, o) for c, o in results if c.author == "u/karthik_rn")
    rule(f"7. PLAN → VALIDATE → APPROVE → EXECUTE — {k_complaint.author}")
    print(f"  {k_complaint.text}\n")
    for item in k_out["state"].get("actions", []):
        p, v = item["proposal"], item["validation"]
        head = p["kind"] + (f" {p['reference']} ₹{p['amount_inr']:,.0f}" if p.get("reference") else "")
        print(f"  {BOLD}{head}{RESET}  {DIM}{p['risk']} · proposed by {p['proposed_by']}{RESET}")
        for c in v["checks"]:
            mark = f"{GREEN}✓{RESET}" if c["passed"] else f"{RED}✗{RESET}"
            print(f"    {mark} {c['name']:<22}{DIM}{c['detail']}{RESET}")
    if k_out["pending"]:
        print(f"\n  {RED}HELD{RESET} — {k_out['pending']['reason']}")
        print(f"  {DIM}The model proposed it and code validated it. A person now approves the "
              f"reply and each money action separately, bound to the exact reference and amount.{RESET}")
        approve = {i["proposal"]["id"]: "approve" for i in k_out["state"].get("actions", [])
                   if i["proposal"]["risk"] == "write_high" and i["validation"]["ok"]}
        k_out = await resume_case(k_out["case_id"], {
            "decision": "approve", "reviewer": "demo", "note": "approved in demo", "actions": approve,
        })
    for r in k_out["state"].get("action_results", []):
        detail = r.get("output") or r.get("detail") or ""
        print(f"  → {r['kind']}: {BOLD}{r['status']}{RESET}  {DIM}{detail}{RESET}")

    # 8 -----------------------------------------------------------------
    rule("8. EVERY TOOL CALL, AUDITED")
    audit = []
    if tool_registry.AUDIT_PATH.exists():
        audit = [json.loads(line) for line in tool_registry.AUDIT_PATH.read_text().splitlines()]
    mine = [a for a in audit if a.get("trace_id") == k_out["case_id"]]
    for a in mine:
        who = ("human" if a["human_approved"] else "policy") if a["human_approved"] is not None else "—"
        outcome = f"{GREEN}ok{RESET}" if a["ok"] else f"{RED}{a['error_code']}{RESET}"
        print(f"  {a['agent']:<8}{a['tool']:<20}{a['risk']:<12}approved by {who:<7} {outcome}")
    print(f"  {DIM}{len(audit)} tool calls across the whole feed, every one with its agent, "
          f"arguments (redacted) and approval.{RESET}")

    rule("TIMELINE")
    for e in state["events"]:
        print(f"  {e['stage']:<13} {e['message']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
