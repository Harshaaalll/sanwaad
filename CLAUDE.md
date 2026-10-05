# Sanwaad

A multi-agent system that turns public complaints about NimbusPay (a fictional
UPI wallet) into resolved cases. LangGraph pipeline, FastAPI console, local
hybrid RAG, Gemini (offline stubs without a key). Python 3.12, `.venv/`.

## Commands

```bash
.venv/bin/python -m pytest tests -q          # full suite, offline, no key
node --test tests/js/*.test.js                         # console logic only (also run by pytest)
.venv/bin/ruff check sanwaad tests           # lint (CI pins ruff 0.15.4)
.venv/bin/python -m sanwaad.api.server       # console on :7870
.venv/bin/python -m sanwaad.demo             # one case end to end
.venv/bin/python -m sanwaad.evals.triage_compare --data file.csv
```

A test gate runs automatically (`.claude/hooks/gate.py`): ruff + related
tests after each Python edit, the full suite before a turn ends, and again on
`git commit`. A change is not done until it passes.

## Where things are

- `sanwaad/graph/nodes.py`: every pipeline step; `graph.py` wires the order.
- `sanwaad/config.py`: every threshold (review, judge, crisis, action policy).
- `sanwaad/auth.py`: accounts, roles (agent < lead < admin) and sessions; `require(role)` guards every API route.
- `sanwaad/policy_store.py`: the only place a policy may change at runtime (bounded, logged, revertible).
- `sanwaad/pipeline.py`: run / resume / list cases over one shared SQLite checkpointer.
- `sanwaad/triage_backends.py`: gemini | laya | jev triage behind one interface.
- `sanwaad/api/server.py` + `api/static/console.html`: the review console (vanilla JS).
- `sanwaad/DESIGN.md`: the reasoning behind the architecture, as lessons.

## Rules this codebase does not bend

- **The model suggests, code decides.** Money moves, escalation and auto-post
  are decided by code with named checks, never by a prompt. `initiate_reversal`
  is never auto-approvable.
- A grounding check that could not run is *unverified*, never "grounded".
- A degraded or offline result is never cached or reported as a real one.
- Customer text reaches prompts only inside `untrusted(...)`; identifiers are
  redacted before anything is logged or saved.
- Nothing the console or an API returns claims more than the code measured.

## How to work here

1. **Think before coding.** State assumptions. If a request has two readings,
   ask rather than pick one silently. Say so when something is unclear.
2. **Simplest thing that works.** No speculative options, abstractions or
   dependencies nobody asked for. Stdlib and existing helpers first.
3. **Surgical changes.** Touch only what the task needs. Match the surrounding
   style. Don't reformat, rename or "improve" adjacent code; mention it instead.
4. **Work to a checkable goal.** Turn the task into something verifiable (a
   failing test, a reproduced bug, a measured number) and loop until it holds.
   Report what was verified and what was not.

## Style

- Comments and docstrings explain *why*, often as the failure that motivated
  the code. Match that; don't narrate what the code does.
- Commit messages: an imperative subject saying what now holds true
  ("Stop X doing Y"), then a body explaining the why.
- Tests name the behaviour they pin (`test_a_burst_of_cases_all_reach_the_checkpoint`).

## Environment notes

- The laptop has ~15 GB RAM and often swaps; the full suite takes 1-3 minutes.
- The Claude-in-Chrome browser cannot reach localhost here: check the console
  through curl and the API, or by rendering its JS in node.
- Laya (`requirements-models.txt`) is installed but parked; see README.
