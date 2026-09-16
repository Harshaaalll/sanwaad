# संवाद Sanwaad

**A multi-agent system that turns public complaints into resolved cases.** It
catches the complaints nobody tagged, works out who is speaking and whether
they're part of an outage, drafts a reply grounded in written policy, and
proposes the actual fix. A refund is only ever *suggested* by the model: code
checks it against the ledger, a person approves the exact amount, and the
executor checks again before anything moves.

Built with LangGraph, Gemini, local hybrid RAG (ONNX embeddings + BM25), typed
tools with approvals, trace-level evaluation, and a loop-engineered support
agent that runs act → observe → verify → retry under explicit budgets. It runs
end to end with no API key.

```bash
python -m sanwaad.demo                # the whole system, offline
python -m sanwaad.loop                # the support agent loop, pass by pass
python -m sanwaad.evals.trajectory    # grade every step of 15 scenarios
```

---

## What it does

| Agent | Question it answers | Why a single comment can't answer it |
|---|---|---|
| **Listener** | Is anyone talking about us, tagged or not? | A mentions inbox only sees the people who @ you |
| **Pattern** | Is this the ninth report of the same thing? | No single comment says "there is an outage" |
| **Judge** | Customer, audience, troll or bot? | The words can be identical; the accounts aren't |
| **Ghostwriter** | What do we say that is true, and what do we fix? | The fix needs the ledger, the policy and a person |

```
every platform, tagged or not
        │
        ▼
   listener ──────────────►  dedupe, mark untagged
        │
        ▼
    triage                   cheapest model · a severity-5 item is never dropped
        │
        ▼
  pattern ──► judge ──► prioritise     how many? who? → one queue decision
        │                               (the only place a case closes unanswered)
        ▼
  retrieve ◄── policy index            33 clauses · local embeddings + BM25 · ₹0
        │
        ▼
 ghostwriter ──► ground_check ──┐       every claim must trace to a clause
        ▲                       │       not backed → rewrite (max 2) → a person
        └───────────────────────┘
        │
        ▼
      plan ──► lookup_transaction       propose the fix · code runs 9 checks
        │
        ▼
  review_gate ── interrupt() ──►  a person approves the reply and each money move
        │                         (a detected crisis always stops here)
        ▼
  publish ──► act ──► escalation ──► voice (WebRTC, same clauses) ──► close
               └─ re-validates, then executes approved actions through tools
                                                                       │
                                     consistency receipt · cost per case ◄┘
```

---

## Engineering highlights

**The model suggests, code decides.** For a ₹640 double debit, `plan` proposes
reversing one transaction. `validate_action` runs nine named checks against
the ledger: it exists, belongs to the complainant, identity is verified, the
amount matches, policy says it's owed (and which clause), it's under the
₹25,000 ceiling, and it isn't already reversed. The console shows every check.
A person's approval is bound to a digest of the exact arguments, and the
executor re-validates before calling the tool, because the ledger can change
while a case waits. Money can never be auto-approved.

**Tools designed like APIs.** Four tools on a risk ladder: read, low-risk write,
high-risk write. Every call goes through one registry, which checks that the
tool exists, that this agent may call it (least privilege), that the inputs and
outputs are valid, and that an approval is present when required. It retries
only when repeating the call can't repeat the effect, and writes a redacted
audit record. Contracts export as MCP tool definitions.

**The model is an unreliable dependency.** Every call has a timeout, a token cap
and a fallback model. Output that fails its schema is retried once with the
problem named. A step that still fails runs a safe default marked `degraded`,
or stops. A grounding check that couldn't run is treated as *unverified*, never
"grounded".

**Trace-level evaluation.** 15 scenarios, including prompt injection, someone
else's transaction, a ledger outage, a refund inside the auto-reversal window,
a live incident and a troll, are graded at every step, not just on the final
reply. Failures are attributed to the first wrong step, and three safety
invariants are checked on every run.

**Explicit memory, minimal context.** Eight stores chosen by access pattern,
each with retention and a rule for whether a model may see it. Customer text,
model-written summaries and tool output reach prompts only inside
`<untrusted>` blocks that can't be closed from inside, and identifiers are
removed before any prompt is built.

**Enforced agent contracts.** Each agent declares the state it writes and the
tools it may call. The graph rejects a node that writes outside its contract,
and the registry rejects a tool call outside its list.

---

## Loop engineering

The case pipeline above is code choosing each step. Private support
conversations ("where's my refund?") are different: the model chooses each
action, and Sanwaad engineers the loop it runs inside.

```
customer message
      │
      ▼
  ┌─► model decides ONE action ──► harness runs it (registry: contracts, retries, audit)
  │                                         │
  │                                         ▼
  └── context window ◄── observation, shaped: filtered, capped, redacted
      budgeted: compaction,
      external memory, sub-agents
      │
      final answer ──► cheap verifiers ──► pass: done
                              │             rejected: feedback, try again (bounded)
      money or legal language ──────────────► needs_human
      budget · pass cap · stall · timeout ──► stop
```

- **Stopping conditions.** Seven stop reasons are checked before every pass,
  whichever fires first. A runaway agent is stopped at pass 3 by stall
  detection; one that never repeats itself is stopped by a budget.
- **Loop economics.** Every pass is priced, offline as an estimate, so loop
  length always shows up as money.
- **Context rot, managed.** Tool results are shaped before they enter; facts go
  to a scratchpad that outlives compaction; old steps compact under a token
  budget; a policy sub-agent answers in its own clean context and returns one
  line (407 tokens consumed, 34 returned).
- **Verification asymmetry.** Cheap verifiers sit inside the loop and turn a bad
  draft into feedback: the agent's remembered "5 to 7 days" is rejected, and it
  answers "3 working days" from policy. Money moves and legal language have no
  cheap check, so they stop with `needs_human`.
- **MINT.** Minimal Intelligence, Necessary Tools: six rungs, each adding one
  layer only after the one below showed a measured need. `check_layering`
  refuses a configuration that skips one.
- **The three nested loops.** Recorded runs feed a system-level trace report;
  runs that stalled or needed correcting become draft scenarios for human
  review; traffic splits deterministically for A/B tests that refuse to call a
  winner on too few runs.

---

## Results

### Trajectory eval: 15 scenarios, every step graded

_Mode: offline, deterministic stubs, no API key_

| Metric | Value |
|---|---|
| Scenarios passing every step | 15 / 15 |
| Safety violations | 0 |
| Intent accuracy | 1 |
| Author accuracy | 1 |
| Retrieval hit rate | 1 |
| Action decision accuracy | 1 |
| Approval gate accuracy | 1 |
| Escalation accuracy | 1 |
| Tool-call success rate (outages injected) | 0.973 |
| Invalid schema rate | n/a |
| Mean cost per case (₹) | 0 |
| Case latency p50 / p95 (ms) | 1349.8 / 1911.6 |

Offline, the model calls are deterministic stand-ins, so this table tests the
system *around* the models: routing, validation, approvals, degradation and the
safety invariants. It says nothing about model quality. The same eval runs
unchanged against Gemini once `GOOGLE_API_KEY` is in `.env`; regenerate this
table with `python -m sanwaad.evals.trajectory --markdown`.

On its first run this eval caught a real bug: "a good place for filter
**coffee**" was triaged as a billing complaint, because the keyword `fee`
matched inside the word.

### Retrieval: the golden set, 19 cases in English, Hindi and Hinglish

These numbers are real offline, because retrieval uses local embeddings and no
language model.

| Strategy | strict@5 | recall@5 | MRR |
|---|---|---|---|
| Dense only | 0.89 | 0.89 | 0.61 |
| Dense + lexical blend | 0.89 | 0.89 | 0.61 |
| **RRF (dense + BM25), used** | **1.00** | **1.00** | **0.75** |
| RRF, dense-weighted 2:1 | 0.95 | 0.95 | 0.71 |
| Blend, no category prior | 0.84 | 0.84 | 0.53 |

`strict@5` is 1.0 only when *every* clause a correct reply needs is retrieved.
Removing the category prior drops Hinglish to 0.50.

### Loop eval: what each MINT layer buys

13 end-to-end scenarios, including a ledger outage, garbage input, an angry
customer, an out-of-policy refund, a runaway agent and a six-part research
question, each run at every rung. Offline, with the scripted policy; costs are
estimates at the loop's model tier.

| | M0 minimal | M1 +tools | M2 +evaluation | M3 +memory | M4 +workflows | M5 +HITL, multi-agent |
|---|---|---|---|---|---|---|
| Scenarios passing | 3/13 | 6/13 | 7/13 | 9/13 | 10/13 | 13/13 |
| Unsafe answers shipped | 0 | 1 | 0 | 0 | 0 | 0 |
| Runs over context budget | 0 | 2 | 2 | 0 | 0 | 0 |
| Mean passes | 1.62 | 3.46 | 3.62 | 3.54 | 3.15 | 2.85 |
| Mean cost per run (₹, est.) | 0.0131 | 0.0491 | 0.0515 | 0.0483 | 0.0390 | 0.0376 |
| Sub-agent tokens kept out | 0 | 0 | 0 | 0 | 0 | 757 |

M1 ships an unsupported timeline, so evaluation is added and unsafe answers
drop to zero. M2 overruns the context window on long tasks (a peak of 1,069
tokens against an 800-token budget), so memory and compaction are added (peak
752). Regenerate with `python -m sanwaad.evals.loop_eval --ladder --markdown`.

---

## Run it

```bash
git clone https://github.com/Harshaaalll/sanwaad && cd sanwaad
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # every key is optional

python -m sanwaad.demo                     # CLI walkthrough
python -m sanwaad.api.server               # review console at http://localhost:7870
python -m sanwaad.evals.trajectory         # 15 scenarios, step by step
python -m sanwaad.evals.retrieval          # retrieval strategies compared
python -m sanwaad.memory                   # what is remembered, where, how long
python -m sanwaad.router                   # which model runs each step
python -m sanwaad.loop                     # the support loop, pass by pass
python -m sanwaad.evals.loop_eval --ladder # the MINT ladder
python -m sanwaad.loop.outer               # the external loop over recorded runs
pytest tests/ -q                           # 184 tests, no API key
```

The first run downloads a ~470 MB multilingual embedding model; later starts are
instant. Add `GOOGLE_API_KEY` to `.env` for real Gemini calls. The live browser
voice leg is optional: `pip install -r requirements-voice.txt`, plus Sarvam and
Murf keys.

---

## Learn the design

[`sanwaad/DESIGN.md`](sanwaad/DESIGN.md) is a twenty-lesson course in two parts,
taught through this codebase. Part I covers the building blocks of an agentic
system: model routing, tools, memory and state, orchestration, evaluation,
approvals, reliability, cost and latency, context and RAG, observability,
security and privacy. Part II covers loop engineering: the loop primitive,
stopping conditions and loop economics, harness engineering and system-level
evaluation, context rot, verification asymmetry, MINT, the three nested loops
and the four agentic design patterns. Each lesson points to the
code, explains the decision behind it, and ends with a command to run and a
question to check yourself.

---

## Layout

```
sanwaad/
  graph/          the LangGraph state machine: state, nodes, edges
  loop/           the support agent loop: kernel, budgets, window, verifiers,
                  policies, sub-agent, MINT ladder, outer loops
  agents.py       agent contracts: role, writes, tools
  listener.py     multi-channel polling, dedupe, untagged mentions
  pattern.py      cross-case window, clustering, crisis detection
  judge.py        author scoring and the reply-worthy rule
  tools/          tool contracts, registry, built-in tools, mock ledger
  actions.py      proposed fixes and the nine checks they must pass
  llm.py          structured calls, retries, fallbacks, cost accounting
  router.py       per-step model, token cap, timeout, fallback
  context.py      minimal, trust-separated prompt context
  memory.py       the memory map and retention
  rag/            clause index, local embeddings, BM25 + RRF, agentic retrieval
  guardrails.py   PII redaction, money-promise and injection checks
  evals/          golden set, retrieval, trajectory and loop evals, harness
  voice/          voice brief and the WebRTC agent
  api/            FastAPI, review console, call page
  policy/         the knowledge base: plain markdown clauses
  DESIGN.md       the course
tests/            184 tests
```

---

## Project log

| Date | Commit | What landed |
|---|---|---|
| 2026-09-15 | `1a3af80` | Sanwaad as a standalone repo: the four agents (listener, pattern, judge, ghostwriter), the LangGraph case graph, hybrid RAG over the policy index, typed tools with human-approved money actions, trace-level evals, and DESIGN.md Part I (12 lessons) |
| 2026-09-16 | `8fb9400` | Loop engineering: the support agent loop (kernel, stopping conditions, context window, verifiers, policy sub-agent), the MINT ladder, the three nested loops, a 13-scenario loop eval, and DESIGN.md Part II (8 lessons) |

`git log --oneline` for the full history. The repo starts from a clean commit:
earlier exploratory work on speech-to-speech voice agents lives in a separate
private repository.

---

## Status and limits

- **Mock systems of record.** The payments ledger and ticket desk are mocks, and
  the sample feed is a seeded set of 15 realistic comments. The Reddit connector
  reads live data; posting to any real platform is off unless
  `SANWAAD_ALLOW_POSTING=true`.
- **Offline numbers.** The trajectory table above grades the system with stubbed
  model calls. Live-model results will differ; that's the point of running it.
- **Voice leg.** It needs Sarvam and Murf keys, and the test suite doesn't cover it.
- **Retention.** Retention is enforced for file-backed stores. The workflow-state
  checkpoint declares 90 days but doesn't prune yet.
- **Offline loop policy.** Without a key, a scripted policy drives the support
  loop. It reads only what a model would see, but it doesn't wander, so offline
  the ladder shows workflows' value only on the misbehaving-agent scenario; the
  benefit of offering fewer tools needs the live model to measure.
