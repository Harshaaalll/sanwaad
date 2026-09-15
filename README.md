# संवाद Sanwaad

**A multi-agent system that turns public complaints into resolved cases.** It
catches the complaints nobody tagged, works out who is speaking and whether
they're part of an outage, drafts a reply grounded in written policy, and
proposes the actual fix. A refund is only ever *suggested* by the model: code
checks it against the ledger, a person approves the exact amount, and the
executor checks again before anything moves.

Built with LangGraph, Gemini, local hybrid RAG (ONNX embeddings + BM25), typed
tools with approvals, and trace-level evaluation. It runs end to end with no
API key.

```bash
python -m sanwaad.demo                # the whole system, offline
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
pytest tests/ -q                           # 147 tests, no API key
```

The first run downloads a ~470 MB multilingual embedding model; later starts are
instant. Add `GOOGLE_API_KEY` to `.env` for real Gemini calls. The live browser
voice leg is optional: `pip install -r requirements-voice.txt`, plus Sarvam and
Murf keys.

---

## Learn the design

[`sanwaad/DESIGN.md`](sanwaad/DESIGN.md) is a twelve-lesson course on agentic AI
system design, taught through this codebase: model routing, tools, memory and
state, orchestration, evaluation, approvals, reliability, cost and latency,
context and RAG, observability, security and privacy. Each lesson points to the
code, explains the decision behind it, and ends with a command to run and a
question to check yourself.

---

## Layout

```
sanwaad/
  graph/          the LangGraph state machine: state, nodes, edges
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
  evals/          golden set, retrieval eval, trajectory eval, harness
  voice/          voice brief and the WebRTC agent
  api/            FastAPI, review console, call page
  policy/         the knowledge base: plain markdown clauses
  DESIGN.md       the course
tests/            147 tests
```

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
