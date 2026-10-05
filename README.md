# संवाद Sanwaad

[![tests](https://github.com/Harshaaalll/sanwaad/actions/workflows/ci.yml/badge.svg)](https://github.com/Harshaaalll/sanwaad/actions/workflows/ci.yml)

**A multi-agent system that turns public complaints into resolved cases.** It
catches the complaints nobody tagged, works out who is speaking and whether
they're part of an outage, drafts a reply grounded in written policy, and
proposes the actual fix. A refund is only ever *suggested* by the model: code
checks it against the ledger, a person approves the exact amount, and the
executor checks again before anything moves.

Built with LangGraph, Gemini, local hybrid RAG (ONNX embeddings + BM25), typed
tools with approvals, trace-level evaluation, and a loop-engineered support
agent that runs act → observe → verify → retry under explicit budgets. A review
console shows every case's state *and the reasons for it*, an operator overview,
the policy thresholds (editable, on the record), and a comparison of triage
models on your own labelled data. It runs end to end with no API key.

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

**Tools designed like APIs.** Six tools on a risk ladder: read, low-risk write,
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

**Explicit memory, minimal context.** Twelve stores chosen by access pattern,
each with retention and a rule for whether a model may see it. Customer text,
model-written summaries and tool output reach prompts only inside
`<untrusted>` blocks that can't be closed from inside, and identifiers are
removed before any prompt is built.

**Enforced agent contracts.** Each agent declares the state it writes and the
tools it may call. The graph rejects a node that writes outside its contract,
and the registry rejects a tool call outside its list.

**Every decision explains itself.** The console shows why a case is where it
is: the priority score split into its parts, the author's place on the
troll/audience scale with the evidence, every review rule with a ✓ or ✗, and
all five voice-callback triggers, including the ones that didn't fire. Each is
computed by the function that made the decision (the gate *is* the checklist),
so the explanation can't drift from what happened.

**Autonomy is earned, and policy changes are on the record.** A reply type
posts without a person only after reviewers agreed with its drafts often
enough, and loses that the moment they stop. The thresholds behind every
decision are on a Policy page; an admin can change thirteen of them inside
hard bounds, each with a reason, an impact preview on stored cases, and an
append-only, revertible log. The grounding rule and the ban on auto-promised
compensation can't be changed there at all.

**Drafts learn from reviewers.** When a reviewer changes a reply before
approving it, that rewrite (identifiers removed) becomes one of up to three
examples in the next draft prompt for the same category, fenced as untrusted
text that can't override instructions; facts still come only from policy. The
timeline says when examples were in the prompt, and the Overview tracks the
edit rate per category, the number that should fall as drafts improve.

**The triage model is measured, not assumed.** Triage runs behind one interface
with three backends: Gemini, Laya (an open-weight decision model that runs
locally and reports calibrated confidence) and Jev (TypeSafe AI, hosted; not yet
wired). A comparison scores them on labelled complaints; the winner goes live
with a setting, and anything it is unsure of falls back to Gemini.

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
| Case latency p50 / p95 (ms) | 1349.8 / 1911.6 *(this machine)* |

Offline, the model calls are deterministic stand-ins, so this table tests the
system *around* the models: routing, validation, approvals, degradation and the
safety invariants. It says nothing about model quality. Every row reproduces
exactly on a rerun except the latency, which is whatever the machine that ran
it could do — a number worth measuring per run, not worth quoting from a README. The same eval runs
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

## Show it

`DEMO.md` is a tested eight-minute runbook — six commands, no API key, with
what each one prints and what to say over it.

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
python -m sanwaad.delivery                 # what gave up, and why
python -m sanwaad.autonomy                 # what it has earned the right to do alone
python -m sanwaad.missions                 # a second workflow on the same harness
python -m sanwaad.obs                      # per-step p50/p95, budgets, spend
python -m sanwaad.router                   # which model runs each step
python -m sanwaad.loop                     # the support loop, pass by pass
python -m sanwaad.evals.loop_eval --ladder # the MINT ladder
python -m sanwaad.loop.outer               # the external loop over recorded runs
python -m sanwaad.evals.triage_compare     # triage models compared on labelled data
pytest tests/ -q                           # 472 tests, no API key
node --test tests/js/*.test.js                      # the console's own logic (pytest runs it too)
```

### A live link for free (GitHub Codespaces)

The repo carries a codespace config, so a cloud machine sets itself up and
nothing runs on your laptop. On GitHub: **Code → Codespaces → Create codespace
on main**, wait for the setup to finish, then in its terminal:

```bash
python -m sanwaad.auth add-user --email you@company.com --name "You" --role admin
python -m sanwaad.api.server
```

In the **Ports** tab, right-click port 7870 → **Port visibility → Public**,
and share the forwarded `https://…app.github.dev` address; visitors get the
sign-in page. The link works while the codespace runs (it stops after a period
of inactivity; restart it from GitHub). Accounts and cases persist inside the
codespace until you delete it.

A Hugging Face Space also works (`scripts/deploy_hf_space.py --space <you>/sanwaad
--set-admin`), but Docker Spaces now need a PRO subscription.

Or in Docker, where that download already happened at build time:

```bash
docker build -t sanwaad .
docker run --rm -p 7870:7870 sanwaad                 # offline stubs, no key
docker run --rm -p 7870:7870 --env-file .env sanwaad # live models
```

The image bakes in the embedding model and builds the policy index at build
time, so the container reports `/ready` about a second after start rather than a
minute. `/health` is liveness and depends on nothing; `/ready` returns 503 until
the index is loaded, because serving a complaint without retrieval means
answering it ungrounded.

Run it directly and the first run downloads a ~470 MB multilingual embedding
model; later starts are instant. Add `GOOGLE_API_KEY` to `.env` for real Gemini calls. The live browser
voice leg is optional: `pip install -r requirements-voice.txt`, plus Sarvam and
Murf keys.

### The review console

`python -m sanwaad.api.server`, then http://localhost:7870. Four tabs:

- **Cases** — the queue, and for each case: the complaint, the judge's read of
  the author, the grounded draft and the clauses behind it, proposed actions
  with every validation check, the timeline, and *why this case is where it
  is*. Approve, edit or reject here; money actions are approved one by one.
- **Overview** — leads with how many complaints are waiting on a person and
  how long the oldest has waited, then the last 14 days: complaints opened and
  resolved per day, what people are complaining about this week against last,
  first-reply time against a target (`SANWAAD_SLA_FIRST_RESPONSE_MINUTES`,
  default 4 hours), how much posts without a person and how reviewers decided,
  model cost per case, incidents, the autonomy each reply type has earned,
  and what reviewers are teaching it: the edit rate per category and the
  words they most often remove or add.
- **Model comparison** — the latest `triage_compare` run.
- **Policy** — every threshold, as the running server has it.

The header's health strip shows the index, model mode, the live triage
backend, both concurrency pools, open circuits and dead letters.

### Explore any company

The **Explore** tab (team leads and up) searches the Play Store for any company,
pulls its newest 1–3 star reviews (and Reddit posts when `REDDIT_CLIENT_ID` and
`REDDIT_CLIENT_SECRET` are set), classifies them, groups recurring complaints
into themes, and measures how the company itself replies: share answered and
median time to answer. It is insight only: no reply is drafted for a company
that hasn't been onboarded with its own policy. Explore keeps its own store,
so another company's complaints never touch the case queue or the crisis
detector, and reviewer names are never stored. Without `GOOGLE_API_KEY` the
categories come from the keyword stand-in and the report says so.

Onboarding a company as its own workspace, with policy imported from its help
centre and approved by an admin, is the next step.

### Choosing the triage model

Triage is the one model call every comment pays for. `triage_compare` scores
each backend against complaints you have labelled: accuracy, macro-F1, recall
per category, a confusion matrix, calibration, latency and cost. The console's
**Model comparison** tab shows the result.

```bash
pip install -r requirements-models.txt     # only for Laya: CPU torch + a ~650 MB checkpoint
python -m sanwaad.evals.triage_compare --data your_complaints.csv
```

The CSV needs `text` and `category` (see `sanwaad/evals/data/triage_template.csv`).
Backends: **gemini** (generative, the default), **laya** (open-weight decision
model, runs locally, reports calibrated confidence) and **jev** (TypeSafe AI's
hosted decision model; the adapter is waiting on API access). Put the winner in
front of live traffic with `SANWAAD_TRIAGE_BACKEND`. A decision model sets the
labels, the triage tier writes the summary, and any comment it is unsure of
(`SANWAAD_TRIAGE_MIN_CONFIDENCE`) goes to Gemini, with the reason on the case
timeline.

### Changing a policy threshold

The console's **Policy** tab shows every threshold that decides a case's
state. An admin can change thirteen of them: each
inside hard bounds, with a required reason, an impact preview against stored
cases before Apply appears, and an append-only log (`sanwaad/data/policy_audit.jsonl`)
that is replayed on start and supports one-click revert. The grounding rule,
the ban on auto-promised compensation and the autonomy maximum are not
editable there; the reversal ceiling can only be lowered.

### Accounts and roles

Three roles: **agents** work the case queue; **team leads** also see Overview,
Model comparison and policy, and load complaints; **admins** also change policy
and manage accounts in the Team tab. Every decision is recorded under the
signed-in person's name, taken from the session, never from the browser.

With no accounts the console runs open (a banner says so), which keeps the demo
working. Create the first admin on the server, and from then on everyone signs in:

```bash
python -m sanwaad.auth add-user --email you@company.com --name "Your Name" --role admin
```

Passwords are salted scrypt hashes; sessions are HttpOnly, SameSite=Strict
cookies whose tokens are stored only as hashes; five wrong passwords lock an
email out for 15 minutes. Single sign-on (Google, Microsoft) is the planned next
step: `auth.authenticate` is the one place a person is matched to an account.

### Working on it with Claude Code

`CLAUDE.md` gives Claude Code the commands, the rules this codebase does not
bend, and how to work here. `.claude/` adds a test gate: ruff and the related
tests after each Python edit, and the full suite before a turn is allowed to
end. A `/handoff` skill writes a session summary for the next session. The same
gate runs on `git commit` once installed:

```bash
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit   # ruff + pytest before every commit
```

---

## Learn the design

[`sanwaad/DESIGN.md`](sanwaad/DESIGN.md) is a twenty-two-lesson course in two parts,
taught through this codebase. Part I covers the building blocks of an agentic
system: model routing, tools, memory and state, orchestration, evaluation,
approvals, reliability, cost and latency, context and RAG, observability,
security and privacy. Part II covers loop engineering: the loop primitive,
stopping conditions and loop economics, harness engineering and system-level
evaluation, context rot, verification asymmetry, MINT, the three nested loops,
the four agentic design patterns, earned autonomy and agent handoffs. Each lesson points to the
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
  rag/            clause index, local embeddings, BM25 + RRF (agentic retrieval
                  is built but not yet wired into the graph)
  guardrails.py   PII redaction, money-promise and injection checks
  autonomy.py     authority each capability has earned, from real reviews
  triage_backends.py  gemini | laya | jev triage behind one interface
  explain.py      why a case is where it is, from the deciding functions
  overview.py     the operator's counts across cases
  policy_store.py runtime policy changes: bounded, logged, revertible
  handoff.py      agents propose a route; code grants or refuses it
  missions.py     declared workflows — the lead pipeline, on the same harness
  delivery.py     the dead-letter queue: what gave up, and why
  evals/          golden set, retrieval, trajectory, loop and triage-model evals, harness
  voice/          the call brief, hotwords, spoken numbers, the WebRTC agent
  api/            FastAPI, review console, call page
  policy/         the knowledge base: plain markdown clauses
  DESIGN.md       the course
tests/            472 tests
scripts/          pre-commit (the test gate for git)
.claude/          Claude Code settings: the test gate hooks, the /handoff skill
CLAUDE.md         how to work on this repo, for Claude Code
```

---

## Project log

| Date | Commit | What landed |
|---|---|---|
| 2026-09-15 | `1a3af80` | Sanwaad as a standalone repo: the four agents (listener, pattern, judge, ghostwriter), the LangGraph case graph, hybrid RAG over the policy index, typed tools with human-approved money actions, trace-level evals, and DESIGN.md Part I (12 lessons) |
| 2026-09-16 | `8fb9400` | Loop engineering: the support agent loop (kernel, stopping conditions, context window, verifiers, policy sub-agent), the MINT ladder, the three nested loops, a 13-scenario loop eval, and DESIGN.md Part II (8 lessons) |
| 2026-09-21 | `4123af7` | Input-side harness for the voice leg: a spoken-number normaliser (English, Hinglish, Devanagari; Indian scales; declines what it cannot read confidently) and per-call ASR hotwords derived from the complaint and its clauses, with nothing identifying sent |
| 2026-09-23 | `f8b5a99` | Deployable: a Dockerfile that bakes the embedding model and the policy index at build time, and split liveness/readiness probes — `/ready` returns 503 while the index is warming or failed, so an orchestrator can tell "coming up" from "broken" |
| 2026-09-23 | `551ef8c` | A floor under retrying: items that fail three times are dead-lettered with their errors and the customer's words instead of being refetched forever, with `--requeue` to put one back after a fix |
| 2026-09-23 | `2314057` | Latency budgets that something reads: every route declares `max_latency_ms`, the model layer records a breach without aborting the step, and `stage_stats` compares each step's p95 against it via a new `python -m sanwaad.obs` |
| 2026-09-23 | `e4c15e2` | A circuit breaker per tool: five timeouts or upstream errors in a minute and calls fail fast instead of each case re-discovering the outage, with a single half-open probe to recover unattended |
| 2026-09-23 | `fc6399a` | Bounded concurrency at the choke points: cases queue four at a time so a burst does not become a thundering herd, live calls are refused rather than queued, and both pools report their depth on `/ready` |
| 2026-09-23 | `708e5e3` | CI on every push: ruff, the 277 tests, both evals, and a Docker build that boots the image and waits for `/ready` — plus the 17 lint findings that had accumulated, including three `zip()` calls that would truncate silently |
| 2026-09-23 | `1d00dec` | A per-case cost ceiling checked before each model call, degrading the way an outage does so the grounding gate routes it to a person |
| 2026-09-23 | `e2ac06a` | A health strip in the console header: the index, model mode, both pools, any open circuit and any dead letter, red only when something is actually wrong |
| 2026-09-23 | `b8f1512` | Review findings closed: the breaker survives the outage it was written for, the concurrency bound can't disable itself, the dead-letter queue keeps its counters, the number normaliser stops inventing amounts |
| 2026-09-23 | `bc7032b` | Earned autonomy: each reply type moves from shadow to autonomous on measured agreement with reviewers and falls back on disagreement; severity above 3 and money never leave a person (DESIGN Lesson 21) |
| 2026-09-24 | `2adbd74` | Outage hygiene and an audit close-out: no degraded result is cached or reported as real, UPI addresses are redacted, a lost reply goes to the dead-letter queue, four false claims in the docs corrected |
| 2026-09-24 | `b48d4c8` | Missions: agents propose where work goes next and code decides whether they may, on a declared route table with a revisit rule (DESIGN Lesson 22) |
| 2026-09-24 | `cbfe59f` | `DEMO.md`, a tested eight-minute runbook; the call page says why a call can't be placed instead of failing with a 500 |
| 2026-09-29 | `7375811` | A burst of concurrent cases no longer loses checkpoints to "database is locked": one shared SQLite connection instead of one per call (0 failures in 90, from 1 in 45) |
| 2026-09-29 | `f14cf1b` | Triage backends (Gemini, Laya, Jev) behind one interface, a live switch with a confidence fallback, and `triage_compare` to score them on labelled data |
| 2026-09-29 | `b4ff13d` | The console explains every case, and gains Overview, Policy and Model comparison tabs; `CLAUDE.md` and a test gate for working on the repo with Claude Code |
| 2026-09-29 | `dfd00cc` | Policy thresholds editable from the console with an admin token: hard bounds, a reason, an impact preview, and an append-only, revertible log |

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
- **Voice leg.** The call itself needs Sarvam and Murf keys and is not covered by
  tests. What runs around it is: the spoken-number normaliser and the per-call
  hotword list are pure functions and are tested offline.
- **Retention.** `prune()` enforces retention for the seven tiers marked
  prunable. Two declare a retention and are not pruned yet: the workflow-state
  checkpoint (90 days, SQLite) and conversation memory (30 days, which is
  honoured at recall — an older turn is ignored — but not deleted).
  `python -m sanwaad.memory` prints exactly which is which, and nothing is
  scheduled: pruning runs when someone runs it.
- **Triage comparison.** First run on the 11 labelled template rows: the
  offline keyword stub 72.7%, Laya zero-shot 45.5%. That is too few rows to
  decide anything, Laya's latency couldn't be measured on the (swapping) test
  machine, and Jev isn't wired yet; the comparison is meant for your own
  labelled complaints and a live `GOOGLE_API_KEY`.
- **Sign-in is built-in accounts only.** Single sign-on isn't there yet, and
  until the first account is created the console is open to anyone who can
  reach it.
- **Offline loop policy.** Without a key, a scripted policy drives the support
  loop. It reads only what a model would see, but it doesn't wander, so offline
  the ladder shows workflows' value only on the misbehaving-agent scenario; the
  benefit of offering fewer tools needs the live model to measure.
