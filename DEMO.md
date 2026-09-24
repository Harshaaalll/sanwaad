# Showing Sanwaad

Six commands, about eight minutes, no API key. Every timing below was measured
on a clean clone; every output is what the command actually prints.

The order is an argument, not a feature tour. It goes: here is the system, here
is what each layer is *worth*, here is a person in the loop, here is the system
earning its way out of that loop, and here is the same harness doing a
different job.

```bash
git clone https://github.com/Harshaaalll/sanwaad && cd sanwaad
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads a ~470 MB embedding model. Do that **before** anyone is
watching — after it, every command below starts instantly.

---

## 1. One complaint, end to end — 15s

```bash
python -m sanwaad.demo
```

A single comment walks the whole path: triage, the pattern agent, the judge,
retrieval, the draft, the grounding check, the plan, the human gate, the post,
the action, the callback, the receipt. The timeline at the end is one line per
decision.

**Say:** every step is a separate agent with a declared contract, and the graph
decides the order — not a prompt.

---

## 2. What each layer is worth, measured — 13s

```bash
python -m sanwaad.evals.loop_eval --ladder
```

```
M0 Minimal loop            passing  3/13   unsafe 0   ₹0.0131
M1 Tool use                passing  6/13   unsafe 1   ₹0.0491   over-budget 2
M2 Continuous evaluation   passing  7/13   unsafe 0   ₹0.0515   over-budget 2
M3 State and memory        passing  9/13   unsafe 0   ₹0.0483   over-budget 0
M4 Workflows               passing 10/13   unsafe 0   ₹0.0390
M5 HITL and multi-agent    passing 13/13   unsafe 0   ₹0.0376
```

This is the slide most people don't have. Read the **unsafe** column: adding
tools (M1) made the agent ship an unsupported timeline, and adding evaluation
took it back to zero. Then read **over-budget**: M2 overran its context window
until memory and compaction arrived at M3.

**Say:** each layer was added because the one before it failed a scenario, and
the number is the reason it exists.

---

## 3. It degrades instead of inventing — 11s

```bash
python -m sanwaad.loop
```

Scroll to scenario 4, *"The ledger is down"*:

```
1. lookup_transaction   amount_inr=640.0    tool_error:upstream_error
2. open_ticket          category=refund     ok
3. final                                    verified
→ done
  "I couldn't check your transactions just now. I've opened ticket
   TKT-090D44C1 so a colleague follows up."
```

The backend is gone and the agent neither guesses a refund date nor stalls.
Scenario 5 is a policy that never finishes, stopped by the loop's own budget.

**Say:** the interesting behaviour is on the failure path, so that is what the
evals are about.

---

## 4. The console, and a person in the loop — 2 min

```bash
python -m sanwaad.api.server      # http://localhost:7870
```

Click **Ingest mock feed**. Five complaints arrive and each stops at a human
gate with a *reason*. Open one and show, in this order:

- the complaint, and the **judge**'s read of the author (authenticity, reach)
- the **draft**, and the clause ids behind every claim in it
- the **grounding** verdict
- the gate's reason — e.g. *"resolution needs account data not available publicly"*

Approve it. The reply posts, and the case moves to a callback.

Point at the **health strip** in the header: index, model mode, both
concurrency pools, any open circuit, any dead-lettered item. It is red only
when something is actually wrong.

**Say:** nothing goes out without a person until the system has earned it — and
that is the next command.

---

## 5. It earns its way out of the loop — instant

```bash
python -m sanwaad.autonomy
```

```
capability             level      decisions  agreement  why
reply.refund           ASSISTED   1          100%       needs 95% over 20
reply.data_privacy     ASSISTED   1          100%       needs 95% over 20
```

That row exists because of the approval you just gave. Approving unchanged is
agreement; editing or rejecting is not. At 20 decisions and 95% it reaches
SUPERVISED and acts on its own; at 40 it stops announcing itself. One
consequential disagreement takes it back to a person.

**Say:** autonomy is measured per capability, and it falls as fast as it rises
— so the day a model or a policy clause changes, authority contracts before
anyone notices the regression.

Two limits no track record buys past: severity above 3, and any tool that moves
money. `initiate_reversal` is capped at ASSISTED forever.

---

## 6. The harness is not complaint-shaped — 1s

```bash
python -m sanwaad.missions
```

```
Nimbus Retail      400 seats  qualify → enrich → reach_out            booked
Kirana Connect      60 seats  qualify → enrich → reach_out → nurture  nurture
Two Person Studio    4 seats  qualify                                 disqualified
Unlisted Co         80 seats  qualify → enrich → nurture              nurture
```

A lead pipeline, four leads, four routes — each chosen by an agent from what it
found, each validated before it happened. Every hop carries its reason.

It inherits everything without asking: typed tools on a risk ladder, the
circuit breaker, bounded concurrency, the cost ceiling, traces, the dead-letter
queue, and per-step autonomy.

**Say:** an agent *proposes* a handoff and code decides whether it may. Two
agents bouncing work at each other is refused by the revisit rule — the edge
exists in the table and is still not allowed twice.

---

## If someone asks a hard question

| Question | Where the answer is |
|---|---|
| "Does it actually work, or is that a demo?" | `pytest -q` — 361 tests, no key needed |
| "What does it cost per case?" | `python -m sanwaad.obs` — p50/p95, budget, spend per step |
| "What happens when a tool is down?" | scenario 4 above; the breaker stops calling it after 5 failures in a minute |
| "What if something fails permanently?" | `python -m sanwaad.delivery` — three attempts, then dead-lettered with the customer's words |
| "How do you know the model isn't making things up?" | every claim carries a clause id; the grounding check is a separate step and a degraded one routes to a person |
| "What's kept, and for how long?" | `python -m sanwaad.memory` — twelve tiers, each with retention and model exposure |
| "Would you run this on real money?" | No, and the code says so: `initiate_reversal` is not auto-approvable, so no track record ever lets it run unattended |

## In a container

```bash
docker build -t sanwaad .
docker run --rm -p 7870:7870 sanwaad
```

Ready in about 15 seconds — the model and the policy index are baked at build
time. ~350 MB to ship. `/health` is liveness, `/ready` is readiness and answers
503 until the index is loaded.

## With real models

Put `GOOGLE_API_KEY` in `.env` and every command above routes to Gemini. The
eval tables then report real cost and real latency instead of offline
placeholders — which is the one thing the numbers here cannot show.
