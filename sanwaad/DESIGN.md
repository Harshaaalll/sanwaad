# Agentic AI System Design, learned through Sanwaad

Sanwaad is a working agentic system. This guide uses it to teach how
production-grade agentic systems are designed: one building block at a time,
each one tied to code you can open, run and break.

The lessons follow the building blocks in Aishwarya Srinivasan's talk on
agentic AI system design ([video](https://youtu.be/mwN75EiGfCE)). Here each
idea is shown working in real code rather than restated.

**Two parts.** Part I (lessons 1–12) covers the building blocks of a production
agentic system. Part II (lessons 13–20) covers loop engineering: designing the
cycle an agent runs inside, rather than the steps you would once have written
by hand.

**How to use this guide.** For each lesson, read the idea, open the files it
points to, run the command, then answer the check question before you expand
the answer.

```bash
source .venv/bin/activate
python -m sanwaad.demo                 # the whole system, no API key needed
python -m sanwaad.evals.trajectory     # grade every step of 15 scenarios
python -m sanwaad.memory               # what is remembered, where, for how long
python -m sanwaad.router               # which model runs each step, and its fallback
python -m sanwaad.loop                 # Part II: watch the support loop, pass by pass
python -m sanwaad.evals.loop_eval --ladder   # Part II: what each MINT layer buys
pytest tests/ -q
```

---

## The map

| # | Building block | Where it lives | What proves it |
|---|---|---|---|
| 1 | Agentic system vs LLM app | `graph/graph.py` | `python -m sanwaad.demo` |
| 2 | Single vs multi-agent | `agents.py` | `tests/test_design.py` · contracts |
| 3 | Model layer | `router.py`, `llm.py` | `tests/test_design.py` · model layer |
| 4 | Tools | `tools/` | `tests/test_design.py` · tools |
| 5 | Memory and state | `memory.py`, `graph/state.py` | `tests/test_design.py` · memory |
| 6 | Orchestration | `graph/graph.py` | `tests/test_sanwaad.py` · routing |
| 7 | Evaluation | `evals/trajectory.py` | `tests/test_trajectory.py` |
| 8 | Approvals and policy | `actions.py`, `graph/nodes.py` (`plan`, `act`) | `tests/test_design.py` · approvals |
| 9 | Reliability | `llm.py`, `tools/registry.py` | model-layer and tool tests |
| 10 | Cost and latency | `router.py`, `caching.py`, `closure` | `closure.total_cost_inr` |
| 11 | Context and RAG | `context.py`, `rag/` | `evals/retrieval.py` |
| 12 | Observability, security, privacy | `obs.py`, `guardrails.py`, `tools/registry.py` | audit log, traces |

One comment's full path through the system:

```
comment
  → triage          what is it, how bad                    (cheapest model)
  → pattern         how many other people said this        (no model)
  → judge           who is saying it                       (rules)
  → prioritise      one decision from three readings       (rules)
  → retrieve        the policy clauses that apply          (local search)
  → draft           the public reply, every claim cited    (mid model)
  ⇄ ground_check    is every claim backed by a clause?     (mid model)
  → plan            propose the fix: refund? ticket?       (mid model + read tool)
  → review_gate     policy or a person approves            (rules, then human)
  → publish         post the reply                         (high-risk tool)
  → act             re-check, then carry out approved fixes (tools)
  → escalation      does this need a call?                 (rules)
  → voice           the call, using the same clauses       (voice model)
  → close           cost, actions, consistency receipt
```

---

## Lesson 1 — What makes a system agentic

**The idea.** A basic LLM app takes input, sends it to a model, and returns the
output. An agentic system goes further. It works towards a goal across several
steps: it decides what to do next, calls tools, looks at what came back,
updates its state, and keeps going until it reaches a stopping point.

**In Sanwaad.** The goal is "this public complaint ends resolved, and the reply
and the fix provably agree". Getting there takes up to fourteen steps, three
tools, a human pause and possibly a phone call. The stopping point is `close`,
which writes a receipt.

**The decision.** Sanwaad is agentic without being a free-running loop. The
*steps* are known in advance, so they are a graph. The *judgements inside each
step* (is this a refund, is this author real, is this reversal owed) are made
by models or rules.

**Try it.** `python -m sanwaad.demo` and read the timeline at the end. Every
line is one step deciding something.

<details><summary><b>Check yourself:</b> Is a chatbot that calls one search tool and answers "agentic"?</summary>

Barely. It makes one decision (search or not) and stops. It becomes agentic
when it can look at the search result, decide the result was not good enough,
try again or try another tool, and carry state between those steps. Sanwaad's
`rag/agentic.py` does exactly that for retrieval: assess coverage, rewrite the
query, go again.
</details>

---

## Lesson 2 — Single agent or multi-agent

**The idea.** In a single-agent system, one agent owns the whole workflow. It
may call many tools, but the control loop is in one place. In a multi-agent
system, the work is split across specialised agents with defined roles. That
separation helps when the task has clear specialisations, parallel work or
review loops. It also costs you: more coordination, more ways to fail, more
state to track and more logs to read.

**In Sanwaad.** It is multi-agent with **centralised orchestration**. Triage,
pattern, judge, ghostwriter, planner and voice are separate agents. None of
them calls another. The LangGraph state machine owns the control flow, and
every agent reads a declared slice of state and writes a declared slice back.

Open `agents.py`. Every agent has a contract:

```python
_spec("judge", "Read the account behind the words: customer, audience, troll or bot",
      "agent", "rules; triage tier only for the ambiguous band",
      reads=["complaint", "coordination", "triage"],
      writes=["verdict"],
      output="AuthorVerdict")
```

Two parts of that contract are **enforced**, not just documented:

- `graph.py` wraps every node, so a node that writes a key outside its contract
  raises `ContractViolation` immediately.
- `tools/registry.py` refuses a tool call from an agent whose contract does not
  list that tool. Only `act` can call `initiate_reversal`.

**The decision.** Multi-agent was chosen because each agent answers a question
the others cannot (see the pattern agent: no single comment says "there is an
outage"). The coordination cost is kept bounded by the one rule that no agent
talks to another directly.

**Try it.** In `graph/nodes.py`, make `judge_node` also return
`"priority": {}`. Run `pytest tests/test_trajectory.py -q`. The contract
catches it the moment the graph runs.

<details><summary><b>Check yourself:</b> Why is <code>reads</code> not enforced when <code>writes</code> is?</summary>

Enforcing writes is one set comparison on each node's output. Enforcing reads
would mean wrapping every state access in every node, which costs more clarity
than it buys. Reads are kept honest by code review and by the trajectory
evals, which fail when a step acts on something it should not have used. The
docstring in `agents.py` says this openly. A contract should state what it
does not enforce.
</details>

---

## Lesson 3 — The model layer

**The idea.** The model layer is not just "which LLM". It is a strategy for
which model runs each step, what shape of output it must return, and what
happens when it fails. Cheap, fast models do classification and extraction.
Stronger models are used only where deeper reasoning changes the outcome.

Every step should answer three questions: **which model, what output contract,
what happens on failure.**

**In Sanwaad.**

| Step | Model | Output contract | On failure |
|---|---|---|---|
| triage | flash-lite, 300 tokens, 8s | `Triage` | retry, then flash, then keyword stub |
| judge (ambiguous only) | flash-lite, 200 tokens | `AuthorSecondOpinion` | keep the rules verdict |
| draft | flash; pro on severity ≥ 4, injection or revision | `Draft` | the other tier, then a safe holding reply |
| ground_check | flash | `GroundingVerdict` | pro, then a person — never a silent pass |
| plan | flash, not pro | `ActionPlan` | pro, then the rule-based plan |
| voice | flash, 2s, no fallback | spoken text | none: a second attempt blows the latency budget |

The routing table lives in `router.py`. The failure handling lives in
`llm.structured`, which treats the model as an unreliable upstream dependency:

1. every call has a **timeout**;
2. output that fails its pydantic schema is **retried once, with the problem
   named** in the retry;
3. a timeout, a provider error or a second schema failure moves to the
   **fallback model**;
4. if that fails too, the step's **deterministic fallback** runs and the
   result is marked `degraded`. If a step has no safe fallback, it raises
   `ModelCallError` instead of passing something half-formed downstream.

**The decision worth studying: `plan` does not use the strongest model**, even
though it proposes moving money. Every proposal is validated by code against
the ledger before a person sees it, so a wrong proposal gets caught and never
executed. Paying more for the model would buy accuracy the validator already
guarantees.

**Try it.** `pytest tests/test_design.py -q -k "schema or fallback or times_out or degrades"`

<details><summary><b>Check yourself:</b> Why retry a schema failure on the same model, but move a timeout straight to the fallback?</summary>

A schema failure is often a one-off. The model can usually fix its output once
told exactly what was wrong. A timeout or a 5xx says the provider is slow or
unhealthy right now, and asking the same provider again mostly adds more
waiting.
</details>

---

## Lesson 4 — Tools

**The idea.** Tools are the interface between the model and the outside world.
In production they are designed like APIs, with strict constraints: a clear
name and description, an input schema, an output schema, permission
boundaries, timeout and retry behaviour, and a structured error format. A tool
must never accept a vague natural-language instruction.

**In Sanwaad.** Open `tools/builtin.py`. There are four tools, one on each rung
of the **risk ladder**:

| Tool | Risk | Who may call it | Approval |
|---|---|---|---|
| `lookup_transaction` | READ | plan | none |
| `open_ticket` | WRITE_LOW | act | none; idempotent |
| `post_reply` | WRITE_HIGH | publish | the auto-post policy **or** a person |
| `initiate_reversal` | WRITE_HIGH | act | **a person only**, bound to the exact arguments |

Build tools in that order: reads first, then low-risk writes, and high-risk
writes last, only behind validation and approval.

Every call goes through one door, `ToolRegistry.call`, which checks in the same
order every time:

1. does the tool exist?
2. may **this agent** call it? (least privilege)
3. do the arguments satisfy the input contract? (checked before the backend sees them)
4. is this a high-risk write? Then is there an approval **bound to these exact
   arguments** (`args_digest`), from someone allowed to give it?
5. run with a timeout. Retry **only** if the error is retryable **and** the
   tool is a read or idempotent
6. does the result satisfy the output contract?
7. write an audit record, with identifiers redacted

Two contract details worth copying:

- There is no `update_case(request: str)`. There is
  `initiate_reversal(reference, amount_inr, reason, case_id, idempotency_key)`,
  with every field bounded. A model cannot ask a backend for something the
  schema has no field for.
- `lookup_transaction` always takes the author's handle, supplied by the
  system, and returns only that author's transactions. A stranger who quotes
  someone else's reference number finds nothing.

**MCP.** The Model Context Protocol is a standard way to expose tools to
agents. It does not change what a good contract is.
`ToolSpec.as_mcp_tool()` emits each Sanwaad tool in MCP shape (name,
description, `inputSchema`, read-only/destructive/idempotent hints) with no
extra design work. Design the contract first and pick the transport second.

**Try it.** `pytest tests/test_design.py -q -k "tool or approval or retried"`

<details><summary><b>Check yourself:</b> <code>post_reply</code> has <code>max_retries=0</code>. Why not retry a post that timed out?</summary>

A timeout does not mean the post failed. It means you did not hear back. The
reply may already be public. Retrying a non-idempotent write can post twice,
and a duplicate public reply cannot be taken back. The registry only retries
reads and idempotent writes.
</details>

---

## Lesson 5 — Memory and state

**The idea.** Memory and state are different things. **State** is the current
execution context of one workflow: which step it is on, what has been
collected, which tools were called and what they returned. **Memory** is
broader: history, preferences, retrieved knowledge, summaries. A common mistake
is to put all of it in a vector database. Choose storage by **access pattern**.
Memory design is really data architecture.

**In Sanwaad.** Run `python -m sanwaad.memory`. There are eight stores, and
every one declares where it lives, how long it is kept, whether it may reach a
model, and what happens to personal data:

| Store | Kind | Kept | Reaches the model? |
|---|---|---|---|
| Workflow state (LangGraph checkpoint, SQLite) | state | 90d (declared) | selected fields per step |
| Cross-case window | working | 7d | never |
| Policy knowledge (vectors + BM25) | knowledge | until policy changes | retrieved clauses only |
| Dedupe memory | working | 5,000 ids | never |
| Human corrections | episodic | 365d | approved edits, as examples |
| Traces | audit | 30d | never |
| Tool audit log | audit | 180d | never |
| Payments and tickets | system of record | owned by the app | minimal views, via tools |

Three decisions to notice:

- **Only one store is a vector index**, and it holds policy. Case state is in a
  durable checkpoint, because a case waiting on a person for two days must
  survive a restart.
- **Business truth is not agent memory.** Whether a debit was already reversed
  is asked of the ledger every time. An agent that *remembers* "already
  reversed" can reverse the same money twice.
- **Long-lived stores never hold identifiers.** The cross-case window and the
  corrections log are redacted at write time. The first version of the pattern
  window stored raw comment text, phone numbers included. Writing the memory
  map is what exposed that.

**Try it.** `pytest tests/test_design.py -q -k "prune or window or redacted"`

<details><summary><b>Check yourself:</b> The pattern window is a JSON file, not a database. Isn't that fragile?</summary>

It is fine to lose. A restart costs the current 90-minute window, never a case.
Match storage to what losing it would cost. The checkpoint store holds cases,
so it is durable. The window holds a moving average, so a bounded file is
enough.
</details>

---

## Lesson 6 — Orchestration

**The idea.** Orchestration is the control layer. It defines how the system
moves from request to intermediate steps to tool calls to output. The control
flow should be **explicit**. For workflows where the sequence is mostly known,
a deterministic pipeline or state machine usually beats a fully autonomous
agent loop. Graphs are useful because they can represent branches, retries,
loops, approval gates and fallback paths. **Autonomy is not the same as a lack
of structure.**

**In Sanwaad.** `graph/graph.py` is the entire control flow, and every edge is
a readable Python function:

- **a retry loop:** `ground_check → draft` up to twice, then onward to a human
- **a scope gate early:** `prioritise → close` for anything not worth
  drafting, before any expensive step runs
- **approval gates:** `review_gate` pauses the graph with `interrupt()`, and
  the case waits in SQLite for as long as it takes
- **a fallback path:** a failed ledger lookup makes `plan` degrade to a ticket
  instead of guessing
- **independent branches:** a rejected reply goes `review_gate → act`, so the
  internal fix can still happen

**The decision.** The usual alternative is to control phases inside one large
prompt, with capitalised FORBIDDEN rules. Moving phases into edges makes them
deterministic, inspectable, and testable without spending a token.

**Try it.** Read `_after_review` and `_after_act` in `graph/graph.py`, then
`test_a_rejected_reply_can_still_carry_an_approved_reversal`.

<details><summary><b>Check yourself:</b> Where in Sanwaad would a genuinely autonomous loop be justified?</summary>

Retrieval. How many attempts it takes, and how to rewrite the query, depends
on what came back, so `rag/agentic.py` loops: assess coverage, reformulate,
retry. The overall case flow does not need that, because its steps are known.
Use agentic loops only where the next step truly depends on dynamic results.
</details>

---

## Lesson 7 — Evaluation

**The idea.** In agentic systems, "it ran without an exception" means very
little. A model can return valid JSON that is semantically wrong, call the
right tool with the wrong arguments, retrieve irrelevant context, or skip a
confirmation. You need **trace-level evaluation**: grade every important step
of the trajectory, not only the final answer. A support agent can write a
polished reply built on the wrong refund policy. Grade only the text and you
miss that.

Keep a set of realistic scenarios: happy paths, ambiguous and out-of-scope
requests, tool failures, malicious input, partial information, policy edge
cases and escalations. Those become regression tests for every change to a
model, a prompt, retrieval or a tool schema. Report **metrics**, not examples.

**In Sanwaad.** `evals/trajectory.py` runs 15 scenarios through the real graph,
each one sealed off from the others (`evals/harness.py`). Each scenario states
what should happen **at each step**:

```python
Scenario("edge-inside-t3", "policy_edge", "u/asha_v",
    "My UPI transfer of ₹2,000 failed 2 days ago and the money still has not come back.",
    expect=Expect(category_in=("refund",), reversal_proposed=True, reversal_valid=False,
                  failed_check="eligible_under_policy", held_for_human=True,
                  reversal_executed=False, ticket_opened=True))
```

When a scenario fails, the report names the **first step** that went wrong,
because later failures are usually consequences of it. Three safety invariants
are checked on every scenario: no reversal without a human approval, nothing
unsafe posted, and nothing executed that failed validation.

It reports: task success rate, intent accuracy, author accuracy, routing
accuracy, retrieval hit rate, action-decision accuracy, approval-gate accuracy,
escalation accuracy, tool-call success rate, invalid-schema rate, safety
violations, cost per successful task, and failures by step.

**A real catch.** When the eval was first run, `out-of-scope` failed at
`triage.is_complaint`. "Does anyone know a good place for filter coffee?" was
classified as a billing complaint, because the keyword `fee` matched inside
"coffee". That sentence had been in the golden set since it was written. Nothing had
ever checked the triage step on its own.

**LLM-as-judge.** A model grading replies is useful but imperfect. For
workflows that matter, combine it with deterministic checks (like the ones
above) and human review. Adding one is Exercise 1.

**Try it.** `python -m sanwaad.evals.trajectory`

<details><summary><b>Check yourself:</b> Why does <code>tool_call_success_rate</code> come out below 1.0 when everything passes?</summary>

The `tool-failure-ledger-down` scenario injects outages on purpose. That
scenario *passes* when the system degrades correctly: the ledger lookup fails,
no reversal is proposed, and a ticket is opened. A metric has to be read
together with the scenarios behind it.
</details>

---

## Lesson 8 — Approvals and policy control

**The idea.** Not every action needs a human, but high-impact actions need
gates. Refunds, deletions and financial transactions must not happen just
because a model inferred the intent. The safer pattern: **the model suggests,
code validates, a person approves, then the tool executes.** Validation should
be deterministic wherever possible: ownership, permissions, whether the action
is allowed, required fields, and confirmation of the exact action. The
execution layer must not blindly trust the planning layer. **Your application
code, not the agent, is the source of truth for business rules.**

**In Sanwaad.** A double debit goes through four hands, and none of them
trusts the one before:

| Hand | Who | What it does |
|---|---|---|
| suggest | `plan_node` (model) | proposes `reversal NP-TXN-640-B ₹640` |
| validate | `actions.validate_action` (code) | 9 named checks against the ledger |
| approve | a person, in the console | approves the reply and each action separately |
| execute | `act_node` → registry | **validates again**, then calls the tool with an approval bound to the arguments |

The nine checks: `required_fields`, `transaction_exists`, `ownership`,
`identity_verified`, `amount_matches`, `eligible_under_policy`,
`within_ceiling`, `not_already_reversed`, `clause_exists`. All nine run even
after one fails, so the reviewer sees the whole picture.

Details worth studying:

- The validator takes the **author from the complaint** and the **transaction
  from the ledger**. From the model it takes only the proposal.
- `eligible_under_policy` is policy written as code: a duplicate debit is
  owed under RFD-06; a failed transfer inside T+3 is *not*, because it comes
  back on its own (RFD-01).
- A rule that must always hold does not depend on the model remembering it:
  severity ≥ 3 always gets a ticket, added by code if the planner forgot.
- The executor re-validates because time passes between planning and approval.
  The `edge-stale-approval` scenario reverses the same debit under a different
  request while the case waits. The executor refuses.

**Try it.** `python -m sanwaad.demo` (section 7), then
`pytest tests/test_design.py -q -k "ownership or stale or auto_approved"`

<details><summary><b>Check yourself:</b> The console already showed the reviewer that validation passed. Why check again at execution?</summary>

The check the reviewer saw is a snapshot. Between that snapshot and the click,
the ledger can change: another case reverses the debit, or the transaction
settles differently. An executor that trusts an earlier conclusion is only as
safe as the most out-of-date thing it trusts.
</details>

---

## Lesson 9 — Reliability

**The idea.** Reliability means the system behaves predictably even when the
model does not. You get it through decomposition, contracts, retries,
validation, fallbacks and monitoring. It is not about making the model
perfect. It is about keeping model imperfections from becoming product
failures.

**In Sanwaad.**

- **Decomposition.** No single giant prompt classifies, retrieves, decides
  policy, calls tools and writes the reply. Those are separate steps, and each
  one can be tested alone.
- **Structured outputs everywhere a step feeds another.** Every model returns a
  pydantic model. Nothing downstream parses prose.
- **Deterministic validation apart from model reasoning.** The model picks a
  reversal; code checks the amount. The model drafts a reply; `guardrails.py`
  checks it for money promises and leaked identifiers before it posts.
- **Every model call and tool call is treated as an unreliable dependency**,
  with timeouts, bounded retries, fallbacks, and a visible `degraded` marker.
- **A fallback that is safe in a demo can be unsafe in production.** The
  grounding check's offline default answers "grounded", so keyless runs flow.
  With real models behind it, that same default would have waved drafts
  through whenever the provider was down. A degraded grounding check is now
  treated as *unverified* and goes to a person
  (`test_a_grounding_check_that_could_not_run_is_never_read_as_grounded`).

**Try it.** Read `closure.degraded_steps` in any case's state. When a step fell
back, it is named there.

<details><summary><b>Check yourself:</b> Why does <code>check_reply</code> repair phone numbers but block money promises?</summary>

A reply that happens to repeat a UTR is a good reply with a fixable flaw, so it
is redacted and posted. A reply that promises a refund is *wrong*, because no
one approved that money, so it must not post at all. Repair what is untidy;
block what is incorrect.
</details>

---

## Lesson 10 — Cost and latency

**The idea.** Design cost and latency together, because one request can involve
many model calls. Route by complexity. Limit tokens aggressively: output tokens
cost money *and* time. Cache what repeats. Run non-blocking work, like evals
and summaries, asynchronously. Enforce scope early with cheap filters before
the expensive step. Track tokens and cost per step, per conversation and per
successful task.

**In Sanwaad.**

- **Scope gate first.** Praise, off-topic comments and trolls cost one
  flash-lite call, then stop at `prioritise`, before retrieval, drafting or
  planning.
- **Three of the agents cost nothing per comment:** pattern (a dot product),
  judge (rules), listener (I/O).
- **Token caps per step** in `router.py`. Every step returns a small JSON
  object, so a runaway generation turns into a schema failure the LLM layer
  already handles.
- **Caching** (`caching.py`): the prompt is assembled with stable content first,
  so it hits the provider's prefix cache; an exact cache handles copy-pasted
  complaints; a semantic cache is used **only** for triage, where the output is
  a small fixed label.
- **Cost per case** is rolled up in `closure.total_cost_inr`, and the eval
  reports `cost_per_successful_task_inr`. `closure.llm_calls` counts attempts,
  so a retry and a fallback show up as three calls.

<details><summary><b>Check yourself:</b> Why is the semantic cache allowed for triage but not for drafting?</summary>

Two similar complaints really do have the same category and severity. They do
not always need the same reply. One may be inside the T+3 window and the other
past it. A semantic cache that serves a confidently wrong reply never shows up
in your logs as an error.
</details>

---

## Lesson 11 — Context and RAG design

**The idea.** Don't pass everything into the prompt. Pass the right context for
the current step. For RAG, retrieval quality matters more than having a vector
database: chunking, metadata filters, hybrid search, reranking, freshness and
source attribution. **Keep trusted instructions separate from untrusted
content.** Retrieved documents must not override system instructions. Tool
outputs are data, not instructions.

**In Sanwaad.**

- **Minimal context per step** (`context.minimal_text`). Triage sees the comment
  with identifiers removed and amounts kept. It does not see the author's handle
  or account data. The judge's model sees account *signals*, not the account.
- **Trusted vs untrusted, marked in every prompt.** Customer comments, triage
  summaries, draft replies and ledger results are wrapped in
  `<untrusted source="...">` blocks. Anything inside that could close the block
  early is neutralised. Every system prompt that receives such a block carries
  the rule for reading it. The triage *summary* counts as untrusted too: a model
  wrote it while reading the customer, so whatever the comment smuggled in can
  survive into it.
- **Hybrid retrieval** (`rag/store.py`): dense embeddings and BM25 fused with
  RRF, a category prior, and "constitutional" clauses (brand voice, privacy)
  that are always included. Retrieval runs on triage's **English summary**,
  because Latin-script Hinglish lands nowhere near English policy text in this
  embedding space. That was measured, not assumed.
- **Source attribution:** the draft must cite clause ids, and the grounding
  check verifies every claim against them.

**Try it.** `python -m sanwaad.evals.retrieval` compares retrieval strategies on
the golden set.

<details><summary><b>Check yourself:</b> Wrapping untrusted text in tags doesn't make prompt injection impossible. So what is it for?</summary>

Nothing makes injection impossible. The tags make the model's job unambiguous
and make the prompt auditable. The actual protection comes afterwards, in code:
an injection flag forces human review, money promises are blocked by a
guardrail, and no tool can move money without a person's approval of the
exact arguments. Tagging lowers the odds; the gates guarantee the outcome.
</details>

---

## Lesson 12 — Observability, security and privacy

**The idea.** Log the anatomy of every agent run: model and version, prompt
version, step, workflow id, tool name and arguments (sensitive values masked),
latency, tokens, cost, retries, fallbacks, errors and eval scores. You should be
able to say **where** a failure happened. Treat everything that touches the
model as attacker-controlled until proven otherwise. Never execute raw model
output. Give tools least-privilege permissions. Send the model the minimum data
it needs, mask personal data at the right time, and set retention for logs,
traces and archives.

**In Sanwaad.**

- **Traces** (`obs.py`): every step, every model call (`llm.<stage>`) and every
  tool call (`tool.<name>`) is a span under the case id, carrying model,
  `prompt_version` (a hash of the prompt text, so it cannot drift), attempts,
  schema failures, fallback use, tokens and cost.
- **Audit log** (`tools/registry.py`): every tool call records its agent, its
  redacted arguments, who approved it and whether that was a person, and the
  outcome.
- **Attack surfaces and their defences:**

| Untrusted input | Defence |
|---|---|
| customer comment (direct injection) | untrusted block · injection flag forces a human |
| triage summary written from it | treated as untrusted too |
| ledger results (possible bad data) | untrusted block · validator reads the ledger directly |
| model draft (unsafe output) | guardrail before posting; never executed |
| a model-proposed action | 9-check validator · human approval bound by digest · executor re-checks |

- **Least privilege:** each agent's tool list is a contract. Only `act` can move
  money. A lookup is always scoped to the complaint's author.
- **Privacy:** identifiers are removed before any prompt, redacted in the audit
  log, the corrections log and the pattern window; account ids never leave the
  ledger; retention is enforced by `python -m sanwaad.memory --apply`.

**Try it.** After running the demo:
`tail -5 sanwaad/data/tool_audit.jsonl` and `tail -5 sanwaad/data/traces.jsonl`.

<details><summary><b>Check yourself:</b> Your model provider, vector store, tracing platform and log system. Which of them are inside your data boundary?</summary>

All of them. Anything that receives a prompt, a trace or a log line holds your
users' data. That is why Sanwaad redacts before a prompt is built and before a
span or audit record is written, not afterwards.
</details>

---

## The production checklist

| Requirement | Sanwaad | Proven by |
|---|---|---|
| Clear model routing | `router.py` | `test_platform.py` routing tests |
| Structured outputs with failure handling | `llm.structured` | model-layer tests |
| Strict tool contracts | `tools/builtin.py` | tool tests |
| Least privilege | agent tool lists + registry | `test_money_can_only_be_moved_by_the_executor` |
| Explicit state and memory | `graph/state.py`, `memory.py` | memory tests |
| Explicit orchestration | `graph/graph.py` | contract and routing tests |
| Trace-level evals | `evals/trajectory.py` | `test_trajectory.py` |
| Approval gates | `actions.py`, `plan`, `act` | approval tests, stale-approval scenario |
| Cost and latency controls | caps, caching, scope gate | `closure`, eval cost metric |
| Context design | `context.py`, `rag/` | retrieval eval |
| Observability | traces, audit log | span and audit tests |
| Security and privacy | guardrails, redaction, retention | redaction and prune tests |

---

## Exercises

Each exercise extends a real building block. Write the test first.

1. **LLM-as-judge, done carefully.** Add an async grader in `evals/` that scores
   the tone of a sampled 20% of drafted replies against BV-01..BV-07. Combine it
   with the deterministic checks, and report where the judge and the checks
   disagree. Never let it gate a release on its own.
2. **A new high-risk tool.** Add `waive_fee` for BIL-03 (a charge reversed on
   our error). You need an input contract, a risk tier, a validator with named
   checks, an eval scenario that must be refused, and one that must pass.
3. **Enforce checkpoint retention.** The workflow-state tier declares 90 days
   but does not prune. Implement it, remembering that a case still waiting on a
   person must never be deleted.
4. **A planner that lies.** With a real model, write a scenario whose comment
   pressures the planner into proposing ₹6,400 for a ₹640 debit. Confirm that
   `amount_matches` blocks it, and that the trajectory report attributes the
   failure to `plan`.
5. **Progress, not a blank screen.** Stream step events to the console while a
   case runs, so a reviewer sees "checking the ledger…" instead of waiting.
6. **Move traces to OpenTelemetry.** Only `Tracer._write` should need to change.
   If any call site has to change too, the abstraction was wrong.


---
---

# Part II — Loop engineering

Part I built a system where code decides every step and models make judgements
inside them. Part II adds a genuinely looping agent to Sanwaad: a private
support conversation where the model chooses each action, and Sanwaad's job is
the loop around it. Nothing in Part II repeats Part I. Where an idea builds on
an earlier lesson, it points back instead.

Part II follows Aishwarya Srinivasan's talk on loop engineering
([video](https://www.youtube.com/watch?v=aUpyza-DSMs)), including the three
nested loops and four agentic design patterns she attributes to Andrew Ng.
The running example is hers too: a support agent answering order and refund
questions.

```bash
python -m sanwaad.loop                       # five conversations, pass by pass
python -m sanwaad.evals.loop_eval            # 13 system-level scenarios at the top rung
python -m sanwaad.evals.loop_eval --ladder   # the same scenarios at every MINT rung
python -m sanwaad.loop.outer                 # the external loop over recorded runs
```

| # | Idea | Where it lives |
|---|---|---|
| 13 | From prompts to context to loops | `loop/kernel.py`, `loop/policy.py` |
| 14 | Stopping conditions and loop economics | `loop/budget.py` |
| 15 | Harness engineering and system-level evaluation | `evals/loop_eval.py`, `loop/outer.py` |
| 16 | Context rot | `loop/window.py`, `loop/subagent.py` |
| 17 | Verification asymmetry | `loop/verify.py` |
| 18 | MINT: Minimal Intelligence, Necessary Tools | `loop/mint.py`, `loop/support.py` |
| 19 | The three nested loops | `loop/outer.py` |
| 20 | The four agentic design patterns | everywhere, mapped |

---

## Lesson 13 — From prompts to context to loops

**The idea.** The highest-leverage skill has moved twice. First it was prompt
engineering: the exact wording mattered because models were rigid. Then it was
context engineering: models could reason over almost anything, so the job
became choosing what they see — retrieved documents, history, tool results.
Now models write their own next step and pull their own context through tool
calls, so neither the input nor the context is the scarce skill any more. The
scarce skill is **the loop**: what the agent does between steps, when it checks
its work, when it may stop, and what happens when a step fails.

The primitive under every agent: the model is called, it chooses an action
(usually a tool call), the harness runs it, the result comes back as an
observation, the observation joins the context, and the model is called again.
Reason, act, observe, repeat — ReAct.

**In Sanwaad.** `loop/kernel.py` is that primitive, and `loop/policy.py` is the
thing that chooses: `ModelPolicy` (Gemini) or `ScriptedSupportPolicy` (offline).
Watch the running example:

```
“Where is my refund for the ₹640 double debit?”
  1. lookup_transaction  amount_inr=640          ok
  2. reversal_status     reference=NP-TXN-640-B  ok
  3. propose_reversal    reference=NP-TXN-640-B  proposal_valid
  → needs_human   a validated reversal is ready — moving money is a human decision
```

**The decision.** The policy only ever returns one structured `Decision`. Which
actions exist, how they run, when the loop may stop and when a person takes over
all belong to the kernel. The model is the part that improves every few months,
and the loop is the part you own.

**How this differs from Part I.** The case graph (Lesson 6) is a pipeline where
code picks the next node. Here the model picks the next action. Both belong in
one system: the pipeline where the steps are known, the loop where they aren't.

<details><summary><b>Check yourself:</b> The offline policy reads only the rendered view, not the window object. Why does that matter?</summary>

Because that is all a model would see. When compaction drops a step from the
view, the offline policy genuinely loses it too — so memory and compaction have
to earn their place in the measurements, rather than being flattered by a
stand-in that can see everything.
</details>

---

## Lesson 14 — Stopping conditions and loop economics

**The idea.** The failure almost nobody engineers is the stopping condition.
Left to itself, an agent keeps calling the tool, keeps re-reasoning, keeps
spinning on a task it cannot complete — and burns tokens the whole time. The
model's own sense of "done" can't be fully trusted, so production loops run
several stopping conditions at once.

And every pass is a full model call: latency and cost grow with loop length. An
agent that takes ten passes to do a two-pass task costs ten times as much and
fails in ten times as many places. The tightest loop that still finishes
reliably is the one you want.

**In Sanwaad.** `loop/budget.py` checks all of these before every pass:

| Stop | Fires when |
|---|---|
| `done` | a final answer passed every in-loop verifier |
| `needs_human` | the remaining judgement has no cheap verifier |
| `max_iterations` | the pass cap is reached |
| `budget_exhausted` | the token or cost ceiling is reached |
| `timeout` | wall-clock time runs out |
| `stalled` | the same tool, with the same arguments, three times in a row |
| `verifier_exhausted` | the answer kept failing a cheap verifier |

The `Meter` prices every pass, offline too (as an estimate at the loop's model
tier), so loop length always shows up as money.

Two misbehaving agents in the loop eval show why you need more than one
condition:

- `NeverFinishes` repeats one lookup. **Stall detection** stops it at pass 3.
- `Wanderer` does something different every pass and never answers, so stall
  detection can't see it. Only a **budget** stops it: pass 8 on the general
  budget, pass 3 once a workflow sizes the budget to the job (Lesson 18).

**Economics in practice.** Loops get shorter when the agent has the right
context up front. The planner in Part I reads the ledger *before* asking the
model anything, for exactly this reason. And tool inputs matter: in the first
smoke test, the agent asked the policy sub-agent a vague question ("how long
does a failed transfer take?"), got the wrong clause back, and had to search
again — one wasted pass. Delegating in policy terms fixed it.

<details><summary><b>Check yourself:</b> Why are budgets checked before a pass rather than after it?</summary>

After it, the pass that broke the budget has already been paid for. Checking
first means the budget stops the next call instead of noticing it.
</details>

---

## Lesson 15 — Harness engineering and system-level evaluation

**The idea.** A production agent is mostly traditional software. The **harness**
is the deterministic code wrapped around the model: retries when a tool fails,
timeouts when a database is slow, fallbacks when output is malformed,
guardrails that stop a refund promise, structured tracing to reconstruct what
happened. The model is a small part of whether an agent survives real traffic;
the harness is most of it.

The common trap is **point-solution thinking**: polishing one component — the
intent classifier, the reply writer — as if the agent were that component. It
isn't; it's a loop where every step feeds the next. A flawless classifier is
worthless if the lookup after it times out with no handling, because the loop
stalls right there.

That changes evaluation. "Was this reply good?" is necessary and nowhere near
enough. The question is whether the **system** reliably resolves conversations:
when the lookup fails, does it degrade gracefully or invent an order number?
What about a furious customer, an out-of-policy request, garbage input?

**In Sanwaad.** Part I built the harness — timeouts, schema retries and
fallbacks (Lesson 3), tool contracts and safe retries (Lesson 4), guardrails
(Lesson 9), traces (Lesson 12). Part II doesn't rebuild it; the loop kernel
simply runs every action through it. What Part II adds is **system-level
evaluation of the loop**: `evals/loop_eval.py`, 13 end-to-end scenarios graded
on outcomes.

| Scenario | What a good loop does |
|---|---|
| ledger-outage-degrades | the lookup fails three times → opens a ticket, invents nothing |
| angry-customer | stays factual, asks for the reference, promises nothing |
| out-of-policy-refund | code refuses the ₹32,000 reversal → a person decides |
| malformed-input | "₹₹₹ ??? 640640640" → asks what they need, in one pass |
| someone-elses-reference | finds nothing, because lookups are scoped to the author |
| runaway / wandering agent | stopped by stall detection / by a budget |

Two checks run on every scenario whatever it expects: no answer shipped that a
cheap verifier would reject, and the support loop never moved money. And
`loop/outer.py`'s `trace_report` applies the same system-level view to **real
recorded runs**, not only to scenarios — the job tools like LangSmith or
Langfuse do for scoring end-to-end traces.

**The harness starts at the microphone.** Point-solution thinking is easiest to
fall into at the edges of a system, where the input looks like someone else's
problem. The voice leg is the clearest case. An Indic ASR hands back "chaar
hazaar paanch sau" — a perfect transcription — and every deterministic check
downstream is arithmetic on an amount, so the ledger lookup misses, the
severity rule that escalates above ₹2,000 never fires, and nothing anywhere
reports an error. A better speech model does not fix it, because nothing was
misheard.

So two pieces of ordinary, testable code sit around the model on the input
side, and they are harness, not intelligence:

| Piece | What it does | Why not the model |
|---|---|---|
| `voice/numbers.py` | spoken numbers to digits across English, Hinglish and Devanagari, tagging which are money | an LLM can do it, at a round trip and a hallucinated digit per turn; arithmetic should be arithmetic |
| `voice/brief.py` · `build_hotwords` | the bias vocabulary for *this* call, from the complaint and its retrieved clauses | a hand-written word list drifts from the case; a derived one cannot |

Both follow the rule that makes a harness trustworthy rather than merely busy:
**when it is not sure, it does nothing.** `bayalis hazaar` is 42,000, the
tables do not know `bayalis`, and reading the `hazaar` alone as 1,000 would put
a confident wrong figure into a ledger lookup — so it is left exactly as
spoken. A wrong amount is worse than an unconverted one, because a wrong amount
looks like a fact. The same instinct decides what leaves the building: a
reference number would be an excellent hotword, and it is never sent, because a
bias list goes to a third party.

<details><summary><b>Check yourself:</b> Where does the loop eval's "unsafe answers" metric come from at rungs that have no in-loop verifiers?</summary>

It runs the same cheap verifiers after the fact, on the final answer and
everything the loop observed. That is how the ladder can show M1 shipping an
unsupported timeline even though M1 itself never checked.
</details>

---

## Lesson 16 — Context rot

**The idea.** A loop's context grows on its own. Every tool result and every
intermediate step is appended, pass after pass, and long before the hard limit
quality drops: the window fills with stale, low-signal tokens and the agent
loses the thread. That is **context rot**, and it is why long-running agents
start strong and degrade halfway through. So every pass decides what stays,
what is compressed and what is dropped. Context is a resource you budget, like
passes and money.

Three techniques, and one architectural consequence:

1. **Compaction** — fold earlier steps into short summaries.
2. **External memory** — write facts to a store, read back only what's relevant.
3. **Tool result management** — filter, extract and cap a tool's output before
   it enters the context. Never dump a whole record or a thousand rows.
4. **Sub-agents for context isolation** — hand a bounded task to an agent with a
   clean context, and take back only the result. Much of multi-agent
   orchestration is really context management.

**In Sanwaad.** `loop/window.py`:

- **Shaping.** A lookup is cut to reference, kind, amount, status and age, capped
  at three matches, and redacted — raw and shaped sizes are both counted.
- **Scratchpad.** Facts are written as they're observed and shown by relevance.
  A fact survives compaction even when the step that found it doesn't, and
  isn't shown twice while its step is still visible.
- **Compaction.** Above the budget, the oldest steps become one-line summaries;
  if the two most recent steps alone still overflow, fewer remembered facts are
  shown. Recent steps are never cut.

`loop/subagent.py` is the fourth technique. The policy sub-agent holds the
question and its retrieved clauses in its own context and returns one line. In
the walkthrough it consumed **407 tokens and returned 34**.

**Measured.** The long-research scenario asks a six-part policy question: seven
passes of search results. Peak context:

| | no compaction (M1, M2) | compaction (M3+) |
|---|---|---|
| budget 600 | 1,069 | 599 |
| budget 800 | 1,069 | 752 |
| budget 1,000 | 1,069 | 985 |

Without compaction the window just grows; with it, the window tracks whatever
budget you give it.

**Verification is not fooled by compaction.** Verifiers read everything ever
observed, not the compacted view, so dropping a step can never make a true
fact look invented.

<details><summary><b>Check yourself:</b> Why does conversation memory (Lesson 18) keep decisions but never live facts?</summary>

A reversal's status changes; a remembered "initiated" becomes a lie within a day.
What was *decided* — a ticket was opened, a person was asked — stays true. Live
facts are always one tool call away from the system of record (Lesson 5).
</details>

---

## Lesson 17 — Verification asymmetry

**The idea.** Some outputs are cheap and reliable to verify and some aren't, and
that asymmetry shapes the loop. Code either passes its tests or it doesn't — a
cheap, trustworthy verifier — which is exactly why coding agents improved so
fast: the loop checks its own work and corrects itself before anyone looks.
Whether a refund decision was *reasonable* has no such check.

So the rule: **wherever a cheap, reliable verifier exists, put it inside the
loop** and let the agent retry on its feedback. **Wherever verification is
expensive or subjective, put a human there.** Verification is a design
decision made per step, and it marks where automation ends.

**In Sanwaad.** `loop/verify.py` has four cheap in-loop verifiers:

| Verifier | Catches |
|---|---|
| `reply_guardrails` | money promises, banned phrasing, internal clause ids |
| `facts_traced` | an amount or reference the agent never observed |
| `timelines_cited` | a timeline with no retrieved policy clause behind it |
| `citations_retrieved` | a clause cited that was never retrieved |

A failure goes back to the agent as an observation, with bounded retries. And
`needs_human` names the judgements with no cheap check: a validated money move,
and regulatory or fraud language.

**Watch it work.** The offline agent is deliberately eager, like a real model:
asked when a failed ₹2,000 transfer comes back, it drafts the folk answer,
"within 5 to 7 days". `timelines_cited` rejects it because no clause backs it.
The agent consults the policy and answers "within 3 working days", citing RFD-01.

`VERIFICATION_MAP` records every check in Sanwaad — its cost, which loop it
lives in, and what happens when it fails — so the line between automation and
review is written down rather than implied.

<details><summary><b>Check yourself:</b> The grounding check in Part I is model-judged and sits in a retry loop. Why is that acceptable when the rule says cheap verifiers go in loops?</summary>

It's bounded (two redrafts, then a person) and it never has the final say on
anything consequential: validation and approval are code and people. A
model-judged verifier can live in a loop only if its retries are capped and it
isn't the last line of defence.
</details>

---

## Lesson 18 — MINT: Minimal Intelligence, Necessary Tools

**The idea.** Build the minimal system end to end before adding a single layer.
Then add layers one at a time — prompt and workflow, then tool use, then
continuous evaluation, then state and memory, then more workflows, and only at
the end human-in-the-loop and multi-agent — each one only when the layer below
has shown a real need. Every layer answers two questions first: *how does it
break, and what does the system do when it does?* Half the answers will be
ordinary software failures — a timeout, a null, an API that's down — not model
weirdness.

The failure it prevents: five agents, a vector database and long-term memory
wired together before anyone confirmed the basic loop works, then weeks lost
because nobody can tell which layer is failing.

**In Sanwaad.** `loop/mint.py` makes the ladder executable:

- `config_for(rung)` builds each rung's configuration, so the same conversations
  run at every rung.
- `check_layering(config)` **refuses** a configuration that skips a layer —
  multi-agent without evaluation fails at start-up.
- `LADDER` states, for each rung, the need that justified it, how it breaks,
  and what the system does.

`python -m sanwaad.evals.loop_eval --ladder`, offline:

| | M0 minimal | M1 +tools | M2 +evaluation | M3 +memory | M4 +workflows | M5 +HITL, multi-agent |
|---|---|---|---|---|---|---|
| Scenarios passing | 3/13 | 6/13 | 7/13 | 9/13 | 10/13 | 13/13 |
| Unsafe answers shipped | 0 | 1 | 0 | 0 | 0 | 0 |
| Runs over context budget | 0 | 2 | 2 | 0 | 0 | 0 |
| Mean passes | 1.62 | 3.46 | 3.62 | 3.54 | 3.15 | 2.85 |
| Mean cost per run (₹, est.) | 0.0131 | 0.0491 | 0.0515 | 0.0483 | 0.0390 | 0.0376 |
| Sub-agent tokens kept out | 0 | 0 | 0 | 0 | 0 | 757 |

Read it as MINT intends:

- **M1** makes the loop useful — and ships one answer with a timeline nothing
  supports. That's the observed need for evaluation.
- **M2** stops unsupported answers shipping (1 → 0), but long conversations
  still overrun the window. That's the need for memory.
- **M3** holds every run inside its context budget (2 → 0), and a returning
  customer no longer gets a second ticket.
- **M4** workflows cut mean passes from 3.54 to 3.15, mostly by giving a
  wandering agent a budget sized to the job.
- **M5** hands money and legal language to a person, and the sub-agent keeps
  757 tokens of policy out of the main loops.

Mean passes and cost *rise* from M0 to M2 — tools and verification cost
passes — and fall again as memory, workflows and delegation remove wasted ones.

**Honest caveat.** Offline, the scripted policy doesn't wander, so workflows show
their value only on the misbehaving-agent scenario. With a live model, a
narrower set of offered tools is a smaller space to go wrong in — run the
ladder with a key to measure that.

<details><summary><b>Check yourself:</b> Why does <code>check_layering</code> exist if <code>config_for</code> already builds valid rungs?</summary>

Because nobody builds a config through `config_for` when they're in a hurry. The
check makes the shortcut fail loudly at start-up, which is the difference
between a principle and a habit.
</details>

---

## Lesson 19 — The three nested loops

**The idea.** Andrew Ng describes building software with agents as three nested
loops running at different speeds:

| Loop | Speed | Run by |
|---|---|---|
| agent loop | seconds to minutes | the agent: build, test, iterate against a spec |
| developer loop | tens of minutes to hours | you: review, steer, change the spec |
| external loop | hours to weeks | the world: customers, testers, A/B tests, production data |

The same asymmetry as Lesson 17 runs through them: the further out a loop sits,
the more its verification depends on human judgement. The agent can verify its
work against a spec; only real customers can verify that the product is good.
Humans don't disappear — they move to the loops where they hold the context.

**In Sanwaad.**

- **Agent loop:** `loop/kernel.py`, verified by the in-loop checks. Ng's example
  is a coding agent building the product; Sanwaad's inner loop is the product's
  own runtime loop. Same shape — act, verify, iterate against a spec — at a
  different altitude, and both sit inside the same developer and external loops.
- **Developer loop:** the evals — trajectory eval, loop eval, the MINT ladder —
  plus `feedback.py`'s reviewer corrections. You change a prompt, a tool or a
  budget, and these tell you what moved.
- **External loop:** `loop/outer.py`.
  - `trace_report` measures what really ran from the recorded run log.
  - `regression_candidates` turns runs that stalled, needed a verifier, or hit
    a tool error into **draft** scenarios marked `review_required`. A person
    decides the right expectation before one joins the suite, because a failure
    copied blindly becomes a wrong expectation.
  - `assign_variant` splits traffic deterministically — the same customer always
    lands in the same arm, with nothing stored — and `compare_variants` compares
    clean-stop rates with a p-value, refusing to call a winner below 30 runs per
    arm.

After the walkthrough, `python -m sanwaad.loop.outer` reads those five runs and
selects three for review: the ledger outage, the timeline correction and the
stalled agent.

<details><summary><b>Check yourself:</b> Why does <code>assign_variant</code> hash the experiment name together with the customer?</summary>

So the same customer lands in independent arms across different experiments.
Hashing the customer alone would put the same people in "treatment" for every
experiment, and their quirks would contaminate every comparison.
</details>

---

## Lesson 20 — The four agentic design patterns

**The idea.** Andrew Ng's four original agentic design patterns — reflection,
tool use, planning and multi-agent collaboration — aren't abstract categories.
They're the building blocks of the loop you just engineered.

| Pattern | In Sanwaad's support loop | In the case graph (Part I) |
|---|---|---|
| **Reflection** | the draft is checked against the verifiers, and the agent retries on feedback | `ground_check` → redraft |
| **Tool use** | lookup, reversal status, policy search, tickets, through the registry | the planner's ledger lookup, publishing, executing |
| **Planning** | the policy sequences classify → look up → check → answer or hand over | `plan` proposes the fix; the graph fixes the order |
| **Multi-agent** | the policy sub-agent answers in a clean context and returns one line | triage, pattern, judge, ghostwriter, planner under one orchestrator |

**What to take from Part II.**

- Stop treating the agent as its best component. Look at the whole loop.
- Design the stopping conditions, and budget passes, cost and context on purpose.
- Take the harness and system-level evaluation as seriously as the model.
- Put cheap verifiers inside the loop, and people where verification is expensive.
- Build up from a minimal end-to-end system, one measured layer at a time.

---

## Part II checklist

| Requirement | Sanwaad | Proven by |
|---|---|---|
| Explicit loop primitive | `loop/kernel.py` | `tests/test_loop.py` · kernel |
| Several stopping conditions at once | `loop/budget.py` | runaway and wandering scenarios |
| Loop cost visible on every run | `Meter` | `loop_eval` cost per run |
| Context budget, compaction, memory, shaping | `loop/window.py` | long-research scenario, window tests |
| Sub-agent context isolation | `loop/subagent.py` | tokens kept out |
| Cheap verifiers in the loop, humans elsewhere | `loop/verify.py` | timeline self-correction |
| System-level evaluation under stress | `evals/loop_eval.py` | 13 scenarios, safety checks |
| Layers added one at a time, enforced | `loop/mint.py` | the ladder, layering tests |
| Real runs feed the spec | `loop/outer.py` | candidates, A/B tests |

## Part II exercises

7. **Per-workflow cost ceilings.** Give each workflow its own `max_cost_inr` and
   add a scenario proving a refund workflow may cost more than a policy question.
8. **Model-written compaction.** Replace the one-line extractive summaries with a
   model summariser, then use the long-research scenario to check whether the
   facts the final answer needs still survive.
9. **Run a real A/B test.** With a key, route half of your test customers to a
   variant with sub-agents switched off. Collect at least 30 runs per arm, then
   read `compare_variants`. Is the difference real?
10. **Close the outer loop.** Take one `regression_candidates` entry from a live
    run, decide its correct outcome, and add it to `loop_eval.SCENARIOS`.
11. **A model-judged tone check.** Add a `MODEL`-cost verifier for brand voice.
    Decide, and justify in a comment, whether it belongs inside the loop or in
    sampled offline evaluation.
