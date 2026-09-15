# Agentic AI System Design, learned through Sanwaad

Sanwaad is a working agentic system. This guide uses it to teach how
production-grade agentic systems are designed: one building block at a time,
each one tied to code you can open, run and break.

The lessons follow the building blocks in Aishwarya Srinivasan's talk on
agentic AI system design ([video](https://youtu.be/mwN75EiGfCE)). Here each
idea is shown working in real code rather than restated.

**How to use this guide.** For each lesson, read the idea, open the files it
points to, run the command, then answer the check question before you expand
the answer.

```bash
source .venv/bin/activate
python -m sanwaad.demo                 # the whole system, no API key needed
python -m sanwaad.evals.trajectory     # grade every step of 15 scenarios
python -m sanwaad.memory               # what is remembered, where, for how long
python -m sanwaad.router               # which model runs each step, and its fallback
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
