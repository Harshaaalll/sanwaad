// Tests for the review console's own logic, run with Node's built-in runner:
//   node --test tests/js/
// (tests/test_console_js.py runs this as part of pytest, so CI and the commit
// gate include it.)
//
// The console's script lives inline in console.html. It is loaded here as the
// page's actual source into a vm context with a few-line fake DOM, so these
// tests run exactly the code browsers run, with no build step and no dependency.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const HTML = fs.readFileSync(path.join(__dirname, "../../sanwaad/api/static/console.html"), "utf8");
const SCRIPT = HTML.match(/<script>([\s\S]*)<\/script>/)[1];
const EXPORTS = ["statusOf", "inFilter", "catLabel", "esc", "ago", "load", "renderList", "whyCard",
                 "draftChanged", "review", "rejectCase"];

function fakeElement(id) {
  const classes = new Set();
  return {
    id, innerHTML: "", textContent: "", value: "", defaultValue: "", disabled: false, title: "",
    hidden: false, dataset: {}, isConnected: true, children: [],
    classList: {add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c),
                toggle: (c, on) => (on ?? !classes.has(c)) ? classes.add(c) : classes.delete(c)},
    setAttribute() {}, getAttribute() { return null; }, focus() {}, scrollTo() {}, remove() {},
    append(child) { this.children.push(child); }, closest() { return null; }, contains() { return false; },
  };
}

// A fresh page per test: its own elements, fetch log and toasts.
function page({responses = {}} = {}) {
  const elements = {}, selectorResults = {}, fetches = [];
  const body = fakeElement("body");
  const document = {
    body, activeElement: body, hidden: false, documentElement: {dataset: {}},
    getElementById: id => (elements[id] ??= fakeElement(id)),
    querySelectorAll: sel => selectorResults[sel] || [],
    querySelector: () => null,
    createElement: () => fakeElement(""),
    addEventListener() {},
  };
  const fetch = async (url, opts = {}) => {
    fetches.push({url, opts});
    const body = typeof responses[url] === "function" ? responses[url](opts) : responses[url];
    const status = body && body.__status || 200;
    return {ok: status < 400, status, json: async () => body ?? {}};
  };
  const ctx = {
    document, fetch, console, setTimeout: () => 0, setInterval: () => 0, Date, JSON, Math, Object,
    String, Number, Array, Set, Promise, RegExp, Error,
    CSS: {escape: s => s}, matchMedia: () => ({matches: false}),
    localStorage: {getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }},
  };
  vm.createContext(ctx);
  // The trailing load() call is the page booting; tests call load themselves.
  vm.runInContext(SCRIPT.replace(/\nload\(\);\s*$/, "\n") +
    `\nglobalThis.__api = {${EXPORTS.join(", ")}, get cases() { return cases; },
      get filter() { return filter; }, set filter(v) { filter = v; }, set selected(v) { selected = v; }};`, ctx);
  const toasts = () => elements.toasts ? elements.toasts.children.map(t => t.textContent) : [];
  return {api: ctx.__api, el: id => document.getElementById(id), selectorResults, fetches, toasts};
}

// --- statuses: what a reviewer sees on each case ----------------------------------

test("each case shows the status its pipeline state means", () => {
  const {api} = page();
  const cases = [
    [{draft: {text: "x"}}, "needs_review"],
    [{draft: {}, review: {}, escalation: {needed: true}}, "awaiting_call"],
    [{draft: {}, review: {}, escalation: {needed: false}}, "in_progress"],
    [{closure: {resolved: true}}, "resolved"],
    [{closure: {resolved: false}}, "closed_unresolved"],
    [{closure: {resolved: false}, priority: {tier: "ignore"}}, "logged"],
  ];
  for (const [state, expected] of cases) assert.equal(api.statusOf(state), expected, JSON.stringify(state));
});

test("the Closed filter holds every finished state and nothing open", () => {
  const {api} = page();
  for (const s of ["resolved", "closed_unresolved", "logged"]) assert.ok(api.inFilter({_status: s}, "closed"), s);
  for (const s of ["needs_review", "awaiting_call", "in_progress"]) assert.ok(!api.inFilter({_status: s}, "closed"), s);
  assert.ok(api.inFilter({_status: "logged"}, "all"));
});

test("internal category ids never reach the screen", () => {
  const {api} = page();
  assert.equal(api.catLabel("off_topic"), "Not about us");
  assert.equal(api.catLabel("data_privacy"), "Fraud & privacy");
  assert.equal(api.catLabel("new_category"), "new category");   // an unknown id still reads as words
  assert.equal(api.catLabel(undefined), "Not yet classified");
});

test("ago() gives short human ages", () => {
  const {api} = page();
  const before = s => new Date(Date.now() - s * 1000).toISOString();
  assert.equal(api.ago(before(20)), "just now");
  assert.equal(api.ago(before(5 * 60)), "5 min ago");
  assert.equal(api.ago(before(3 * 3600)), "3 h ago");
  assert.equal(api.ago(before(2 * 86400)), "2 d ago");
  assert.equal(api.ago(null), "");
});

// --- the queue ------------------------------------------------------------------

const CASES = [
  {case_id: "low", complaint: {author: "u/a", text: "fee question", channel: "reddit"},
   triage: {category: "billing", severity: 2}, priority: {score: 20}, draft: {text: "x"}},
  {case_id: "crisis", complaint: {author: "u/b", text: "app down", channel: "reddit"},
   triage: {category: "service_outage", severity: 3}, priority: {score: 30}, pattern: {level: "crisis", cluster_size: 6},
   draft: {text: "x"}},
  {case_id: "high", complaint: {author: "u/c", text: "wallet frozen", channel: "mock"},
   triage: {category: "account_access", severity: 4}, priority: {score: 60}, draft: {text: "x"}},
  {case_id: "done", complaint: {author: "u/d", text: "thanks", channel: "mock"},
   triage: {category: "praise", severity: 1}, priority: {score: 99, tier: "ignore"}, closure: {resolved: false}},
];

function queuePage(cases = CASES) {
  return page({responses: {"/api/cases": {cases}, "/ready": {ready: true}, "/api/delivery": {dead: []}}});
}

test("the queue puts incidents first, then the highest priority score", async () => {
  const {api} = queuePage();
  await api.load();
  assert.deepEqual(api.cases.map(c => c.case_id), ["crisis", "done", "high", "low"]);
});

test("the queue opens on 'Needs review' when anything is waiting, else on 'All'", async () => {
  let p = queuePage();
  await p.api.load();
  assert.equal(p.api.filter, "needs_review");
  p = queuePage([CASES[3]]);
  await p.api.load();
  assert.equal(p.api.filter, "all");
});

test("filter chips count every case in their state", async () => {
  const {api, el} = queuePage();
  await api.load();
  const chips = el("filters").innerHTML;
  assert.match(chips, /aria-label="Needs review, 3 cases"/);
  assert.match(chips, /aria-label="Closed, 1 case"/);
  assert.match(chips, /aria-label="All, 4 cases"/);
});

test("search matches the plain-language category, not just the raw text", async () => {
  const {api, el} = queuePage();
  await api.load();
  el("search").value = "account access";
  api.renderList();
  assert.match(el("list").innerHTML, /data-id="high"/);
  assert.doesNotMatch(el("list").innerHTML, /data-id="low"/);
  assert.match(el("qcount").textContent, /^1 of 4 cases match/);
});

test("a hostile complaint is shown as text, never run as HTML", async () => {
  const evil = {case_id: "x", complaint: {author: "<img src=x onerror=alert(1)>", text: "<script>steal()</script>",
                                          channel: "reddit"}, triage: {}, draft: {text: "x"}};
  const {api, el} = queuePage([evil]);
  await api.load();
  const html = el("list").innerHTML;
  assert.doesNotMatch(html, /<script>|<img/);
  assert.match(html, /&lt;script&gt;steal\(\)&lt;\/script&gt;/);
  assert.equal(api.esc(`"'`), "&quot;&#39;");      // attributes are safe too
});

test("an empty queue explains how to start", async () => {
  const {api, el} = queuePage([]);
  await api.load();
  assert.match(el("list").innerHTML, /No complaints yet/);
});

// --- deciding on a case -----------------------------------------------------------

test("editing the reply swaps Approve for Post my edited reply", () => {
  const {api, el} = page();
  Object.assign(el("draft"), {defaultValue: "Sorry about this.", value: "Sorry about this."});
  api.draftChanged();
  assert.equal(el("btn-approve").disabled, false);
  assert.equal(el("btn-edit").disabled, true);
  el("draft").value = "Sorry — we have refunded it.";
  api.draftChanged();
  assert.equal(el("btn-approve").disabled, true, "approving would post text the reviewer changed");
  assert.equal(el("btn-edit").disabled, false);
});

test("a money action is approved only when its box is ticked", async () => {
  const {api, el, selectorResults, fetches, toasts} = page({responses: {
    "/api/cases/c1/review": {}, "/api/cases": {cases: []}, "/api/cases/c1": {state: {}}}});
  el("draft").value = "reply";
  selectorResults[".act-approve"] = [{dataset: {id: "refund-1"}, checked: true}, {dataset: {id: "refund-2"}, checked: false}];
  selectorResults[".act-approve:checked"] = [selectorResults[".act-approve"][0]];
  await api.review("c1", "approve");
  const sent = JSON.parse(fetches.find(f => f.url === "/api/cases/c1/review").opts.body);
  assert.deepEqual(sent.actions, {"refund-1": "approve", "refund-2": "reject"});
  assert.equal(sent.decision, "approve");
  assert.equal(toasts()[0], "Reply approved and posted. 1 money action(s) approved.");
});

test("a failed decision says nothing was posted and lets the reviewer retry", async () => {
  const {api, el, selectorResults, toasts} = page({responses: {
    "/api/cases/c1/review": {__status: 500, detail: "server error"}}});
  const button = el("btn-approve");
  selectorResults[".decide button"] = [button];
  await api.review("c1", "approve");
  assert.match(toasts()[0], /Couldn't save your decision \(server error\)\. Nothing was posted/);
  assert.equal(button.disabled, false);
});

test("Reject needs a second click before anything is sent", async () => {
  const {api, fetches} = page({responses: {"/api/cases/c1/review": {}, "/api/cases": {cases: []},
                                          "/api/cases/c1": {state: {}}}});
  const btn = {classList: new Set(), textContent: "Reject reply", isConnected: true};
  btn.classList.contains = c => btn.classList.has(c);
  btn.classList.remove = c => btn.classList.delete(c);
  api.rejectCase("c1", btn);
  assert.equal(fetches.length, 0, "the first click only arms the button");
  assert.equal(btn.textContent, "Confirm: reject reply");
  api.rejectCase("c1", btn);
  await new Promise(r => setImmediate(r));
  assert.ok(fetches.some(f => f.url === "/api/cases/c1/review" && JSON.parse(f.opts.body).decision === "reject"));
});

// --- explanations -----------------------------------------------------------------

test("the why-card marks the first failing rule as the reason given", () => {
  const {api} = page();
  const html = api.whyCard({
    priority: {tier: "routine", score: 43.1, components: {severity: 30, authenticity: 8}, reasons: [], formula: "f"},
    author: {class: "customer", authenticity: 1, reach: 75, evidence: [],
             bands: {troll_at_or_below: .35, audience_at_or_above: .55, answer_if_authenticity_at_least: .35,
                     or_reach_at_least: 5000}},
    review: {checks: [{rule: "no money-moving action", passed: false, detail: "money", kind: "hard"},
                      {rule: "needs no private data", passed: false, detail: "private", kind: "hard"}],
             allowed_now: false, at_the_time: {allowed: false, reason: "money"}},
    escalation: {checks: [{rule: "severity ≥ 4", fired: false, detail: ""}]},
  });
  assert.match(html, /no money-moving action \(reason given\)/);
  assert.doesNotMatch(html, /needs no private data \(reason given\)/, "only the first failure is the reason");
  assert.match(html, /No trigger fired, so no call is offered/);
});
