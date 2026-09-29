---
name: handoff
description: Write a session handoff note for sanwaad so the next session can pick up without re-reading the conversation. Use when the user types /handoff, or says they are stopping, switching sessions, or want a summary of where things stand.
---

# Handoff

Write `HANDOFF.md` at the repo root (overwrite the previous one), then show
the user its path and the "Next step" line.

Gather facts, don't recall them: run `git status --short`, `git log --oneline -5`,
`git diff --stat`, and read `.claude/.gate` (a `green` fingerprint means the
last full test run passed). Only write down what you checked this session.

Keep it under 60 lines, in this shape:

```markdown
# Handoff — <YYYY-MM-DD>

## Goal
<one or two lines: what the user is trying to achieve, in their words>

## Done this session
- <change> — <file:line> — verified by <test / command / "not verified">

## In progress / uncommitted
<git status summary; what is half-finished and why>

## Decisions made (and by whom)
- <decision> — <user chose X / Claude defaulted to X because …>

## Open questions for the user
- <anything waiting on them: data, keys, a choice>

## Known problems
- <bugs found and not fixed, flaky tests, environment limits>

## Next step
<the single most useful thing to do first next session>
```

Rules:
- No secrets, API keys or customer text in the note.
- Say plainly what was not verified. "Tests pass" only if you saw them pass.
- Don't commit HANDOFF.md unless the user asks.
