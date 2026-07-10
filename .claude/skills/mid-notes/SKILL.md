---
name: mid-notes
description: Use during a build to keep a running log of every deviation from the plan and the conservative call you made, so I can review and steer without restarting from scratch. Produces and continuously updates a dated markdown notes file alongside the work.
---

# Implementation Notes

While you implement this change, keep a running **implementation log**. The plan will meet reality and lose; I want to see where it bent and why, in real time, so I can course-correct instead of discovering it all at the end.

## How to keep the log

- Create the notes file at the start, before the first edit, and **append to it as you go** — not as a write-up at the end. Each entry is cheap; write it the moment a decision happens.
- Log an entry every time you **deviate from the plan or hit a fork the plan didn't anticipate**: what you expected, what you actually found, the options, and **the call you made — defaulting to the conservative, reversible choice** when unsure.
- Also note anything I'd want to revisit: an assumption you had to make, a shortcut taken under time pressure, a TODO you're deferring, a place the plan was simply wrong.
- Keep entries short — a few lines each. This is a flight recorder, not an essay.

## What the log must contain

- A short **header**: the change, the plan you started from, and where the notes file lives.
- A **timeline of entries**, each: what deviated, the decision, and one line of rationale.
- At the end, **three bullet points** distilling what you'd fold into the next attempt if we redid this from scratch — the durable lessons, not the play-by-play.

## Format

- Write a plain **markdown file**, saved outside the code repo with today's date prefixed in `YYYY-MM-DD-` format so it stays time-sorted and out of version control. For example: `/tmp/2026-01-12-notes-<short-name>.md`
- Tell me the file path up front, and update the file live as you work rather than reconstructing it afterward — the value is in the honest, in-the-moment record.
- When you finish, point me at the file and read me the three closing bullets.
