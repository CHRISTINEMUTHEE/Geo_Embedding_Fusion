---
name: pre-interview
description: Use when a feature or request is ambiguous and you want the important decisions pinned down before any code is written. Runs a short interview — questions ordered by architectural blast-radius — and produces a decisions table plus a ready-to-paste implementation prompt.
---

# Interview Me Before You Build

Before writing any code, interview me to surface the decisions this change actually hinges on. The goal is to turn my vague request into a precise spec — and to make me, not you, choose where it matters.

## How to run the interview

- First, silently explore the relevant code so your questions are grounded in how the system actually works, not generic.
- Ask me **one question at a time**, and wait for my answer before the next. Don't dump a list.
- **Order questions by architectural blast-radius**: the decision that constrains the most downstream work comes first (data model, boundaries, sync vs async), mechanical preferences last. If an early answer makes a later question moot, skip it.
- For each question, give me 2–4 concrete options with a one-line trade-off each, and mark the one you'd default to — so I can just say "default" and move on. Always offer an escape hatch ("not sure — you pick").
- Keep it short. Stop once the remaining unknowns are low-stakes; don't interrogate me over things a sensible default settles.

## What to produce at the end

- A **Decisions table**: one row per question — the decision, what I chose, and the one-line reason. This is the record of what we agreed.
- A **ready-to-paste implementation prompt**: a self-contained brief that folds every decision in, so I can hand it to a fresh agent (or to you) and get exactly the change we scoped. Write it so it stands alone without this conversation.
- Flag any decision I deferred or that still feels risky, so it stays visible rather than buried.

## Format

- This is a conversation, not a document — run it in chat. Only the final decisions table and implementation prompt are the deliverable; render them as plain markdown so I can copy them.
- Match the number of questions to the stakes: a small change might need two; a new subsystem might need eight. Don't pad.
