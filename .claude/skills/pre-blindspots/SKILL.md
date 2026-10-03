---
name: pre-blindspots
description: Use when you're about to work in unfamiliar code and want the unknown-unknowns surfaced before you write the prompt or the first line. Produces a single self-contained, interactive HTML page of "blindspot cards" — each a hidden assumption or risk with a copyable prompt refinement — that assemble into an improved implementation prompt.
---

# Blindspot Pass

Before I start on this change, do a **blindspot pass**: hunt for the things I don't yet know I need to worry about, and hand them to me as concrete, actionable cards.

The premise: the map is not the territory, and the gap between them is where I'll get hurt. Your job is to shrink that gap *before* implementation, not explain the wreckage after.

## What to look for

Explore the relevant code and surrounding system, then surface the hidden stuff a first-pass prompt would miss. Aim for a spread across categories such as:

- **Silent assumptions** in my request that the code contradicts.
- **Coupling and blast radius** — callers, invariants, or shared state this change could break.
- **Edge cases and failure modes** — nulls, concurrency, partial failure, migrations, backwards compatibility.
- **Conventions and constraints** — existing patterns, auth, error handling, or perf budgets I'd otherwise violate.
- **Missing context** — a config, feature flag, or dependency I don't know exists.

## What to produce

- Around **seven blindspot cards**. Each card: a short title, 1–2 sentences on the risk (with a specific file/function reference so I can verify it), and a **copyable prompt refinement** — the exact sentence to add to my implementation prompt to defuse it.
- Rank the cards by how badly getting them wrong would hurt.
- At the bottom, an **assembled implementation prompt**: my original intent plus every refinement folded in, in one copyable block, so I can take it straight to an agent.
- Let me dismiss cards I judge irrelevant, and have the assembled prompt update to exclude them.

## Format

- Output a **single self-contained HTML file** with inline CSS and JavaScript — no external dependencies, works offline by double-clicking. Basic responsive styling for a phone is nice.
- Save it outside the code repo with today's date prefixed in `YYYY-MM-DD-` format, so files stay time-sorted and out of version control. For example: `/tmp/2026-01-12-blindspots-<short-name>.html`
- Cards, code excerpts, and the assembled prompt use simple HTML/CSS — never ASCII art. Any code goes in `<pre>` tags (or a div with `white-space: pre-wrap`, or the browser eats the newlines). The refinement on each card and the final prompt each need a copy button.
- Ground every card in real code you actually read. A vague, generic blindspot is worse than none — specificity is the whole point.
