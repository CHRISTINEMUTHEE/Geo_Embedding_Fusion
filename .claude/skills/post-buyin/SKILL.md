---
name: post-buyin
description: Use when a finished change needs sign-off and you want to make the reviewer's yes easy. Produces a single self-contained, interactive HTML pitch that leads with a demo, pre-answers likely objections with evidence, and names the approvers required to merge.
---

# The Buy-in Doc

The change is built; now help me get it merged. Make me a **buy-in document** aimed at the people who have to approve it — one that leads with the result, anticipates the pushback, and makes saying yes low-effort.

## What the doc must contain

- **Lead with the demo.** Open with the change *working* — an interactive walkthrough, a before/after, or a short scripted click-through of the new behavior — not paragraphs of preamble. Show the win in the first screen.
- **The case, briefly**: what problem this solves and why now, in a few sentences. Tie the demo to the motivation.
- **Objections, pre-answered.** List the concerns a sharp reviewer will raise (risk, scope, perf, migration, "why not simpler?") and answer each with **evidence** — a test result, a benchmark, a code reference, a scoped-out non-goal. Anticipating the objection is what earns trust.
- **What's explicitly out of scope**, so reviewers don't block on things I chose not to do.
- **Named approvers**: who needs to sign off and what each of them will care about. Make the path to merge concrete.

## Format

- Output a **single self-contained HTML file** with inline CSS and JavaScript — no external dependencies, works offline by double-clicking. Basic responsive styling for a phone is nice.
- Save it outside the code repo with today's date prefixed in `YYYY-MM-DD-` format, so files stay time-sorted and out of version control. For example: `/tmp/2026-01-12-buyin-<short-name>.html`
- The demo is the centerpiece — make it genuinely interactive or visual (real toy data, not lorem ipsum), not a screenshot with a caption. Objections and evidence read best as a two-column table.
- Diagrams and the demo use simple HTML/CSS — never ASCII art. Any code or diff goes in `<pre>` tags (or a div with `white-space: pre-wrap`).
- Explore the actual change and its tests first so every claim and piece of evidence is real. Write in a confident, plain, evidence-first voice — persuasion by transparency, not hype.
