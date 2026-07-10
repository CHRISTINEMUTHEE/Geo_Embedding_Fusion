---
name: pre-plan
description: Use when you want an implementation plan you can steer — organized by what's most likely to change, not by execution order. Produces a single self-contained, interactive HTML plan where each real decision is surfaced with toggleable alternatives and the mechanical work is collapsed at the bottom.
---

# The Tweakable Plan

Give me an implementation plan built for *me to change*, not just to read and approve. Most plans bury the one choice that matters under twenty steps of boilerplate. Invert that.

## Organizing principle

Sort the plan by **likelihood-of-change, not execution order.** The decisions most likely to shift once we hit reality go at the top, loud and editable. The mechanical, only-one-way-to-do-it work goes at the bottom, collapsed. I want to spend my attention exactly where a different call changes the outcome.

## What the plan must contain

- **Decision points at the top.** For each genuine fork (schema shape, data model, an interface, sync vs async, a library choice), show:
  - the option you'd pick and a one-line why;
  - 1–3 **toggleable alternatives** I can flip — and when I flip one, show how it ripples (which later steps or type signatures change);
  - key type/interface definitions annotated inline, so I see the contract, not just the prose.
- **The build steps**, ordered for execution, in the middle — each tied back to the decisions it depends on.
- **Mechanical work collapsed** at the bottom (imports, wiring, test scaffolding) — present but out of the way, expandable if I care.

## Format

- Output a **single self-contained HTML file** with inline CSS and JavaScript — no external dependencies, works offline by double-clicking. Basic responsive styling for a phone is nice.
- Save it outside the code repo with today's date prefixed in `YYYY-MM-DD-` format, so files stay time-sorted and out of version control. For example: `/tmp/2026-01-12-plan-<short-name>.html`
- Decision toggles should visibly update the affected steps or interfaces when flipped — this is the whole point of the artifact, so make the ripple real, not cosmetic.
- Explore the surrounding code first so the decisions, types, and steps reflect the actual codebase. Code and type signatures go in `<pre>` tags (or a div with `white-space: pre-wrap`).
- Prefer a plan I can operate over a pretty one I can only skim.
