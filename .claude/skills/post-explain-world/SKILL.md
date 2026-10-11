---
name: post-explain-world
description: Use when the user wants to *understand* a code change by inhabiting it, not just reading about it. Produces a single self-contained, interactive "micro-world" — a small app where the user drives the change step by step and watches its effects unfold.
---

# Explain Diff as a Micro-World

Please build me a **micro-world** for the specified code change: a small environment I step into and operate to build my own intuition for what the change does.

The guiding idea (from Seymour Papert): if you want to learn math, go live in Mathland. So instead of *telling* me what the change does, build a place where I *do* it and see it happen. The difference between watching you debug and debugging it myself is where understanding actually comes from — so make me the one pressing the buttons.

## The core principle

Don't produce a document to read. Produce a **thing to operate.**

- I should be the agent in the world. I press "next", I scrub the timeline, I flip the input — the world reacts.
- The change's effect must be **visible**: state on screen, before/after side by side, a value updating, a file tree growing.
- Understanding comes from *me* driving. Never auto-play the whole thing; make every step my choice.

## Pick the world that fits the change

Choose the smallest world that makes the change's essence visible. Common shapes:

- **Time-scrubber** — for changes with a sequence of steps (an algorithm, a request lifecycle, a state machine, an interpreter). Show a timeline I can step and scrub through. At each step show the relevant state (the stack, the variables, which branch/rule fired). Let me leave a note on a step.
- **Before/after command center** — for migrations, refactors, or rewrites. Old behavior and new behavior run side by side. I click to apply the change one step at a time and watch both panels — and any file tree or output — evolve together.
- **Input playground** — for changes to a function or transform. I edit the input (sliders, toggles, a text box) and watch old-vs-new output update live, so I feel the boundary of what actually changed.

If one shape doesn't fit, invent one — but keep it operable and keep the effect visible.

## What the world must contain

- **A one-screen orientation**: what this change is and what I'm about to do, in two or three sentences. Then get out of the way.
- **The interactive world itself**: the main event. Real toy data flowing through it, not lorem ipsum. Make the state legible.
- **A "what just happened" cue**: after each action, a short, plain-language line explaining the effect I just caused. This is the bridge from *doing* to *understanding*.
- **A quick self-check**: 2–3 questions I can only answer if operating the world taught me something. Interactive — clicking an answer tells me if I'm right and why.

## Format

- Output a **single self-contained HTML file** with inline CSS and JavaScript — no external dependencies, works offline by double-clicking. Basic responsive styling so it's usable on a phone.
- Save it outside the code repo with today's date prefixed in `YYYY-MM-DD-` format, so files stay time-sorted and out of version control. For example: `/tmp/2026-01-12-world-<short-name>.html`
- Explore the surrounding code first so the toy data and the states you show are real, not invented.
- Diagrams and state displays are simple HTML/CSS (boxes, lists, grids) — never ASCII art. Code shown in the world goes in `<pre>` tags (or a div with `white-space: pre-wrap`, or the browser eats the newlines).
- Prefer clarity over polish. A plain world I can operate beats a beautiful one I only watch.
