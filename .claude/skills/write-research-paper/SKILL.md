---
description: Draft or edit any part of the research paper (abstract, intro, methods, results, conclusion, impact statement, appendix). Use for both new prose and edits to existing paper/paper.tex content.
allowed-tools: Read Write Edit Bash(*)
---

## Scope

Applies to `paper/paper.tex` and any section within it. Edit the `.tex` file directly — do not draft in a separate file and paste in unless the user asks for that.

## Before writing

- Read the surrounding sections of `paper.tex` to match tone, tense, and terminology already in use.
- If the section references a method, dataset, or paper covered in `knowledge_base/wiki/`, cross-link it there (see `AGENTS.md` §13) and pull facts from the wiki page rather than re-deriving them.
- Check `knowledge_base/` for prior notes relevant to the claim being written before asserting anything about related work, datasets, or results.

## Writing rules

- Precise, falsifiable claims only. No unsupported superlatives ("state-of-the-art", "significantly", "novel") unless backed by a cited result or table in this paper.
- Match the paper's existing person/tense (check nearby sections — do not switch between "we" and passive voice mid-paper).
- Keep paragraphs short. One claim/idea per paragraph in results and impact sections.
- Do not invent citations, numbers, or results. If a number is needed and not yet available, leave a `% TODO:` LaTeX comment instead of a placeholder value.
- Do not fill sections with filler text (`\lipsum`, generic template prose) — replace it as content is written, remove it if not.

## After writing

- Run the `stop-slop` skill over any new or edited prose before treating it as done.
- If the change adds/removes a section, check whether `\tableofcontents` or cross-references (`\ref{}`) elsewhere in the file need updating.
