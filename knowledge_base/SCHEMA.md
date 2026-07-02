# Knowledge Base Schema

This document governs how the knowledge base is structured, maintained, and used. The agent reads this file before any KB operation.

## Architecture

See `README.md` for the full directory tree. Three layers:

1. **Sources** (`sources/`) — Immutable raw inputs. The agent reads but never modifies, deletes, or moves files.
2. **Wiki** (`wiki/`) — Agent-generated and maintained markdown pages.
3. **Schema** (this file) — Conventions, templates, and workflows.

Two operational files: **`index.md`** (content catalog, updated after every operation) and **`log.md`** (append-only chronological record).

## Page Templates

Every wiki page uses YAML frontmatter followed by markdown content.

### Paper (`wiki/papers/`)

```markdown
---
title: "{Paper Title}"
authors: [First Author, Second Author]
year: YYYY
venue: "Conference/Journal"
source: "sources/papers/{filename}"
tags: [tag1, tag2]
date_created: YYYY-MM-DD
date_updated: YYYY-MM-DD
---

## Summary
2-3 sentence summary of the paper's contribution.

## Method
Description of the approach.

## Key Results
Main findings, with numbers where available.

## Limitations
Stated or inferred limitations.

## Relevance
How this paper connects to the current project.

## Related
- Concepts: [[concept_name]]
- Methods: [[method_name]]
- Datasets: [[dataset_name]]
```

### Concept (`wiki/concepts/`)

```markdown
---
title: "{Concept Name}"
tags: [tag1, tag2]
date_created: YYYY-MM-DD
date_updated: YYYY-MM-DD
---

## Definition
Clear, concise definition.

## Details
Deeper explanation, variants, key properties.

## Papers
- [[paper_name]] — how this paper relates to the concept

## Related Concepts
- [[other_concept]] — relationship
```

### Method (`wiki/methods/`)

```markdown
---
title: "{Method Name}"
tags: [tag1, tag2]
date_created: YYYY-MM-DD
date_updated: YYYY-MM-DD
---

## Overview
What it does, when to use it.

## Details
How it works, key parameters, implementation notes.

## Strengths and Weaknesses
- Strengths: ...
- Weaknesses: ...

## Papers
- [[paper_name]] — introduced/extended/applied this method

## Related
- Concepts: [[concept_name]]
- Methods: [[related_method]]
```

### Dataset (`wiki/datasets/`)

```markdown
---
title: "{Dataset Name}"
tags: [tag1, tag2]
date_created: YYYY-MM-DD
date_updated: YYYY-MM-DD
---

## Overview
What the dataset contains, its purpose.

## Statistics
- Samples: ...
- Splits: train/val/test ...
- Resolution/dimensions: ...

## Known Issues
Biases, labeling errors, versioning concerns.

## Access
How to download, license, preprocessing steps.

## Papers
- [[paper_name]] — used/introduced this dataset
```

### Synthesis (`wiki/synthesis/`)

```markdown
---
title: "{Topic}"
tags: [tag1, tag2]
date_created: YYYY-MM-DD
date_updated: YYYY-MM-DD
---

## Current Understanding
Integrated narrative synthesizing multiple sources.

## Open Questions
- Question 1
- Question 2

## Evidence
- Claim — supported by [[paper_1]], [[paper_2]]

## Gaps
What is missing from the literature or our understanding.
```

## Naming Conventions

- All filenames: `snake_case`, lowercase, no spaces
- Papers: `{first_author}_{year}_{short_title}.md` (e.g., `vaswani_2017_attention.md`)
- Concepts/methods/datasets: `{descriptive_name}.md` (e.g., `self_attention.md`, `imagenet.md`)
- Synthesis: free-form descriptive names (e.g., `evolving_thesis.md`, `gap_analysis.md`)
- Source files in `sources/`:
  - Papers: `{first_author}_{year}_{short_title}.pdf`
  - Articles: `{source}_{date}_{short_title}.md` (e.g., `distill_2020_attention_viz.md`)
  - Notes: no naming convention enforced
  - Assets: descriptive filenames with context

## Cross-References

- Use `[[page_name]]` notation (filename without extension and without directory prefix)
- Example: `[[vaswani_2017_attention]]` refers to `wiki/papers/vaswani_2017_attention.md`
- Cross-references must be **bidirectional**: if page A links to page B, page B must link back to page A
- The agent resolves `[[page_name]]` by searching across all wiki subdirectories
- **Filenames must be unique across all wiki subdirectories.** If `attention.md` exists in `concepts/`, do not create another `attention.md` in `methods/` — use a distinct name like `attention_mechanism.md`

## Workflows

### Ingest

Triggered when a new source is added to `sources/`.

1. Read the source file from `sources/` (use PDF reader for PDFs, image viewer for images, direct read for markdown/text)
2. Create a summary page in the appropriate `wiki/` subdirectory using the correct template
3. Identify entities mentioned in the source: concepts, methods, datasets
4. For each entity: if a wiki page exists, update it with the new reference. If no page exists, create one
5. Update `index.md` with all new and modified pages
6. Append to `log.md`: `## [YYYY-MM-DD] ingest | {source_name}` followed by list of created/updated files

For long PDFs (>20 pages): read abstract, introduction, method, results, and conclusion first. Read remaining sections only if needed for specific entities.

### Query

Triggered when the user asks a research question.

1. Read the user's question
2. Read `index.md` to identify relevant wiki pages
3. Read those pages and synthesize an answer
4. If the answer is substantial and reusable, file it as a new page in `wiki/synthesis/`
5. Append to `log.md`: `## [YYYY-MM-DD] query | {question_summary}` followed by answer location or "inline"

### Lint

Triggered when the user requests a knowledge base health check.

1. Read all pages in `wiki/` (use index.md as starting point, then scan directories for unlisted pages)
2. Check for:
   - Broken `[[cross-references]]` pointing to nonexistent pages
   - Orphan pages not listed in `index.md`
   - Index entries pointing to nonexistent files
   - Pages missing required template sections (per the templates above)
   - Missing bidirectional links (A links to B but B doesn't link back to A)
   - Concepts or methods mentioned in text but lacking their own page
3. Fix automatically: add missing index entries, add missing bidirectional links, fill obvious template gaps
4. Report to user: contradictions between pages, stale claims, suggested new pages, suggested new sources to find
5. Append to `log.md`: `## [YYYY-MM-DD] lint | summary` followed by findings and fixes

## Integration

### With `paper/paper.tex`
`paper/paper.tex` tracks experiment results, the project narrative, and key progress. The knowledge base tracks literature and domain knowledge. They complement each other:
- `paper.tex` sections can reference wiki pages: "Used [[method_name]] approach, see KB for details"
- Wiki synthesis pages can reference `paper.tex`: "Our experiments (see `paper/paper.tex`) confirm/contradict this finding"

### With `paper/`
Wiki synthesis pages serve as drafting grounds for paper sections. The wiki is working memory; the paper is polished output. When writing paper sections, draw from `wiki/synthesis/` for narrative and `wiki/papers/` for citations.
