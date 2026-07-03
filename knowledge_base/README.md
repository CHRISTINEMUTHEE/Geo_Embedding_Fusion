# Knowledge Base

Persistent, structured research context store following the LLM Wiki pattern. The agent incrementally builds and maintains this wiki as sources are added and questions are asked.

## Architecture

```
knowledge_base/
├── SCHEMA.md          # Agent conventions, templates, workflows
├── index.md           # Content catalog (agent's navigation entry point)
├── log.md             # Chronological operation log
├── sources/           # Immutable raw inputs (papers, articles, notes, images)
│   ├── papers/        # Research papers
│   ├── articles/      # Blog posts, tutorials, reports
│   ├── notes/         # Personal and meeting notes
│   └── assets/        # Images, diagrams, data files
└── wiki/              # Agent-generated and maintained pages
    ├── overview.md    # Research landscape & project positioning
    ├── papers/        # One summary per ingested paper
    ├── concepts/      # Domain concepts & definitions
    ├── methods/       # Techniques & algorithms
    ├── datasets/      # Dataset descriptions
    └── synthesis/     # Evolving thesis, comparisons, gap analyses
```

## Usage

### Adding Sources

Drop files into the appropriate `sources/` subdirectory:
- PDFs and paper files go in `sources/papers/`
- Web articles (clipped to markdown) go in `sources/articles/`
- Your own notes go in `sources/notes/`
- Images and diagrams go in `sources/assets/`

Then invoke `/kb-ingest` to process them into the wiki.

### Querying

Ask research questions with `/kb-query`. The agent reads the wiki index, finds relevant pages, and synthesizes an answer. Substantial answers get filed back as synthesis pages.

### Maintenance

Run `/kb-lint` periodically to check for broken cross-references, orphan pages, stale claims, and missing connections. The agent fixes what it can and reports what needs your judgment.

## Conventions

- Sources are **immutable** — the agent never modifies them
- Wiki pages are **agent-owned** — the agent creates, updates, and maintains them
- See `SCHEMA.md` for page templates, naming conventions, and detailed workflows
- Large binary files in `sources/` are git-ignored (PDFs, images). Keep them local or use external storage for sharing.
