---
name: kb-ingest
description: Ingest a source into the knowledge base. Use when the user adds a new paper, article, or note to knowledge_base/sources/ and wants it processed into the wiki.
---

# KB Ingest

Process a new source document into the knowledge base wiki.

Follow the **Ingest** workflow in `knowledge_base/SCHEMA.md`. That file is the single source of truth for page templates, naming conventions, and workflow steps.

## Source reading hints

- PDF: Use the PDF reader. Read abstract, intro, method, results, conclusion first; appendices only if needed.
- Image: Use multimodal reading to extract information.
- Markdown/text: Read directly.
- For long documents (>20 pages), process in sections rather than all at once.

## Guidelines

- Discuss key takeaways with the user before or during processing
- A single source typically touches 5-15 wiki pages
- When in doubt about categorization, create the page and note the uncertainty
- Check for contradictions with existing wiki content and flag them explicitly
