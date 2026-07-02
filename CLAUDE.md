# Agent Rules

## 1. Core Principles

On every task:
1. **Think independently** about the best solution; consider missing information that would improve it.
2. **Pause and suggest** if your solution differs from the user's. If the approach is wrong or suboptimal, say so directly and suggest the better path.

- Be concise. Minimal output. No fluff, no filler, no unnecessary explanation.
- Give the simplest working solution first. Iterate to add complexity if needed.
- Prioritize correctness and objectivity.
- Read and understand existing code before making changes.
- If a task has multiple valid approaches, pick the simplest one. Briefly note alternatives only if the tradeoff matters.
- Keep explanations short and plain. No markdown formatting, headers, or bullet lists in chat responses unless specifically requested.
- When reporting errors, state the cause and the fix. Skip the preamble.

## 1.1 Honest Reporting
- Do not oversell work. No checkmarks, "all complete," or summary lists that imply thoroughness when parts were skipped.
- Do not downplay or omit problems. If something was skipped, say so directly — do not bury it in a "limitations" or "future work" note phrased to sound like a deliberate scoping choice.
- Do not stop early and claim completion. Do not invent excuses to exit ("time constraints," "budget limits," "left for future work," "documented limitation") when the user gave no such constraint.
- Do not silently relax constraints. If the task seems too hard or unclear, say so and ask — do not quietly solve an easier version and present it as the requested result.
- No fake polish. Code that lints/compiles but is wrong, tests that pass trivially, narrow unit tests that dodge real end-to-end failure, and mocked behavior presented as real are all prohibited.
- Investigate every test failure, including timeouts. Do not dismiss them.
- Proactively flag flaws, cheating, or reward-hacking in your own or prior work. Surface issues saliently — do not wait to be asked.
- Delete broken, irrelevant, or failed artifacts. Do not fabricate post-hoc justifications (e.g. relabeling as a "control" or "baseline") to keep them.
- Do not assert confidence you do not have. If unsure whether your work is correct, say so. Do not blame external impossibility when the task is actually tractable.
- When reporting completion, explicitly list: what was done, what was skipped or deferred and why, what remains uncertain, and any end-to-end verification performed. A report without these is not a completion report.

## 2. Code Style
- Python unless otherwise specified.
- No docs. Only essential comments (explain "why", not "what").
- Minimal code, essential checks only.
- Must be quickly reviewable.
- Mark critical functionalities with `# REVIEW REQUIRED` for final human approval before deployment. Critical = a bug here compromises the system's purpose (e.g. core training/inference logic, data integrity, evaluation metrics, anything users will rely on for decisions).
- Never write test files.
- Favor clarity over cleverness.
- Always check if your implemented code could be run faster.

## 2.1 Code Comprehension Comments

Use `##` comments (double-hash) to make generated code easy to navigate and understand.

### When to use `##`

- **Pointers**: Mark important code blocks so the reader can scan quickly.
  ```python
  ## <- [GOAL] Build the user profile from raw API response
  def build_profile(data):
      ...
  ```
- **Explanations**: For non-obvious logic, add a `##` comment that explains *what* and *why* in plain language with a simple example.
  ```python
  ## Flatten nested dict: {"a": {"b": 1}} -> {"a.b": 1}
  def flatten(d, parent_key=""):
      ...
  ```

### When NOT to use `##`

- Self-explanatory code (`x = x + 1`) needs no comment.
- Standard `#` comments are still used for TODOs, disabled code, and general notes.

### Rules

1. `##` is reserved for reader-facing guidance — never for disabled code or TODOs.
2. Keep `##` comments short (one line when possible).
3. Prefer a simple example over a long explanation.

## 3. Artifact-based development
- Default to scripts with text/numerical outputs for verification.
- If user requests a notebook: write cells that output both text (for agent verification) and Figures (for user verification).
- In "plan" mode, suggest the most reliable artifact(s) for code verification.

## 4. Prototyping Speed
- All commands: 30-second timeout.
- If >30s: add `--light` flag with faster approximation.

## 5. Experiment YAML Files
- When creating experiment YAML files, add concise documentation at the top as comments.
- Explain what the experiment does and why it exists.

## 7. README Maintenance
- Every sub-directory has a `README.md` describing its purpose, contents, and usage.
- After making changes to a directory (adding, removing, or modifying files), update that directory's `README.md` to reflect the current state.
- When creating a new sub-directory, always create a `README.md` in it.
- Keep READMEs concise: purpose, contents list, and usage guidelines.
- The root `README.md` holds the canonical **Repository Map**. Whenever a change adds, removes, moves, or renames a file or directory, check that map and update it in the same change so it stays an accurate guide for navigation.

## 13. Knowledge Base

The `knowledge_base/` directory is a persistent, structured research context store following the LLM Wiki pattern. Read `knowledge_base/SCHEMA.md` for full conventions.

Key rules:
- **Never modify files under `knowledge_base/sources/`.** Those are immutable raw inputs.
- All files under `knowledge_base/wiki/` are agent-owned. Maintain them using the templates in `SCHEMA.md`.
- After every KB operation (ingest, query-with-save, lint), update `knowledge_base/index.md` and append to `knowledge_base/log.md`.
- Use `[[page_name]]` cross-references between wiki pages. Keep them bidirectional.
- When `paper/paper.tex` references methods or papers, cross-link to the relevant wiki page.

## 14. Design & Visualization

For paper figures and tables under `paper/`, use LaTeX/TikZ via the `generate-tikz-pdf` skill.

## 15. Research Paper Writing

When writing or editing any part of a research paper — abstract, introduction, methods, results, conclusion, or appendix — use the `write-research-paper` skill. Apply it for both drafting from scratch and editing existing content.

Run the `stop-slop` skill alongside it for any substantial prose pass.

---

# Project Structure

Quick orientation below. The canonical **Repository Map** lives in the root `README.md` —
consult it to navigate, and update it whenever you change the structure (see §7).

```
configs/           # YAML experiment configs (organized by research direction)
scripts/           # Exported production scripts from verified notebooks
notebooks/         # Primary development location for prototyping and verification
knowledge_base/    # Persistent research context (LLM wiki pattern)
emb2heights/  # Reusable modules: models, datasets, trainers, config
```

## Environment

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management. All dependencies are declared in `pyproject.toml` (runtime under `[project] dependencies`, dev tooling under `[dependency-groups] dev`, release tooling under `[dependency-groups] release`).

Create the environment:

```bash
uv sync
```

Run commands without activating:

```bash
uv run python scripts/train.py ...
```

Or activate the venv once and use plain commands:

```bash
source .venv/bin/activate
python scripts/train.py ...
```

When you need a new library:
1. `uv add <package>` (runtime dep) or `uv add --group dev <package>` (dev tool).
2. Commit the updated `pyproject.toml` and `uv.lock`.
