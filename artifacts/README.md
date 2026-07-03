# Artifacts

An **artifact** is the smallest deliverable that verifies code achieves its goal.

## Purpose

- Replace traditional tests with tangible outputs
- Provide quick visual/manual verification
- Enable fast prototyping feedback loops

## Usage

Each script in this directory produces an artifact:
- Image file, CSV, JSON, plot, etc.
- Something you can open and inspect immediately

## Example

```bash
python artifacts/example_artifact.py
# Produces: artifacts/output/example.png
```

## Guidelines

1. Scripts should run in <30 seconds (use `--light` if needed)
2. Output to `artifacts/output/` directory
3. Name outputs clearly to indicate what they verify
