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

## Contents

- `example_artifact.py` — template example
- `verify_datamodule.py` — end-to-end check of the region-grouped data pipeline for any
  embedding source (`--source`, default `alphaearth`): prints the manifest QC summary, the
  train/val region split (asserting the region sets are disjoint), and per-batch shapes/
  ranges/NaN counts; writes a train-vs-val sample grid. Patch-token sources (see
  `emb2heights.datamodule.PATCH_SOURCES`) get correctly different image/target shapes.

## Example

```bash
python artifacts/example_artifact.py
# Produces: artifacts/output/example.png

python artifacts/verify_datamodule.py --source tessera
# Produces: artifacts/output/samples_tessera.png  (requires data/subset/, see scripts/acquire_subset.py)
```

## Guidelines

1. Scripts should run in <30 seconds (use `--light` if needed)
2. Output to `artifacts/output/` directory
3. Name outputs clearly to indicate what they verify

