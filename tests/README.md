# Tests

Unit tests for the `emb2heights` package. All tests run on small synthetic
GeoTIFFs generated in temp dirs (fixtures in `conftest.py`) — no real data needed.

## Contents

- `conftest.py`: shared fixtures — synthetic (embedding, label) tif pairs and a ready `ExperimentConfig`
- `test_config.py`: config defaults, derived paths, YAML loading + CLI overrides, validation
- `test_datasets.py`: filename → core-ID normalization, file pairing, dataset shapes,
  height normalization, padding, train/val split
- `test_models.py`: model factory, forward-pass shapes at several channel counts, gradient flow
- `test_losses.py`: loss factory
- `test_trainers.py`: hand-computed metric values (IoU, masked RMSE), optimizer/scheduler
  builders, train/eval epoch behavior, end-to-end `train()` artifact check

## Usage

```bash
python -m pytest tests/ -q            # full suite (~15 s)
python -m pytest tests/test_datasets.py -q   # one module
python -m pytest tests/ -k "height" -q       # by keyword
```

When something breaks during development, run the test module matching the module
you changed; the failing test name states the violated behavior.
