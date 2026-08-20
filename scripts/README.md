# Training and Evaluation Scripts

This directory contains scripts for training, evaluating, and analyzing models. These scripts form the core of the Flow4ML workflow and support the iterative machine learning development process.

## Main Scripts

- `train.py`: Train models using configuration files (see `emb2heights/trainers.py`)
- `label_efficiency_sweep.py`: Train the same config at several training-tile budgets,
  compare best achieved height accuracy across them (RQ2)
- `evaluate.py`: Evaluate trained models on test data (template, not yet adapted)
- `infer.py`: Run inference (template, not yet adapted)
- `acquire.py`: Download the **full** dataset from EOTDL (110+ GB; needs an EOTDL login and a
  pre-staged catalog at `~/.cache/eotdl/datasets/embed2heights/catalog.v1.parquet`)
- `acquire_subset.py`: Download a **<1% region-balanced subset** (~1.4 GB) from the public
  Hugging Face mirror. No login required. This is the one to use for local development.

## Acquiring Data

`acquire_subset.py` samples whole regions (the only geographic grouping the dataset exposes),
downloads them, and writes a `manifest.csv` with per-tile QC:

```bash
python scripts/acquire_subset.py --dry-run     # show selection and size, download nothing
python scripts/acquire_subset.py --limit 6     # quick smoke test
python scripts/acquire_subset.py               # alphaearth only -> data/subset/
```

Six sources are available (see `SOURCES` in the script): `alphaearth`, `tessera` (pixel-aligned,
~17/~34 MB per tile) and `thor_s1`, `thor_s2`, `terramind_s1`, `terramind_s2` (patch-token
embeddings, ~1 MB per tile — see `emb2heights.datamodule.PATCH_SOURCES`). A tile is only kept if
it has every requested source, so pulling more sources for the *same* tiles needs a bigger
`--max-gb`, not a smaller subset:

```bash
python scripts/acquire_subset.py --sources alphaearth tessera thor_s1 thor_s2 \
    terramind_s1 terramind_s2 --max-gb 6
```

Already-downloaded files are skipped (resumable), so re-running with more `--sources` only
fetches what's missing. Subset training configs in `configs/0_baselines/` (`03`–`08`) all
point at the resulting `data/subset/`. Pixel-aligned sources use LightUNet (`03` AlphaEarth,
`04` Tessera); patch-token sources use EfficientDecoder (`05`/`06` THOR S1/S2, `07`/`08`
TerraMind S1/S2).

## Training Models

The `train.py` script trains a model from a YAML experiment config:

```bash
# Standard training
python scripts/train.py --config configs/0_baselines/01_alphaearth_lightunet.yaml

# With overrides (quick smoke run)
python scripts/train.py --config configs/0_baselines/01_alphaearth_lightunet.yaml --epochs 1 --batch_size 4
```

### Key Training Options

- `--config`: Path to YAML configuration file (required)
- `--experiment_name`, `--model_name`, `--batch_size`, `--patch_size`, `--epochs`,
  `--learning_rate`, `--weight_decay`, `--num_workers`, `--random_seed`: override
  the corresponding YAML value
- `--loss_name`, `--w_height`, `--w_landcover`, `--bg_weight`: loss overrides (see
  `emb2heights/losses.py`) — `--w_landcover 0.0` is the RQ3 ablation
- `--max_train_tiles`: cap training tiles (val stays full) — one point of a
  label-efficiency sweep; use `label_efficiency_sweep.py` to run the whole sweep

Hyperparameter search (`--search_mode`, Optuna) was removed with the old template
`train.py`; re-add it when the baseline pipeline is stable.

## Label-Efficiency Sweep

`label_efficiency_sweep.py` trains a config once per training-tile budget (val fixed
across all of them) and plots best achieved overall height RMSE against tiles used —
the actual RQ2 curve, not a single run's epoch-by-epoch progress. Each run's
checkpoints/loss-curve/visualizations are **not** saved by default — the sweep calls
`train(config, save_artifacts=False)`, since only the final aggregate CSV/plot is
normally wanted, not N full `outputs/<experiment_name>/` trees:

```bash
python scripts/label_efficiency_sweep.py --config configs/0_baselines/03_alphaearth_subset_datamodule.yaml --tile-counts 10 20 40 80 --epochs 20
# Produces: outputs/<sweep-name>_sweep/label_efficiency_results.csv, label_efficiency_curve.png
```

One line on purpose — a backslash-continued command silently breaks if a trailing
space survives copy-paste; zsh then runs the first line alone and tries to run
`--tile-counts ...` as its own command ("command not found"). If a run in the sweep
hangs (see the known `num_workers` issue in `emb2heights/README.md`), add
`--num-workers 0`. Add `--save-artifacts` to keep each run's full per-experiment
outputs too, e.g. to inspect one budget's model or sample predictions.

## Evaluating Models

`evaluate.py` is a template carried over from the original project scaffold and is not
adapted to this pipeline — it imports modules (`TrainerConfig`, `datamodules.get_datamodule`,
`trainers.get_task`) that no longer exist and will fail on import. Not usable yet.

## Extending Scripts

When developing new functionality:

1. Maintain the command-line interface pattern
2. Add descriptive help text for all parameters
3. Keep core functionality modular
4. Add new parameters with sensible defaults

## Custom Scripts

For project-specific needs, create additional scripts in this directory. Common extensions include:

- Data preprocessing scripts
- Custom inference scripts
- Ensemble generation scripts

Follow the same command-line interface pattern and documentation standards used in the core scripts.