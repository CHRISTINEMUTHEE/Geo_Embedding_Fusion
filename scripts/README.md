# Training and Evaluation Scripts

This directory contains scripts for training, evaluating, and analyzing models. These scripts form the core of the Flow4ML workflow and support the iterative machine learning development process.

## Main Scripts

- `train.py`: Train models using configuration files (see `emb2heights/trainers.py`)
- `label_efficiency_sweep.py`: Train the same config at several training-tile budgets,
  compare best achieved height accuracy across them (RQ2)
- `evaluate_sources.py`: Evaluate existing checkpoints (no training) across embedding
  sources into one comparison table — the evaluation script for this pipeline
- `compute_band_stats.py`: Per-source, per-channel mean/std over the training split
  only, written to `<data_root>/band_stats.json` for embedding standardization
- `report_class_distribution.py`: Per-source, per-split building/vegetation/water
  coverage — checks the stratified split actually preserved rare classes in val
- `infer.py`: Run inference (template, not yet adapted)
- `acquire.py`: Download the **full** training split from the public HF mirror (~110 GB;
  no login). Writes `data/manifest.csv` for the DataModule. On Unity use
  `sbatch slurm/acquire.slurm` so files land on `/work` (home quota is 100 GB).
- `acquire_subset.py`: Download a **<1% region-balanced subset** (~1.4 GB) from the public
  Hugging Face mirror. No login required. This is the one to use for local development.

## Acquiring Data

Full training split (all six sources + labels, ~110 GB) via the HF mirror:

```bash
sbatch slurm/acquire.slurm          # Unity: writes to /work, then data/manifest.csv
python scripts/acquire.py --check   # verify what's on disk, rewrite the manifest
```

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
point at the resulting `data/subset/`. Full-split counterparts (`09`–`14`) point at `data/`
after `scripts/acquire.py`. Pixel-aligned sources use LightUNet (AlphaEarth, Tessera);
patch-token sources use EfficientEncoderDecoder (THOR/TerraMind S1/S2).

## Training Models

The `train.py` script trains a model from a YAML experiment config:

```bash
# Full-data DataModule baseline (needs scripts/acquire.py first)
python scripts/train.py --config configs/0_baselines/09_alphaearth_datamodule.yaml

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

`evaluate_sources.py` loads each config's own `best_model.pth` (no training) and runs
`evaluate_metrics()` (pooled IoU + height MAE/RMSE, see `emb2heights/README.md`) on
that config's own validation set, writing one row per source to a CSV. Configs with
no checkpoint yet are skipped, not errored on. Only the validation set is used — the
HF catalog's test split (`data/test/*_test_*_emb/`) is embeddings-only with no label
assets, so there's nothing to locally score a test set against; the challenge scores
test submissions itself.

The table also gets one shared `baseline` row (`true_mean_height`/`true_median_height`
over nonzero-height validation pixels, no model involved — the same for every source
here since they share one `data_root`'s split) plus, per source, that source's own
`pred_mean_height`/`pred_median_height` at the identical pixels (`trainers.height_distribution`).
This catches systematic bias RMSE alone can hide — e.g. an undertrained checkpoint with
a middling RMSE can still be predicting well below the true average height everywhere.

```bash
python scripts/evaluate_sources.py --configs configs/0_baselines/*.yaml
# Produces: outputs/evaluation_table.csv
# On Unity: sbatch slurm/eval.slurm
```

(The old `evaluate.py` Lightning-era template — `TrainerConfig`, `datamodules.get_datamodule`,
`trainers.get_task`, none of which exist anymore — has been removed; this replaces it.)

## Band Standardization

Raw embedding scales vary wildly across sources (verified on `data/subset`: AlphaEarth's
dequantized mean range is [-0.43, 0.40], THOR-S2's is [-12743, 13900]) — without
standardizing, RQ1's fusion-vs-best-single-source comparison partly just measures which
source's raw scale suits the optimizer. `compute_band_stats.py` computes per-channel
mean/std over the training split only (same `split_regions()` the DataModule uses, so
val never leaks in) and writes `<data_root>/band_stats.json`:

```bash
python scripts/compute_band_stats.py --data-root data/subset
# Produces: data/subset/band_stats.json
```

`Embed2HeightsDataModule` loads this automatically (`standardize_bands=True`, the
config default) and applies `(x - mean) / std` per channel before nodata zeroing. If
the json doesn't exist yet, training proceeds on raw values with a printed warning
rather than failing — run this once per `data_root` before training.

## Class Distribution & Stratified Splitting

Building and water are rare at both the pixel level (~1-3% mean coverage) and the tile
level (present in under a third of tiles in `data/subset`) — see
`emb2heights/README.md` on how `split_regions()` and the training sampler handle this.
`report_class_distribution.py` prints/writes each split's actual mean coverage and
%-tiles-present per class, using the exact split each config's `train()` call would use:

```bash
python scripts/report_class_distribution.py --configs configs/0_baselines/0{3,4,5,6,7,8}_*.yaml
# Produces: outputs/class_distribution.csv
```

Today every subset config shares one `data/subset/manifest.csv`, so this reports the
same distribution for all 6 sources — it will only diverge if a source's tile
completeness (`keep`) ever differs from the others.

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