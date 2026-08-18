# Training and Evaluation Scripts

This directory contains scripts for training, evaluating, and analyzing models. These scripts form the core of the Flow4ML workflow and support the iterative machine learning development process.

## Main Scripts

- `train.py`: Train models using configuration files (see `emb2heights/trainers.py`)
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
fetches what's missing. See `configs/0_baselines/03_alphaearth_subset_datamodule.yaml` for a
config pointed at the resulting `data/subset/`.

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

Hyperparameter search (`--search_mode`, Optuna) was removed with the old template
`train.py`; re-add it when the baseline pipeline is stable.

## Evaluating Models

The `evaluate.py` script runs inference and computes metrics:

```bash
python evaluate.py --model_path model_runs/experiment/best.ckpt --test_data path/to/test/data
```

### Key Evaluation Options

- `--model_path`: Path to model checkpoint (required)
- `--test_data`: Path to test data (required)
- `--config`: Path to original config file (optional)
- `--output_dir`: Directory to save results (default: "evaluation_results")
- `--batch_size`: Batch size for evaluation
- `--save_predictions`: Save model predictions to disk
- `--task_type`: Task type (base, segmentation, classification, regression)
- `--gpu_id`: GPU ID to use for evaluation

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