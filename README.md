# emb2heights

A multi-task model that takes pre-computed embeddings from several geospatial foundation models and predicts sub-pixel land cover and height over France.


* Free software: MIT license


## Challenge Resources

* **Baseline repository** — <https://github.com/VMarsocci/emb2heights-baselines>
  Rules for running a baseline experiment and how to package results for submission.
* **Dataset (EOTDL)** — <https://www.eotdl.com/datasets/embed2heights>
  Data description and a CLI for downloading the embeddings.
* **Project information** — <https://docs.google.com/document/d/1QF9tJyGQXLREnGYqgsdlO959FqY3pndtmdwSrh44saw/edit>
  Full challenge brief (goals, dataset, evaluation, submission).


## Guidelines

What follows is a structured approach for developing your emb2heights project by organizing experiments, iterating quickly, and maintaining effective feedback loops.

## Step-by-Step Process

### 1. Set Up Experiment Tracking

* Create a [Google Sheet](https://docs.google.com/spreadsheets) with columns: Experiment ID, Description/Hypothesis, Key Parameters, Results/Metrics, Observations, Next Steps
* This sheet will be your central record for all experiments

### 2. Establish Baselines

* Implement simple baseline methods (random predictions, mode, median, mean, persistence)
* Document baseline results in your tracking sheet
* This gives you a reference point for improvement

### 3. Prepare Your Environment

This project uses [uv](https://docs.astral.sh/uv/). If a conda env is active, `conda deactivate` first.

```console
$ uv sync --group dev
```

That creates `.venv` (Python 3.12) with all project packages. Run commands with `uv run ...`, or `source .venv/bin/activate`. Add a library with `uv add <package>` (or `uv add --group dev <package>` for test/lint tools), then commit `pyproject.toml` and `uv.lock`.

### 4. Organize Your Data

* Prepare dataset files or URLs using one of these approaches:
  
  - CSV file with pointers to input/output data
  - Local file paths organized in directories
  - Remote data URLs with access tokens

### 5. Run Your First Experiment

* Train a baseline model:

```console
$ uv run python scripts/train.py --config configs/0_baselines/09_alphaearth_datamodule.yaml
```

* Evaluate existing checkpoints (see `scripts/README.md`):

```console
$ uv run python scripts/evaluate_sources.py --configs configs/0_baselines/03_alphaearth_subset_datamodule.yaml
```

* Document results in your tracking sheet

### 6. Iterative Improvement Loop

* Identify a specific change to implement based on error analysis
* Create a new configuration file in the appropriate `configs/` subdirectory
* Train the updated model using the new config
* Evaluate and analyze errors
* Document results in tracking sheet
* If performing better, consider producing a release
* Brainstorm ideas to reduce mistakes, prioritize and repeat

### 7. Hyperparameter Tuning

* Categorize hyperparameters:
  
  - Scientific: measure effect on performance
  - Nuisance: must be tuned for fair comparisons
  - Fixed: keep constant for now

* Run hyperparameter search: not implemented yet (the old Optuna `--search_mode`
  was template code and was removed; re-add it in `scripts/train.py` when needed)

### 8. Speeding Up Experimentation

* For training: subsample data, increase batch size, maxout on GPU usage
* For inference: subsample test set
* For evaluation: parallelize and distribute training & evaluation jobs
* For analysis: focus on model collapses for faster error analysis

## Repository Map

This is the canonical map of the repository. It exists so agents (and humans) can
navigate the project quickly and know where new work belongs. **Keep it current:**
whenever you add, remove, move, or rename a file or directory, update this map in the
same change (see `AGENTS.md` §7).

```text
.
├── AGENTS.md           - Agent rules and conventions (CLAUDE.md and
│                         .github/copilot-instructions.md symlink to this)
├── README.md           - This file: project overview, workflow, and repository map
├── pyproject.toml      - Dependencies and project metadata (managed with uv)
├── Makefile            - Common developer commands
├── emb2heights/  - Core package: reusable modules
│   ├── config.py       - ExperimentConfig (pydantic) + YAML loading
│   ├── models.py       - Model architectures and the build_model() factory
│   ├── datasets.py     - File pairing, datasets, and build_dataloaders()
│   ├── datamodule.py   - Region-grouped train/val split + manifest-driven loading
│   ├── losses.py       - Loss functions and the build_loss() factory
│   ├── trainers.py     - Plain PyTorch training loop, metrics, visualization
│   └── cli.py          - Command-line entry points
├── data/               - Local datasets (gitignored)
│   ├── manifest.csv    - Full-split tile index written by scripts/acquire.py
│   ├── band_stats.json - Per-source channel mean/std from compute_band_stats.py
│   └── subset/         - <1% dev subset: inputs/, outputs/, manifest.csv, band_stats.json
├── scripts/            - Runnable scripts (train, evaluate, infer, acquire)
│   ├── acquire.py      - Full training split from the HF mirror (~110 GB)
│   ├── acquire_subset.py - Region-balanced <1% subset from the public HF mirror
│   ├── compute_band_stats.py - Per-source train-split channel mean/std -> band_stats.json
│   └── report_class_distribution.py - Per-source/split building/vegetation/water coverage
├── tests/              - Unit tests (pytest, synthetic data, no real data needed)
├── configs/            - YAML experiment configs, organized by research direction
│   └── 0_baselines/    - Per-source LightUNet / EfficientDecoder YAMLs (full data + subset)
├── slurm/              - Slurm job scripts for running on Unity (unity.rc.umass.edu)
│   ├── env.sh          - Shared uv/cache/data-symlink setup for the jobs below
│   ├── acquire.slurm   - CPU job: scripts/acquire.py onto /work
│   ├── train.slurm     - Single-GPU submission wrapper around scripts/train.py
│   ├── eval.slurm      - Single-GPU wrapper around scripts/evaluate_sources.py
│   └── sweep.slurm     - Single-GPU submission wrapper around label_efficiency_sweep.py
├── artifacts/          - Scripts that produce verifiable text/numerical artifacts
├── knowledge_base/     - Persistent research context (LLM wiki pattern; see SCHEMA.md)
│   ├── sources/        - Immutable raw inputs (never modify)
│   └── wiki/           - Agent-owned wiki pages with [[cross-references]]
└── paper/              - LaTeX paper, figures, tables, and references (git submodule)
```

Every sub-directory also has its own `README.md` describing its purpose, contents, and
usage in more detail.

## Customization

### 1. Define your data

* Update `datasets.py` with your data loading logic
* Configure input and output formats

### 2. Choose/implement models

* Select from standard models or add custom architectures in `models.py`
* Configure via YAML files

### 3. Set evaluation metrics

* Customize metrics in `trainers.py` for your specific task
* Add task-specific visualizations

### 4. Document your process

* Use your tracking sheet to record iterations
* Keep error analysis for each significant improvement

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.