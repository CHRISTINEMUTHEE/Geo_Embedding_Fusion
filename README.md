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

* Sync the project environment with [uv](https://docs.astral.sh/uv/):

```console
$ uv sync
```

### 4. Organize Your Data

* Prepare dataset files or URLs using one of these approaches:
  
  - CSV file with pointers to input/output data
  - Local file paths organized in directories
  - Remote data URLs with access tokens

### 5. Run Your First Experiment

* Train a baseline model:

```console
$ uv run python scripts/train.py --config configs/0_baselines/0_simple_baseline.yaml
```

* Evaluate the model:

```console
$ uv run python scripts/evaluate.py --model-path model_runs/experiment_name/best.ckpt --test-data path/to/test
```

* Analyze errors:

```console
$ uv run python scripts/analyze.py --model-path model_runs/experiment_name/best.ckpt --test-data path/to/test
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

* Run hyperparameter search:

```console
$ uv run python scripts/train.py --config configs/0_baselines/0_simple_baseline.yaml --search_mode --n_trials 20
```

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
│   ├── config.py       - Configuration validation and management
│   ├── models.py       - Model architectures and the get_model() factory
│   ├── datasets.py     - Dataset loading and preprocessing
│   ├── datamodules.py  - PyTorch Lightning data modules
│   ├── trainers.py     - Training logic, tasks, and metrics
│   └── cli.py          - Command-line entry points
├── scripts/            - Runnable scripts (train, evaluate, infer, acquire)
├── configs/            - YAML experiment configs, organized by research direction
├── artifacts/          - Scripts that produce verifiable text/numerical artifacts
├── notebooks/          - Jupyter notebooks for prototyping and analysis
├── knowledge_base/     - Persistent research context (LLM wiki pattern; see SCHEMA.md)
│   ├── sources/        - Immutable raw inputs (never modify)
│   └── wiki/           - Agent-owned wiki pages with [[cross-references]]
├── paper/              - LaTeX paper, figures, tables, and references (git submodule)
└── .claude/            - Claude config: skills/, commands/, settings.json
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