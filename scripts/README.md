# Training and Evaluation Scripts

This directory contains scripts for training, evaluating, and analyzing models. These scripts form the core of the Flow4ML workflow and support the iterative machine learning development process.

## Main Scripts

- `train.py`: Train models using configuration files (see `emb2heights/trainers.py`)
- `evaluate.py`: Evaluate trained models on test data (template, not yet adapted)
- `infer.py`: Run inference (template, not yet adapted)
- `acquire.py`: Data acquisition (template, not yet adapted)

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