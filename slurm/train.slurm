#!/bin/bash
# Slurm submission script for scripts/train.py on Unity (unity.rc.umass.edu).
#
# Usage:
#   sbatch slurm/train.slurm configs/0_baselines/03_alphaearth_subset_datamodule.yaml
#   sbatch slurm/train.slurm configs/0_baselines/03_alphaearth_subset_datamodule.yaml --epochs 1 --num_workers 0
#
# First arg is the config path; anything after it is passed through as train.py overrides.
#
#SBATCH --job-name=emb2heights
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

CONFIG="$1"
shift || true

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

module load uv 2>/dev/null || true  # falls back to a uv already on PATH
uv sync --group dev

# num_workers=4 (the config default) is a known DataLoader hang on macOS dev machines
# (see scripts/README.md); unconfirmed on Unity's Linux nodes -- smoke-test a new node
# type with --epochs 1 before trusting the default on a long run.
uv run python scripts/train.py --config "$CONFIG" "$@"
