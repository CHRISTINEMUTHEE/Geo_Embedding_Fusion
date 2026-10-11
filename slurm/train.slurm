#!/bin/bash
# Slurm submission script for scripts/train.py on Unity (unity.rc.umass.edu).
#
# Usage:
#   sbatch slurm/train.slurm configs/0_baselines/09_alphaearth_datamodule.yaml
#   sbatch slurm/train.slurm configs/0_baselines/09_alphaearth_datamodule.yaml --epochs 1 --num_workers 0
#
# First arg is the config path; anything after it is passed through as train.py overrides.
# Job name stays emb2heights so logs are slurm/logs/emb2heights_<jobid>.{out,err}.
#
# GPU: torch 2.13+cu130 only has kernels for sm_75+ (Turing and newer). The gpu
# partition also contains gypsum titan_x / m40 / 1080_ti (sm_52/sm_61) and v100
# (sm_70); those fail at the first conv2d. --constraint=sm_75 keeps 2080_ti,
# rtx_8000, a40, a100, l40s, l4, h100, a4000, a16 (confirmed via sinfo -o "%G %f").
#
#SBATCH --job-name=emb2heights
#SBATCH --partition=gpu
#SBATCH --constraint=sm_75
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch slurm/train.slurm <config.yaml> [train.py overrides...]}"
shift || true

source slurm/env.sh

# num_workers=4 (the config default) is a known DataLoader hang on macOS dev machines
# (see scripts/README.md); unconfirmed on Unity's Linux nodes -- smoke-test a new node
# type with --epochs 1 before trusting the default on a long run.
uv run python scripts/train.py --config "$CONFIG" "$@"
