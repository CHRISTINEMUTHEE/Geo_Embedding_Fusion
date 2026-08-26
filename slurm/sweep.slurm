#!/bin/bash
# Slurm submission script for scripts/label_efficiency_sweep.py on Unity.
#
# Unlike train.slurm, all args are forwarded as-is to label_efficiency_sweep.py
# (it already takes --config as a named flag, not a bare positional), so pass its
# full CLI through directly:
#
#   sbatch slurm/sweep.slurm --config configs/0_baselines/09_alphaearth_datamodule.yaml \
#       --tile-counts 10 20 40 80 --epochs 20
#
# One line is safer to copy-paste (see scripts/README.md on backslash-continuation
# breaking silently) -- the two-line form above is only for readability here.
#
#SBATCH --job-name=emb2heights-sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source slurm/env.sh

# A sweep runs one train() call per tile budget in the same process/job -- more
# DataLoader worker spawn/teardown cycles than a single training run, which raises
# the odds of the known macOS num_workers>0 hang (unconfirmed on Unity's Linux
# nodes). Pass --num-workers 0 here if a budget in the sweep seems to hang.
uv run python scripts/label_efficiency_sweep.py "$@"
