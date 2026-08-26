#!/bin/bash
# Slurm submission script for scripts/evaluate_sources.py on Unity.
#
# Evaluates every config that already has outputs/<experiment>/best_model.pth.
# Configs without a checkpoint are skipped (see evaluate_sources.py).
#
# Usage:
#   sbatch slurm/eval.slurm
#   sbatch slurm/eval.slurm --configs configs/0_baselines/*.yaml
#   sbatch slurm/eval.slurm --configs configs/0_baselines/09_alphaearth_datamodule.yaml \
#       --out outputs/eval_alphaearth.csv
#
# With no args, defaults to --configs configs/0_baselines/*.yaml (the glob is
# passed quoted so Python expands it on the compute node). Extra args are
# forwarded as-is. Job name stays emb2heights-eval so logs are
# slurm/logs/emb2heights-eval_<jobid>.{out,err}.
#
# Same sm_75 GPU constraint as train.slurm -- eval still runs conv2d.
#
#SBATCH --job-name=emb2heights-eval
#SBATCH --partition=gpu
#SBATCH --constraint=sm_75
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source slurm/env.sh

if [ "$#" -eq 0 ]; then
  set -- --configs "configs/0_baselines/*.yaml"
fi

uv run python scripts/evaluate_sources.py "$@"
