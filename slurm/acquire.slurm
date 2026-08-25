#!/bin/bash
# Download the full embed2heights training split onto /work (~110 GB).
#
# Usage:
#   sbatch slurm/acquire.slurm
#   sbatch slurm/acquire.slurm --dirs data/train/labels data/train/alphaearth_emb
#
# Args are forwarded to scripts/acquire.py. Default is all six sources + labels.
# Resumable: rerun the same sbatch if the job dies mid-download.
#
#SBATCH --job-name=emb2heights-acquire
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source slurm/env.sh

# env.sh already symlinks ./data -> $WORK_ROOT/data. --out is the repo root so
# catalog ids data/train/... land in that symlink (i.e. on /work, not /home).
uv run python scripts/acquire.py --out "$SLURM_SUBMIT_DIR" --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
