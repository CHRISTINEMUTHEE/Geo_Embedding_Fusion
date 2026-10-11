#!/bin/bash
# Download the region-balanced <1% subset onto /work (data/subset/).
#
# Usage:
#   sbatch slurm/acquire_subset.slurm
#   sbatch slurm/acquire_subset.slurm --limit 6
#
# Default is all six sources (~6 GB). Extra args are forwarded to
# scripts/acquire_subset.py. Resumable: rerun the same sbatch if preempted.
#
#SBATCH --job-name=emb2heights-acquire-subset
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source slurm/env.sh

# env.sh already symlinks ./data -> $WORK_ROOT/data, so --out data/subset
# lands on /work (not /home).
uv run python scripts/acquire_subset.py \
  --out data/subset \
  --sources alphaearth tessera thor_s1 thor_s2 terramind_s1 terramind_s2 \
  --max-gb 6 \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  "$@"
