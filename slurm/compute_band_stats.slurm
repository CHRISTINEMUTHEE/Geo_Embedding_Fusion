#!/bin/bash
# Per-source train-split channel mean/std -> <data_root>/band_stats.json.
#
# Must run AFTER the manifest for that data_root has *_frac columns
# (acquire.py --check or acquire_subset.py). --val-frac and --seed must match
# the YAML configs that will train on this data_root (03-14: 0.3 / 42).
#
# Usage:
#   sbatch slurm/compute_band_stats.slurm
#   sbatch slurm/compute_band_stats.slurm --data-root data/subset
#   sbatch slurm/compute_band_stats.slurm --data-root data --sources alphaearth
#
# With no args, defaults to the full split. cpu-preempt is resumable: sources
# already in band_stats.json are skipped (delete the json to recompute).
#
#SBATCH --job-name=emb2heights-band-stats
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source slurm/env.sh

if [ "$#" -eq 0 ]; then
  set -- --data-root data --val-frac 0.3 --seed 42
fi

uv run python scripts/compute_band_stats.py "$@"
