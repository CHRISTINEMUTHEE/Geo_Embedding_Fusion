#!/bin/bash
# Shared Unity setup sourced by slurm/*.slurm (after set -euo pipefail).
#
# uv lives in module uv/latest (plain `module load uv` is a no-op here).
# Builds must not use ~/.cache/uv on shared /home -- rmtree fails with
# "Directory not empty" (see uv#12036). Node-local /tmp is the workaround.
# The full dataset is ~110 GB train; home quota is 100 GB, so data/ is a
# symlink to $EMB2HEIGHTS_WORK/data on /work.

module load uv/latest 2>/dev/null || true
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found after 'module load uv/latest' and is not on PATH." >&2
  exit 1
fi

# Force node-local temp. Do not inherit TMPDIR from the submitting shell
# (login nodes / IDE sandboxes often set a path that does not exist on compute).
export UV_CACHE_DIR="/tmp/${USER}/uv-cache"
export TMPDIR="/tmp/${USER}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

WORK_ROOT="${EMB2HEIGHTS_WORK:-/work/pi_jtaneja_umass_edu/${USER}/embed2heights}"
mkdir -p "$WORK_ROOT/data"

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

## Point repo data/ at /work so configs can use data_root: "data"
if [ -L data ] || [ ! -e data ]; then
  ln -sfn "$WORK_ROOT/data" data
elif [ -d data ]; then
  echo "data/ is a real directory; not replacing with $WORK_ROOT/data" >&2
fi

uv sync --group dev
