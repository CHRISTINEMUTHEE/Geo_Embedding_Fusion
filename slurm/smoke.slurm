#!/bin/bash
# Pipeline smoke: 1-epoch LightUNet + EfficientEncoderDecoder, then evaluate_sources.
# Uses --experiment_name smoke_* so it does not overwrite real outputs/<config>/.
#
# Submit AFTER data/manifest.csv has *_frac and data/band_stats.json exists:
#   sbatch --dependency=afterok:<band-stats-jobid> slurm/smoke.slurm
#
#SBATCH --job-name=emb2heights-smoke
#SBATCH --partition=gpu
#SBATCH --constraint=sm_75
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

source slurm/env.sh

export SMOKE_DIR="${TMPDIR}/emb2heights-smoke-configs"
mkdir -p "$SMOKE_DIR" outputs

uv run python - <<'PY'
import yaml
from pathlib import Path
import os
out = Path(os.environ["SMOKE_DIR"])
pairs = [
    ("configs/0_baselines/09_alphaearth_datamodule.yaml", "smoke_alphaearth"),
    ("configs/0_baselines/11_thor_s1_datamodule.yaml", "smoke_thor_s1"),
]
for src, name in pairs:
    cfg = yaml.safe_load(Path(src).read_text())
    cfg["experiment_name"] = name
    Path(out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out / f'{name}.yaml'}")
PY

for name in smoke_alphaearth smoke_thor_s1; do
  uv run python scripts/train.py \
    --config "${SMOKE_DIR}/${name}.yaml" \
    --epochs 1 --max_train_tiles 8 --num_workers 0 --batch_size 2
done

uv run python scripts/evaluate_sources.py \
  --configs "${SMOKE_DIR}/smoke_alphaearth.yaml" "${SMOKE_DIR}/smoke_thor_s1.yaml" \
  --out outputs/smoke_evaluation_table.csv

echo "smoke evaluation table:"
cat outputs/smoke_evaluation_table.csv
