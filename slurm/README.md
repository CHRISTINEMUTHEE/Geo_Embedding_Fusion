# Slurm

Job submission scripts for running this project on Unity (unity.rc.umass.edu).

## Contents

- `env.sh`: shared setup sourced by the other scripts (`module load uv/latest`,
  node-local `UV_CACHE_DIR`/`TMPDIR` under `/tmp/$USER`, `uv sync`, symlink
  `data/` -> `/work/pi_jtaneja_umass_edu/$USER/embed2heights/data`).
- `acquire.slurm`: `cpu-preempt` job that runs `scripts/acquire.py` (full training
  split onto `/work`; extra args forwarded). Preempt is resumable — rerun sbatch.
- `train.slurm`: single-GPU `scripts/train.py` on `--partition=gpu` with
  `--constraint=sm_75` (torch 2.13+cu130 cannot run on gypsum TITAN X / M40 /
  1080 Ti or V100). First positional arg is the config path; extra args are
  forwarded as CLI overrides. Job name stays `emb2heights` (logs:
  `emb2heights_<jobid>.out`).
- `eval.slurm`: single-GPU `scripts/evaluate_sources.py` (same `gpu` /
  `sm_75` constraint as `train.slurm`). Default is `--configs
  configs/0_baselines/*.yaml`; extra args are forwarded. Configs without a
  checkpoint are skipped. Job name `emb2heights-eval` (logs:
  `emb2heights-eval_<jobid>.out`).
- `sweep.slurm`: single-GPU `scripts/label_efficiency_sweep.py`. Forwards its full
  CLI as-is (`--config`, `--tile-counts`, ...).
- `logs/`: created on first submission (`%x_%j.out`/`.err` per job), gitignored.

## Usage

On the Unity login node:

```bash
cd ~/work_experiments/Geo_Embedding_Fusion
sbatch slurm/acquire.slurm
sbatch slurm/train.slurm configs/0_baselines/09_alphaearth_datamodule.yaml
sbatch slurm/eval.slurm
sbatch slurm/sweep.slurm --config configs/0_baselines/09_alphaearth_datamodule.yaml --tile-counts 10 20 40 80 --epochs 20
```

Wait for the acquire job to finish (and `data/manifest.csv` to exist) before
submitting train/sweep. Check `squeue --me`; output lands in `slurm/logs/`.

## Notes

- `--partition=gpu --constraint=sm_75 --gpus=1` and the `--mem`/`--time`/
  `--cpus-per-task` values in train/sweep are starting points — adjust once you
  know actual GPU memory and wall-clock (`seff <jobid>`). `sm_75` is required
  for the current torch wheel; do not drop it or you can land on a TITAN X again.
  `train.slurm` defaults `--time=12:00:00` for a full-data run; `eval.slurm`
  defaults `--time=04:00:00`; `sweep.slurm` defaults `--time=08:00:00` and
  still has no GPU constraint.
- `module load uv/latest` is required on this cluster (`module load uv` is a
  no-op). If that module is missing, install uv once
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- uv's cache must not live on `/home`: source builds fail with `OSError: Directory
  not empty` there. `env.sh` sets `UV_CACHE_DIR`/`TMPDIR` to node-local `/tmp`.
- The known `num_workers>0` DataLoader hang (see `scripts/README.md`) has only been
  observed on macOS. Smoke-test a new node type with `--epochs 1` before trusting
  `num_workers=4` on a long unattended run.
