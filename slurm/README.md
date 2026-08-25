# Slurm

Job submission scripts for running this project on Unity (unity.rc.umass.edu).

## Contents

- `train.slurm`: submits `scripts/train.py` as a single-GPU job. First positional arg
  is the config path; anything after it is forwarded as CLI overrides to `train.py`.
- `sweep.slurm`: submits `scripts/label_efficiency_sweep.py` as a single-GPU job
  running the whole sweep (all tile budgets, sequential `train()` calls) inside one
  job. Forwards its full CLI as-is (`--config`, `--tile-counts`, ... are already named
  flags on the underlying script, unlike `train.slurm`'s positional config arg).
- `logs/`: created on first submission (`%x_%j.out`/`.err` per job), gitignored.

## Usage

Code and config live in this repo; data does not need to be copied by hand. On the
Unity login node:

```bash
git clone <this-repo-url> ~/emb2heights   # or git pull if already cloned
cd ~/emb2heights
uv run python scripts/acquire_subset.py   # pulls the <1% HF subset directly on-cluster
sbatch slurm/train.slurm configs/0_baselines/03_alphaearth_subset_datamodule.yaml
sbatch slurm/sweep.slurm --config configs/0_baselines/03_alphaearth_subset_datamodule.yaml --tile-counts 10 20 40 80 --epochs 20
```

Check job status with `squeue -u $USER`; output lands in `slurm/logs/`.

## Notes

- `--partition=gpu --gpus=1` and the `--mem`/`--time`/`--cpus-per-task` values in both
  scripts are starting points, not measured for this workload — adjust once you know
  actual GPU memory and wall-clock usage for a given config (check with `seff
  <jobid>` after a run). `sweep.slurm` defaults `--time=08:00:00` since it runs every
  tile budget as one job; scale that with tile-count list length x epochs.
- `module load uv` is attempted first; if Unity doesn't provide a `uv` module, install
  it once per account (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and the
  script falls back to whatever `uv` is on `PATH`.
- The known `num_workers>0` DataLoader hang (see `scripts/README.md`) has only been
  observed on macOS; it's unconfirmed whether it reproduces on Unity's Linux nodes.
  Smoke-test a new node type with `--epochs 1` before trusting `num_workers=4` on a
  long unattended run.
