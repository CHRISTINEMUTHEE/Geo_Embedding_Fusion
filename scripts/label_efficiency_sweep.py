#!/usr/bin/env python
"""
Label-efficiency sweep: train the same config at several training-tile budgets and
compare best achieved overall height RMSE across them -- RQ2 ("does fusion change how
many labels are needed to reach target height accuracy?").

Each budget is a SEPARATE training run (best overall height RMSE across that run's
epochs is its score) -- not one run's epoch-by-epoch curve, which is a
training-progress plot, not a label-efficiency one (see
emb2heights.trainers.plot_height_rmse_vs_epoch). Budgets share a nested, deterministic
tile subset (see Embed2HeightsDataModule.max_train_tiles) so budget=10 tiles are a
subset of budget=25's -- a genuine "adding more labels" sweep. Validation tiles are
identical across every run in the sweep.

By default no per-run checkpoints/plots/visualizations are written -- only the final
label_efficiency_results.csv and label_efficiency_curve.png. Pass --save-artifacts to
also keep each run's full outputs/<experiment_name>/ (checkpoints, loss curve, sample
visualizations) if you need to inspect a specific budget's model.

Usage (one line -- safer to copy-paste than a backslash-continued command, which
silently breaks if a trailing space survives the copy and zsh runs the first line
alone, then tries to run "--tile-counts ..." as its own command):
    python scripts/label_efficiency_sweep.py --config configs/0_baselines/03_alphaearth_subset_datamodule.yaml --tile-counts 10 20 40 80 --epochs 20
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emb2heights.config import load_config  # noqa: E402
from emb2heights.trainers import train  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Base YAML config")
    p.add_argument("--tile-counts", type=int, nargs="+", required=True,
                   help="Training-tile budgets to sweep, e.g. 10 20 40 80")
    p.add_argument("--sweep-name", type=str, default=None,
                   help="Prefix for each run's experiment_name (default: base config's experiment_name)")
    p.add_argument("--epochs", type=int, help="Override epochs for every run in the sweep")
    p.add_argument("--num-workers", type=int,
                   help="Override num_workers for every run -- a sweep does many separate "
                        "train() calls (more DataLoader worker spawn/teardown cycles than one "
                        "run), which raises the odds of a known macOS hang at num_workers>0; "
                        "pass 0 if a run in the sweep seems to hang")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write the sweep summary (default: outputs/<sweep-name>_sweep/)")
    p.add_argument("--save-artifacts", action="store_true",
                   help="Also save each run's checkpoints/loss-curve/visualizations under "
                        "outputs/<experiment_name>/ (off by default -- a sweep only needs the "
                        "final label-efficiency plot, not N full per-run artifact sets)")
    args = p.parse_args()

    base = load_config(args.config)
    sweep_name = args.sweep_name or base.experiment_name
    out_dir = Path(args.out_dir) if args.out_dir else Path(base.base_dir) / f"{sweep_name}_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n in sorted(set(args.tile_counts)):
        overrides = {"max_train_tiles": n, "experiment_name": f"{sweep_name}_n{n}"}
        if args.epochs is not None:
            overrides["epochs"] = args.epochs
        if args.num_workers is not None:
            overrides["num_workers"] = args.num_workers
        config = load_config(args.config, overrides)
        print(f"\n=== label-efficiency sweep: budget={n} tiles -> experiment={config.experiment_name} ===")
        _, metrics = train(config, save_artifacts=args.save_artifacts)
        results.append({
            "requested_tiles": n,
            "actual_tiles": metrics["n_train_tiles"],
            "best_height_rmse_overall": metrics["best_height_rmse_overall"],
            "final_height_rmse_overall": metrics["height_rmse_overall"][-1],
        })
        if metrics["n_train_tiles"] != n:
            print(f"  NOTE: requested {n} tiles, only {metrics['n_train_tiles']} available in the training split")

    csv_path = out_dir / "label_efficiency_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["actual_tiles"] for r in results]
    ys = [r["best_height_rmse_overall"] for r in results]
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Training tiles used")
    ax.set_ylabel("Best overall height RMSE (m) — lower is better")
    ax.set_title(f"Label efficiency: {sweep_name}")
    ax.grid(True, alpha=0.3)
    plot_path = out_dir / "label_efficiency_curve.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
