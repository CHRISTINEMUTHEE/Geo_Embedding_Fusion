"""
Evaluate existing checkpoints (best_model.pth) on their own validation set,
one row per embedding source, into a single comparison table.

Also writes one "baseline" row (true mean/median height over nonzero-height
validation pixels, no model involved) and, per source, that same source's predicted
mean/median height at the identical pixels -- so a source with a small RMSE but a
predicted mean far from the true mean is visible directly, not just inferred from
the error metrics.

No training happens here -- this is a smoke test that the evaluation
pipeline (evaluate_metrics: pooled IoU + height MAE/RMSE) runs correctly
end-to-end against already-trained models. Skips configs with no checkpoint.

Usage:
    python scripts/evaluate_sources.py --configs configs/0_baselines/03_alphaearth_subset_datamodule.yaml configs/0_baselines/04_tessera_subset_datamodule.yaml
    python scripts/evaluate_sources.py --configs configs/0_baselines/*.yaml   # evaluates whatever has a checkpoint
"""
import argparse
import csv
import glob

import torch

from emb2heights.config import load_config
from emb2heights.models import build_model
from emb2heights.trainers import build_train_val_loaders, evaluate_metrics, get_device, height_distribution


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate existing checkpoints across sources")
    parser.add_argument("--configs", type=str, nargs="+", required=True, help="YAML config paths (globs OK)")
    parser.add_argument("--out", type=str, default="outputs/evaluation_table.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config_paths = sorted({p for pattern in args.configs for p in glob.glob(pattern)} or args.configs)
    device = get_device()

    rows = []
    baseline_added = False
    for config_path in config_paths:
        config = load_config(config_path)
        if not config.best_model_path.exists():
            print(f"Skip {config.experiment_name}: no checkpoint at {config.best_model_path}")
            continue

        _, val_loader = build_train_val_loaders(config)
        sample_img, _ = val_loader.dataset[0]
        model = build_model(config, sample_img.shape[0]).to(device)
        model.load_state_dict(torch.load(config.best_model_path, map_location=device))

        metrics = evaluate_metrics(model, val_loader, device, config.height_normalization_constant,
                                    config.iou_threshold, config.height_mask_threshold)
        metrics.update(height_distribution(val_loader, config.height_normalization_constant, device, model))
        metrics["source"] = config.embedding_source
        metrics["experiment_name"] = config.experiment_name
        rows.append(metrics)
        print(f"{config.embedding_source}: "
              f"MAE={metrics['mae_height']:.3f} RMSE={metrics['rmse_height']:.3f} "
              f"IoU(building/veg/water)={metrics['iou_building']:.3f}/"
              f"{metrics['iou_vegetation']:.3f}/{metrics['iou_water']:.3f}  "
              f"true mean/median={metrics['true_mean_height']:.2f}/{metrics['true_median_height']:.2f}  "
              f"pred mean/median={metrics['pred_mean_height']:.2f}/{metrics['pred_median_height']:.2f}")

        ## One shared "baseline" row (ground truth only, no model) -- every config here
        ## points at the same data_root's validation split, so this is the same number
        ## regardless of source; computing it once avoids a misleadingly repeated row.
        if not baseline_added:
            baseline = height_distribution(val_loader, config.height_normalization_constant, device)
            baseline["source"] = "baseline"
            baseline["experiment_name"] = ""
            rows.insert(0, baseline)
            baseline_added = True
            print(f"baseline (true height, nonzero pixels): "
                  f"mean={baseline['true_mean_height']:.2f}  median={baseline['true_median_height']:.2f}")

    if not rows:
        print("No checkpoints found -- nothing evaluated.")
        return

    fieldnames = ["source", "experiment_name", "mae_height", "rmse_height",
                  "rmse_building", "rmse_vegetation", "iou_building", "iou_vegetation", "iou_water",
                  "true_mean_height", "true_median_height", "pred_mean_height", "pred_median_height"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
