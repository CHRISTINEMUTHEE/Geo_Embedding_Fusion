"""
Evaluate existing checkpoints (best_model.pth) on their own validation set,
one row per embedding source, into a single comparison table.

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
from emb2heights.trainers import build_train_val_loaders, evaluate_metrics, get_device


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
        metrics["source"] = config.embedding_source
        metrics["experiment_name"] = config.experiment_name
        rows.append(metrics)
        print(f"{config.embedding_source}: "
              f"MAE={metrics['mae_height']:.3f} RMSE={metrics['rmse_height']:.3f} "
              f"IoU(building/veg/water)={metrics['iou_building']:.3f}/"
              f"{metrics['iou_vegetation']:.3f}/{metrics['iou_water']:.3f}")

    if not rows:
        print("No checkpoints found -- nothing evaluated.")
        return

    fieldnames = ["source", "experiment_name", "mae_height", "rmse_height",
                  "rmse_building", "rmse_vegetation", "iou_building", "iou_vegetation", "iou_water"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
