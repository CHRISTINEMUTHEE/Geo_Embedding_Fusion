#!/usr/bin/env python
"""
Report per-source, per-split (train/val) class distribution: mean building/vegetation/
water coverage and % of tiles where each is present, using the exact same region-grouped,
stratified split each config's train() call would use (same data_root/source/val_split/
random_seed/stratify_threshold). Answers the question "did the split actually preserve enough
building/water tiles in each split" without having to train anything first.

Usage:
    python scripts/report_class_distribution.py --configs configs/0_baselines/0{3,4,5,6,7,8}_*.yaml
"""
import argparse
import csv
import glob

from emb2heights.config import load_config
from emb2heights.datamodule import LABEL_CLASSES, Embed2HeightsDataModule


def parse_args():
    p = argparse.ArgumentParser(description="Report per-source class distribution")
    p.add_argument("--configs", nargs="+", required=True, help="YAML config paths (globs OK)")
    p.add_argument("--out", default="outputs/class_distribution.csv")
    return p.parse_args()


def main():
    args = parse_args()
    config_paths = sorted({p for pattern in args.configs for p in glob.glob(pattern)} or args.configs)

    rows = []
    for config_path in config_paths:
        config = load_config(config_path)
        dm = Embed2HeightsDataModule(
            root=config.data_root,
            source=config.embedding_source,
            val_frac=config.val_split,
            seed=config.random_seed,
            max_train_tiles=config.max_train_tiles,
            stratify_threshold=config.stratify_threshold,
            standardize_bands=False,  # only the split/labels matter here, not embeddings
        ).setup()

        print(f"\n{config.embedding_source} ({config.experiment_name}): "
              f"{len(dm.train_regions)} train regions ({len(dm.train_ds)} tiles) / "
              f"{len(dm.val_regions)} val regions ({len(dm.val_ds)} tiles)")
        for split in ("train", "val"):
            row = {"source": config.embedding_source, "experiment_name": config.experiment_name,
                   "split": split, "n_tiles": len(dm.train_ds) if split == "train" else len(dm.val_ds)}
            for cls in LABEL_CLASSES:
                s = dm.class_distribution[split][cls]
                row[f"{cls}_mean_frac"] = s["mean_frac"]
                row[f"{cls}_pct_present"] = s["pct_tiles_present"]
            rows.append(row)
            print(f"  {split}: " + "  ".join(
                f"{cls}: mean={row[f'{cls}_mean_frac']:.3f} present={row[f'{cls}_pct_present']:.0f}%"
                for cls in LABEL_CLASSES))

    fieldnames = ["source", "experiment_name", "split", "n_tiles"] + [
        f"{cls}_{suffix}" for cls in LABEL_CLASSES for suffix in ("mean_frac", "pct_present")]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
