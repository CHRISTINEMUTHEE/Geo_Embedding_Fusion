#!/usr/bin/env python
"""
Compute per-source, per-channel mean/std over the TRAINING split only (region-grouped,
same split_regions() the DataModule uses -- val never leaks into these stats), and
write <data_root>/band_stats.json. Embed2HeightsDataModule loads this automatically
(standardize_bands=True, the default) to standardize each source's raw embedding scale
before it reaches the model -- without this, sources with very different native ranges
(AlphaEarth dequantizes to roughly [-1, 1]; Tessera/THOR/TerraMind are raw floats from
their own models with unrelated scales) aren't on comparable footing for fusion-vs-best-single-source comparison.

Usage:
    python scripts/compute_band_stats.py --data-root data/subset
    python scripts/compute_band_stats.py --data-root data --sources alphaearth tessera
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from emb2heights.datamodule import read_embedding_raw, read_manifest, split_regions


def compute_stats(rows, root, source):
    n_channels = None
    total = sumsq = count = None
    for row in tqdm(rows, desc=source):
        raw, valid = read_embedding_raw(root / row[f"{source}_path"])
        if n_channels is None:
            n_channels = raw.shape[0]
            total = np.zeros(n_channels, dtype=np.float64)
            sumsq = np.zeros(n_channels, dtype=np.float64)
            count = np.zeros(n_channels, dtype=np.int64)
        for c in range(n_channels):
            vals = raw[c][valid[c]]
            total[c] += vals.sum()
            sumsq[c] += (vals.astype(np.float64) ** 2).sum()
            count[c] += vals.size

    if count is None or (count == 0).any():
        raise ValueError(f"{source}: no valid pixels found in {len(rows)} training tiles")
    mean = total / count
    var = sumsq / count - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))  # floor avoids div-by-zero for a dead channel
    return mean, std


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/subset")
    p.add_argument("--manifest", default=None, help="Default: <data-root>/manifest.csv")
    p.add_argument("--sources", nargs="+",
                   default=["alphaearth", "tessera", "thor_s1", "thor_s2", "terramind_s1", "terramind_s2"])
    p.add_argument("--val-frac", type=float, default=0.3,
                   help="Must match the val_split every config for this data_root uses")
    p.add_argument("--seed", type=int, default=42, help="Must match every config's random_seed")
    p.add_argument("--stratify-threshold", type=float, default=0.01)
    p.add_argument("--out", default=None, help="Default: <data-root>/band_stats.json")
    args = p.parse_args()

    root = Path(args.data_root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"
    out = Path(args.out) if args.out else root / "band_stats.json"

    rows = read_manifest(manifest)
    train_regions, _ = split_regions(rows, args.val_frac, args.seed, args.stratify_threshold)
    tr = [r for r in rows if r["region"] in set(train_regions)]
    print(f"{len(tr)} training tiles across {len(train_regions)} regions "
          f"(val_frac={args.val_frac}, seed={args.seed})")

    stats = {}
    for source in args.sources:
        mean, std = compute_stats(tr, root, source)
        stats[source] = {"mean": mean.tolist(), "std": std.tolist()}
        print(f"  {source}: {len(mean)} channels, mean range "
              f"[{mean.min():.4f}, {mean.max():.4f}], std range [{std.min():.4f}, {std.max():.4f}]")

    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
