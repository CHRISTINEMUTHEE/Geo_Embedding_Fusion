#!/usr/bin/env python
"""
End-to-end check that the region-grouped DataModule loads training and validation patches.

Prints numbers an agent can verify and writes a PNG a human can eyeball.

Usage:
    python artifacts/verify_datamodule.py
    python artifacts/verify_datamodule.py --root data/subset --n-samples 4
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emb2heights.datamodule import Embed2HeightsDataModule, read_manifest  # noqa: E402

BANDS = ["building %", "vegetation %", "water %", "nDSM height"]


## 64 embedding channels -> 3 principal components, stretched to [0,1] for display only.
def pca_rgb(emb):
    c, h, w = emb.shape
    x = emb.reshape(c, -1).T
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    rgb = (x @ vt[:3].T).T.reshape(3, h, w)
    lo = np.percentile(rgb, 2, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(1, 2), keepdims=True)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1).transpose(1, 2, 0)


def check_batch(name, loader, patch_size, failures):
    x, y = next(iter(loader))
    print(f"\n[{name}] batch: image {tuple(x.shape)} {x.dtype} | target {tuple(y.shape)} {y.dtype}")
    print(f"  embedding range [{x.min():.3f}, {x.max():.3f}]  NaNs={int(torch.isnan(x).sum())}")
    print(f"  target range    [{y.min():.3f}, {y.max():.3f}]  NaNs={int(torch.isnan(y).sum())}")
    for i, b in enumerate(BANDS):
        print(f"    band{i} {b:13s} [{y[:, i].min():.3f}, {y[:, i].max():.3f}]")

    def chk(cond, msg):
        print(f"  {'PASS' if cond else 'FAIL'}: {msg}")
        if not cond:
            failures.append(f"{name}: {msg}")

    chk(x.shape[2] == x.shape[3] == patch_size, f"image is {patch_size}x{patch_size}")
    chk(y.shape[1] == 4, "target has 4 bands")
    chk(x.shape[0] == y.shape[0], "image/target batch sizes match")
    chk(int(torch.isnan(x).sum()) == 0 and int(torch.isnan(y).sum()) == 0, "no NaNs")
    ## Dequantization fired: raw tiles are int8-valued (+/-127), so anything above ~2 means
    ## the /127 rescale was skipped.
    chk(float(x.abs().max()) <= 2.0, "embeddings dequantized to ~[-1, 1]")
    chk(float(y[:, 3].max()) <= 1.5, "height normalized and clipped to <= 1.5")
    return x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/subset")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--out", default="artifacts/output/samples.png")
    args = p.parse_args()

    root = Path(args.root)
    failures = []

    all_rows = read_manifest(root / "manifest.csv", keep_only=False)
    kept = [r for r in all_rows if r["keep"]]
    print(f"=== manifest: {len(kept)}/{len(all_rows)} tiles kept ===")
    for r in all_rows:
        if not r["keep"]:
            print(f"  DROPPED {r['tile_id']}  valid_frac={r['valid_frac']:.3f} "
                  f"label_nonzero={r['label_nonzero_frac']:.3f}")
    shapes = {(int(r["height"]), int(r["width"])) for r in kept}
    print(f"  tile shapes present: {sorted(shapes)}")

    dm = Embed2HeightsDataModule(root=root, patch_size=args.patch_size,
                                 batch_size=args.batch_size).setup()

    print("\n=== region-grouped split ===")
    print(f"  train regions ({len(dm.train_regions)}): {dm.train_regions} -> {len(dm.train_ds)} tiles")
    print(f"  val   regions ({len(dm.val_regions)}): {dm.val_regions} -> {len(dm.val_ds)} tiles")
    overlap = set(dm.train_regions) & set(dm.val_regions)
    print(f"  {'PASS' if not overlap else 'FAIL'}: region sets are disjoint (overlap={overlap or '{}'})")
    if overlap:
        failures.append("region sets overlap")
    for nm, ds in [("train", dm.train_ds), ("val", dm.val_ds)]:
        if len(ds) == 0:
            failures.append(f"{nm} split is empty")
            print(f"  FAIL: {nm} split is empty")

    xt, yt = check_batch("train", dm.train_dataloader(), args.patch_size, failures)
    xv, yv = check_batch("val", dm.val_dataloader(), args.patch_size, failures)

    ## Val crop is deterministic; train crop is random.
    a, _ = dm.val_ds[0]
    b, _ = dm.val_ds[0]
    same = torch.equal(a, b)
    print(f"\n  {'PASS' if same else 'FAIL'}: val crop is deterministic")
    if not same:
        failures.append("val crop not deterministic")

    n = min(args.n_samples, xt.shape[0], xv.shape[0])
    rows = 2 * n
    fig, axes = plt.subplots(rows, 5, figsize=(14, 2.9 * rows), squeeze=False,
                             layout="constrained")
    for si, (tag, x, y) in enumerate([("train", xt, yt), ("val", xv, yv)]):
        for k in range(n):
            r = si * n + k
            axes[r, 0].imshow(pca_rgb(x[k].numpy()))
            axes[r, 0].set_ylabel(f"{tag} #{k}", fontsize=11)
            axes[r, 0].set_title("embedding (PCA-RGB)" if r == 0 else "")
            for i in range(4):
                cmap = "viridis" if i == 3 else "magma"
                im = axes[r, i + 1].imshow(y[k, i].numpy(), cmap=cmap, vmin=0,
                                           vmax=1.5 if i == 3 else 1)
                axes[r, i + 1].set_title(BANDS[i] if r == 0 else "")
                fig.colorbar(im, ax=axes[r, i + 1], fraction=0.046)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle("embed2heights: region-grouped train/val patches", fontsize=14)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED:"))
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
