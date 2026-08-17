#!/usr/bin/env python
"""
Download a small, region-balanced subset of embed2heights from the public HF mirror.

The full dataset is 147.8 GB / 19,846 files. This pulls <1% of it so the pipeline can be
developed and verified on a laptop. Regions are the only geographic grouping the dataset
exposes (coordinates were stripped by the organizers), so we sample whole regions and let
the DataModule hold entire regions out for validation.

Usage:
    python scripts/acquire_subset.py --dry-run          # show selection, download nothing
    python scripts/acquire_subset.py --limit 6          # quick smoke test
    python scripts/acquire_subset.py                    # full ~1.4 GB subset
"""
import argparse
import csv
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

HF = "https://huggingface.co/datasets/troni21/esa_philab_embed2heights/resolve/main/"
CATALOG = HF + "catalog.v1.parquet"

## Each source pairs an embedding directory with the shared label directory.
SOURCES = {
    "alphaearth": "data/train/alphaearth_emb",
    "tessera": "data/train/tessera_emb",
}
LABEL_DIR = "data/train/labels"

## 'gee_emb_1431_KE.tif' / 'tessera_emb_1431_KE.tif' / 'label_1431_KE_2023.tif' -> (1431, 'KE')
ID_RE = re.compile(r"_(\d{4})_([A-Z]{2})")


def load_catalog(sources):
    """Catalog -> DataFrame of (tile_id, region, source dirs, hrefs, sizes) for paired tiles."""
    df = pd.read_parquet(CATALOG)
    df["dir"] = df["id"].str.rsplit("/", n=1).str[0]
    df["fname"] = df["id"].str.rsplit("/", n=1).str[-1]
    df["size"] = df["assets"].apply(lambda a: a["asset"]["size"])

    keep_dirs = [LABEL_DIR] + [SOURCES[s] for s in sources]
    df = df[df["dir"].isin(keep_dirs)].copy()

    ids = df["fname"].str.extract(ID_RE)
    df["num"] = pd.to_numeric(ids[0])
    df["region"] = ids[1]
    df = df.dropna(subset=["num", "region"])
    df["tile_id"] = df["num"].astype(int).astype(str).str.zfill(4) + "_" + df["region"]
    ## Labels carry the year, embeddings do not; year travels with the label row.
    df["year"] = df["fname"].str.extract(r"_(\d{4})\.tif$")[0]
    return df


def build_tiles(df, sources):
    """Rows -> one record per tile_id that has a label AND every requested embedding source."""
    labels = df[df["dir"] == LABEL_DIR].set_index("tile_id")
    tiles = {}
    for tid, lab in labels.iterrows():
        rec = {"tile_id": tid, "region": lab["region"], "year": lab["year"],
               "files": [(LABEL_DIR, lab["id"], lab["size"])], "size": lab["size"]}
        tiles[tid] = rec

    for src in sources:
        sub = df[df["dir"] == SOURCES[src]].set_index("tile_id")
        for tid in list(tiles):
            if tid not in sub.index:
                del tiles[tid]  # incomplete pair, drop
                continue
            row = sub.loc[tid]
            tiles[tid]["files"].append((SOURCES[src], row["id"], row["size"]))
            tiles[tid]["size"] += row["size"]
    return list(tiles.values())


# REVIEW REQUIRED
## Whole regions are the sampling unit so train/val can be split without geographic leakage.
def select_regions(tiles, max_gb, seed, min_region_tiles):
    by_region = {}
    for t in tiles:
        by_region.setdefault(t["region"], []).append(t)

    eligible = {r: ts for r, ts in by_region.items() if len(ts) >= min_region_tiles}
    rng = np.random.default_rng(seed)
    order = sorted(eligible)  # sort first so the shuffle is reproducible across pandas versions
    rng.shuffle(order)

    budget = max_gb * 1e9
    chosen, total = [], 0
    for region in order:
        rsize = sum(t["size"] for t in eligible[region])
        if total + rsize > budget:
            continue
        chosen.append(region)
        total += rsize
    picked = [t for r in chosen for t in sorted(eligible[r], key=lambda t: t["tile_id"])]
    return sorted(chosen), picked, total


def download_one(rel_id, size, dest):
    if dest.exists() and dest.stat().st_size == size:
        return  # resumable: size match means already complete
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(HF + rel_id, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{rel_id}: size mismatch (got {got}, expected {size})")
    tmp.rename(dest)


def local_path(out_root, dir_name, rel_id):
    """Label dirs -> outputs/, embedding dirs -> inputs/<source>/."""
    fname = rel_id.rsplit("/", 1)[-1]
    if dir_name == LABEL_DIR:
        return out_root / "outputs" / fname
    src = next(s for s, d in SOURCES.items() if d == dir_name)
    return out_root / "inputs" / src / fname


# REVIEW REQUIRED
## QC: a tile is only usable if its pixels are real. -128 is the embedding nodata sentinel.
def qc_tile(tile, out_root, sources, min_valid):
    row = {"tile_id": tile["tile_id"], "region": tile["region"], "year": tile["year"]}
    label_p = local_path(out_root, LABEL_DIR, tile["files"][0][1])
    row["label_path"] = str(label_p.relative_to(out_root))

    with rasterio.open(label_p) as s:
        lab = s.read().astype(np.float32)
    row["height"], row["width"] = lab.shape[1], lab.shape[2]
    row["label_nonzero_frac"] = float(np.mean(np.any(lab != 0, axis=0)))

    valid_fracs = []
    for src in sources:
        p = local_path(out_root, SOURCES[src], next(
            f[1] for f in tile["files"] if f[0] == SOURCES[src]))
        row[f"{src}_path"] = str(p.relative_to(out_root))
        with rasterio.open(p) as s:
            emb = s.read().astype(np.float32)
        ok = np.isfinite(emb) & (emb != -128.0)
        valid_fracs.append(float(ok.all(axis=0).mean()))
    row["valid_frac"] = min(valid_fracs)

    row["keep"] = bool(row["valid_frac"] >= min_valid and row["label_nonzero_frac"] > 0.0)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/subset")
    p.add_argument("--sources", nargs="+", default=["alphaearth"], choices=list(SOURCES))
    p.add_argument("--max-gb", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-region-tiles", type=int, default=8)
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="Drop tiles whose valid-pixel fraction is below this")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, help="Only take N tiles (smoke test)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out)
    print(f"Reading catalog: {CATALOG}")
    df = load_catalog(args.sources)
    tiles = build_tiles(df, args.sources)
    regions, picked, total = select_regions(
        tiles, args.max_gb, args.seed, args.min_region_tiles)

    if args.limit:
        picked = picked[:args.limit]
        regions = sorted({t["region"] for t in picked})
        total = sum(t["size"] for t in picked)

    print(f"\nsources={args.sources}  seed={args.seed}  budget={args.max_gb} GB")
    print(f"selected {len(regions)} regions, {len(picked)} tiles, {total / 1e9:.2f} GB")
    counts = {}
    for t in picked:
        counts[t["region"]] = counts.get(t["region"], 0) + 1
    print("  " + "  ".join(f"{r}:{counts[r]}" for r in regions))

    if args.dry_run:
        print("\n--dry-run: nothing downloaded")
        return

    jobs = [(rid, size, local_path(out_root, d, rid))
            for t in picked for d, rid, size in t["files"]]
    todo = [j for j in jobs if not (j[2].exists() and j[2].stat().st_size == j[1])]
    print(f"\n{len(jobs)} files, {len(jobs) - len(todo)} already complete, {len(todo)} to fetch")

    failures = []
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(download_one, *j): j[0] for j in todo}
            for f in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
                try:
                    f.result()
                except Exception as e:
                    failures.append((futs[f], str(e)))
    if failures:
        print(f"\n{len(failures)} downloads FAILED (rerun to retry):")
        for rel, err in failures[:10]:
            print(f"  {rel}: {err}")
        return

    print("\nRunning QC...")
    rows = [qc_tile(t, out_root, args.sources, args.min_valid)
            for t in tqdm(picked, desc="QC")]
    man = out_root / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    kept = sum(r["keep"] for r in rows)
    print(f"\nmanifest: {man}")
    print(f"  {kept}/{len(rows)} tiles kept, {len(rows) - kept} dropped")
    for r in rows:
        if not r["keep"]:
            print(f"    DROP {r['tile_id']}  valid_frac={r['valid_frac']:.3f} "
                  f"label_nonzero={r['label_nonzero_frac']:.3f}")


if __name__ == "__main__":
    main()
