#!/usr/bin/env python
"""
Download the full embed2heights training split (labels + all six embedding sources).

Uses the public Hugging Face mirror (no EOTDL login). Resumable: files whose size
already matches the catalog are skipped. After the download, writes data/manifest.csv
so Embed2HeightsDataModule can load the full set with a region-grouped split.

Usage:
    python scripts/acquire.py
    python scripts/acquire.py --dirs data/train/labels data/train/alphaearth_emb
    python scripts/acquire.py --check

Only data/train/* is in DEFAULT_DIRS -- data/test/* embeddings have no labels.
On Unity, run via sbatch slurm/acquire.slurm so files land on /work (home quota
cannot hold ~110 GB).
"""
import argparse
import csv
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HF = "https://huggingface.co/datasets/troni21/esa_philab_embed2heights/resolve/main/"
DEFAULT_CATALOG = Path.home() / ".cache/emb2heights/catalog.v1.parquet"
DEFAULT_DIRS = [
    "data/train/labels",
    "data/train/alphaearth_emb",
    "data/train/tessera_emb",
    "data/train/thor_s1_emb",
    "data/train/thor_s2_emb",
    "data/train/terramind_s1_emb",
    "data/train/terramind_s2_emb",
]
DIR_TO_SOURCE = {
    "data/train/alphaearth_emb": "alphaearth",
    "data/train/tessera_emb": "tessera",
    "data/train/thor_s1_emb": "thor_s1",
    "data/train/thor_s2_emb": "thor_s2",
    "data/train/terramind_s1_emb": "terramind_s1",
    "data/train/terramind_s2_emb": "terramind_s2",
}
ID_RE = re.compile(r"_(\d{4})_([A-Z]{2})")
SOURCES = list(DIR_TO_SOURCE.values())


def ensure_catalog(catalog_path):
    path = Path(catalog_path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Fetching catalog -> {path}")
    urllib.request.urlretrieve(HF + "catalog.v1.parquet", tmp)
    tmp.replace(path)
    return path


def load_assets(catalog_path, dir_prefixes):
    """Catalog rows -> [(relative_path, expected_size)] for chosen dirs."""
    df = pd.read_parquet(ensure_catalog(catalog_path))
    prefixes = tuple(p.rstrip("/") + "/" for p in dir_prefixes)
    rows = df[df["id"].str.startswith(prefixes)]
    assets = []
    for _, row in rows.iterrows():
        assets.append((row["id"], int(row["assets"]["asset"]["size"])))
    return assets


def missing_assets(assets, out_root):
    return [(rel, size) for rel, size in assets
            if not (out_root / rel).exists() or (out_root / rel).stat().st_size != size]


def download_one(rel, size, dest, retries=8):
    if dest.exists() and dest.stat().st_size == size:
        return rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(HF + rel, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            actual = tmp.stat().st_size
            if actual != size:
                tmp.unlink(missing_ok=True)
                raise IOError(f"{rel}: size mismatch (got {actual}, expected {size})")
            tmp.replace(dest)
            return rel
        except urllib.error.HTTPError as e:
            last_err = e
            tmp.unlink(missing_ok=True)
            if e.code != 429:
                raise
            time.sleep(min(120, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            tmp.unlink(missing_ok=True)
            time.sleep(min(120, 2 ** attempt))
    raise last_err


# REVIEW REQUIRED
def download(assets, out_root, workers):
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, rel, size, out_root / rel): rel
                   for rel, size in assets}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                future.result()
            except Exception as e:
                failures.append((futures[future], str(e)))
    return failures


def write_manifest(out_root, dir_prefixes):
    """Pair on-disk train files into data/manifest.csv (paths relative to data/)."""
    tiles = defaultdict(dict)
    for d in dir_prefixes:
        d = d.rstrip("/")
        folder = out_root / d
        if not folder.is_dir():
            continue
        source = "label" if d == "data/train/labels" else DIR_TO_SOURCE.get(d)
        if source is None:
            continue
        for p in folder.glob("*.tif"):
            m = ID_RE.search(p.name)
            if not m:
                continue
            tid = f"{m.group(1)}_{m.group(2)}"
            rec = tiles[tid]
            rec["tile_id"] = tid
            rec["region"] = m.group(2)
            rel = str(Path(*Path(d).parts[1:]) / p.name)  # train/<dir>/<file>
            if source == "label":
                rec["label_path"] = rel
                year = p.stem.rsplit("_", 1)[-1]
                rec["year"] = year if year.isdigit() else ""
            else:
                rec[f"{source}_path"] = rel

    required = ["label"] + [DIR_TO_SOURCE[d.rstrip("/")]
                            for d in dir_prefixes
                            if d.rstrip("/") in DIR_TO_SOURCE]
    rows = []
    for tid in sorted(tiles):
        rec = tiles[tid]
        complete = all(
            (rec.get("label_path") if s == "label" else rec.get(f"{s}_path"))
            for s in required
        )
        rec.setdefault("year", "")
        rec["height"] = 256
        rec["width"] = 256
        rec["valid_frac"] = 1.0
        rec["label_nonzero_frac"] = 1.0
        ## keep=True if this tile has a label. Missing embedding sources are
        ## filtered per-config in Embed2HeightsDataModule (so AlphaEarth can
        ## train on 2024 tiles while thor_s2 is still downloading).
        rec["keep"] = bool(rec.get("label_path"))
        rec["_complete"] = complete
        for src in SOURCES:
            rec.setdefault(f"{src}_path", "")
        rec.setdefault("label_path", "")
        rows.append(rec)

    man = out_root / "data" / "manifest.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tile_id", "region", "year", "label_path"] + [f"{s}_path" for s in SOURCES] + [
        "height", "width", "valid_frac", "label_nonzero_frac", "keep",
    ]
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    kept = sum(1 for r in rows if r["keep"])
    n_complete = sum(1 for r in rows if r["_complete"])
    print(f"manifest: {man}  ({kept}/{len(rows)} tiles with labels, "
          f"{n_complete} complete for {required})")
    return man


def main():
    parser = argparse.ArgumentParser(description="Download embed2heights training data")
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                        help="Dataset directories to download (catalog id prefixes)")
    parser.add_argument("--out", type=str, default=".",
                        help="Output root (files land under <out>/data/...)")
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--check", action="store_true",
                        help="Only report what is complete/missing, download nothing")
    args = parser.parse_args()

    out_root = Path(args.out).resolve()
    assets = load_assets(args.catalog, args.dirs)
    todo = missing_assets(assets, out_root)
    total_gb = sum(s for _, s in todo) / 1e9
    print(f"{len(assets)} files in selection, {len(assets) - len(todo)} already complete, "
          f"{len(todo)} to download ({total_gb:.1f} GB)")

    if args.check:
        write_manifest(out_root, args.dirs)
        return

    failures = []
    if todo:
        failures = download(todo, out_root, args.workers)
        if failures:
            print(f"\n{len(failures)} files FAILED (rerun to retry):")
            for rel, err in failures[:10]:
                print(f"  {rel}: {err}")
        else:
            print("\nAll files downloaded and size-verified.")

    write_manifest(out_root, args.dirs)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
