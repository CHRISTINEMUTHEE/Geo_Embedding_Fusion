#!/usr/bin/env python
"""
Selective downloader for the embed2heights EOTDL dataset.

The eotdl CLI can only stage ALL assets (110+ GB) or one file at a time. This
script reads the already-staged STAC catalog and downloads only the directories
you need. Resumable: files whose size already matches the catalog are skipped.

Usage:
    # baseline needs: labels (2.1 GB) + alphaearth (33.9 GB)
    python scripts/acquire.py
    # other sources later:
    python scripts/acquire.py --dirs data/train/tessera_emb data/test/tessera_test_emb
    # verify what's on disk without downloading:
    python scripts/acquire.py --check
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

CATALOG = Path.home() / ".cache/eotdl/datasets/embed2heights/catalog.v1.parquet"
DEFAULT_DIRS = [
    "data/train/labels",
    "data/train/alphaearth_emb",
    "data/test/alphaearth_test_emb",
    # Tessera
    "data/train/tessera_emb",
    "data/test/tessera_test_emb",
    # Thor S1
    "data/train/thor_s1_emb",
    "data/test/thor_test_s1_emb",
    "data/train/thor_s2_emb",
    "data/test/thor_test_s2_emb",
    # Terramind S1s
    "data/train/terramind_s1_emb",
    "data/test/terramind_test_s1_emb",
    "data/train/terramind_s2_emb",
    "data/test/terramind_test_s2_emb",
]


def load_assets(catalog_path, dir_prefixes):
    """Catalog rows -> [(relative_path, href, expected_size)] for chosen dirs."""
    df = pd.read_parquet(catalog_path)
    prefixes = tuple(p.rstrip("/") + "/" for p in dir_prefixes)
    rows = df[df["id"].str.startswith(prefixes)]
    assets = []
    for _, row in rows.iterrows():
        asset = row["assets"]["asset"]
        assets.append((row["id"], asset["href"], int(asset["size"])))
    return assets


def missing_assets(assets, out_root):
    return [(rel, href, size) for rel, href, size in assets
            if not (out_root / rel).exists() or (out_root / rel).stat().st_size != size]


# REVIEW REQUIRED
def download(assets, out_root, workers):
    from eotdl.auth import auth
    from eotdl.repos import FilesAPIRepo

    user = auth()  # one login, shared by all workers
    repo = FilesAPIRepo()

    def fetch(rel, href, size):
        repo.stage_file_url(href, str(out_root), user)
        actual = (out_root / rel).stat().st_size
        if actual != size:
            raise IOError(f"{rel}: size mismatch (got {actual}, expected {size})")
        return rel

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, *a): a[0] for a in assets}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            try:
                future.result()
            except Exception as e:
                failures.append((futures[future], str(e)))
    return failures


def main():
    parser = argparse.ArgumentParser(description="Download embed2heights data selectively")
    parser.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                        help="Dataset directories to download (catalog id prefixes)")
    parser.add_argument("--out", type=str, default=".",
                        help="Output root (files land under <out>/data/...)")
    parser.add_argument("--catalog", type=str, default=str(CATALOG))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check", action="store_true",
                        help="Only report what is complete/missing, download nothing")
    args = parser.parse_args()

    out_root = Path(args.out)
    assets = load_assets(args.catalog, args.dirs)
    todo = missing_assets(assets, out_root)
    total_gb = sum(s for _, _, s in todo) / 1e9
    print(f"{len(assets)} files in selection, {len(assets) - len(todo)} already complete, "
          f"{len(todo)} to download ({total_gb:.1f} GB)")

    if args.check or not todo:
        return

    failures = download(todo, out_root, args.workers)
    if failures:
        print(f"\n{len(failures)} files FAILED (rerun to retry):")
        for rel, err in failures[:10]:
            print(f"  {rel}: {err}")
    else:
        print("\nAll files downloaded and size-verified.")


if __name__ == "__main__":
    main()
