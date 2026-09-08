"""
Region-grouped data loading for embed2heights (full split or data/subset/).

Why this exists alongside datasets.py: the challenge tiles carry no coordinates, so the only
geographic grouping available is the anonymized region code. Splitting whole regions into
train/val is the closest honest analogue of a spatial split -- a tile from a validation region
can never appear in training. build_dataloaders() in datasets.py splits tiles at random, which
leaks across regions.

This module also fixes two data bugs that datasets.py does not handle: the embeddings are
int8-quantized (values in +/-127, nodata -128) and must be dequantized, and nodata must be
masked before arithmetic rather than collapsed into 0.0 by nan_to_num.
"""

import csv
import random

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

EMB_NODATA = -128.0
EMB_SCALE = 127.0

## Sources whose native embedding grid is coarser than the label's pixel grid (patch
## tokens, e.g. TerraMind/THOR at 16x16x768) need LatentTokenDataset instead of
## TilePairDataset. AlphaEarth/Tessera are pixel-aligned and are not listed here.
PATCH_SOURCES = {"terramind_s1", "terramind_s2", "thor_s1", "thor_s2"}

## Target channels 0/1/2 (see trainers.target_names); channel 3 is height, handled
## separately. Manifest columns "<cls>_frac" (mean per-tile coverage) drive both
## stratified train/val splitting and the class_balance_boost training sampler below.
LABEL_CLASSES = ("building", "vegetation", "water")


def read_manifest(manifest_path, keep_only=True):
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        ## Older manifests predate per-class fractions -- default to 0.0 rather than
        ## KeyError. Every region then lands in the same stratum, so split_regions()
        ## degrades to a plain random split automatically, no special-casing needed.
        has_class_fracs = bool(reader.fieldnames) and f"{LABEL_CLASSES[0]}_frac" in reader.fieldnames
        rows = list(reader)
    for r in rows:
        r["keep"] = r["keep"] == "True"
        r["valid_frac"] = float(r["valid_frac"])
        r["label_nonzero_frac"] = float(r["label_nonzero_frac"])
        for cls in LABEL_CLASSES:
            r[f"{cls}_frac"] = float(r[f"{cls}_frac"]) if has_class_fracs else 0.0
    return [r for r in rows if r["keep"]] if keep_only else rows


# REVIEW REQUIRED
## Whole regions go to exactly one split, so no region is ever in both. Regions are
## first bucketed by whether they contain any tile with meaningful building/water
## coverage (the two rare classes -- see LABEL_CLASSES), then split proportionally
## *within* each bucket, so a rare-class bucket can't be dropped entirely into one
## side by an unlucky shuffle. With no per-class data (old manifest, or a dataset
## with no building/water anywhere) every region lands in the same single bucket,
## which is exactly the old plain-shuffle behaviour -- no separate code path needed.
def split_regions(rows, val_frac=0.3, seed=42, stratify_threshold=0.01):
    by_region = {}
    for r in rows:
        by_region.setdefault(r["region"], []).append(r)

    def stratum(region_rows):
        has_building = any(t["building_frac"] > stratify_threshold for t in region_rows)
        has_water = any(t["water_frac"] > stratify_threshold for t in region_rows)
        return (has_building, has_water)

    buckets = {}
    for region, region_rows in by_region.items():
        buckets.setdefault(stratum(region_rows), []).append(region)

    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(buckets):  # deterministic iteration order
        group = sorted(buckets[key])
        rng.shuffle(group)
        n_val = round(len(group) * val_frac)
        ## A bucket with >=2 regions that actually carries a rare class always sends
        ## at least one region to val -- otherwise the class could vanish from
        ## validation IoU entirely just because round() rounded down to 0.
        if key != (False, False) and len(group) >= 2 and n_val == 0:
            n_val = 1
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    if not train:
        raise ValueError(f"val_frac={val_frac} left no training regions ({len(by_region)} total)")
    if not val:
        raise ValueError(f"val_frac={val_frac} left no validation regions ({len(by_region)} total)")
    assert not (set(train) & set(val)), "region leakage between train and val"
    bucket_sizes = {key: len(group) for key, group in buckets.items()}
    print(f"  stratified split: {len(buckets)} bucket(s), "
          f"(has_building, has_water) -> n_regions = {bucket_sizes}")
    return sorted(train), sorted(val)


## Not every source is int8-quantized (only AlphaEarth is, so far) -- detect it from
## the data itself rather than assuming: quantized values are integer-valued and
## bounded to +/-127, real float embeddings essentially never are both.
def _looks_quantized(values):
    if values.size == 0:
        return False
    return bool(np.abs(values).max() <= EMB_SCALE and np.allclose(values, np.round(values)))


## Dequantized embedding + per-pixel validity mask, before nodata zeroing or band
## standardization. Nodata comes from the file's own metadata (falls back to
## EMB_NODATA if a file has none set). Shared by TilePairDataset._read_embedding and
## scripts/compute_band_stats.py so both apply identical dequantization -- band
## statistics must be computed over exactly what training will see.
def read_embedding_raw(path):
    with rasterio.open(path) as s:
        raw = s.read().astype(np.float32)
        nodata = s.nodata if s.nodata is not None else EMB_NODATA
    valid = np.isfinite(raw) & (raw != nodata)
    if _looks_quantized(raw[valid]):
        raw = raw / EMB_SCALE
    return raw, valid


# REVIEW REQUIRED
class TilePairDataset(Dataset):
    """One 256x256 tile pair -> one cropped (embedding, target) sample."""

    def __init__(self, rows, root, source="alphaearth", patch_size=128,
                 is_train=True, height_norm=30.0, band_mean=None, band_std=None):
        self.rows = rows
        self.root = root
        self.source = source
        self.patch_size = patch_size
        self.is_train = is_train
        self.height_norm = height_norm
        ## Per-channel standardization stats from scripts/compute_band_stats.py
        ## (<data_root>/band_stats.json), train-split-only so val never leaks in.
        ## None (the default) means "no standardization" -- e.g. synthetic-data tests.
        self.band_mean = np.asarray(band_mean, dtype=np.float32) if band_mean is not None else None
        self.band_std = np.asarray(band_std, dtype=np.float32) if band_std is not None else None

    def __len__(self):
        return len(self.rows)

    def _read_embedding(self, path):
        raw, valid = read_embedding_raw(path)
        if self.band_mean is not None:
            raw = (raw - self.band_mean[:, None, None]) / self.band_std[:, None, None]
        return np.where(valid, raw, 0.0).astype(np.float32)

    def _read_target(self, path):
        with rasterio.open(path) as s:
            tar = np.nan_to_num(s.read().astype(np.float32))
        ## Band 3 is nDSM height in metres -> roughly [0, 1]; clip allows 1.5x outliers.
        tar[3] = np.clip(tar[3] / self.height_norm, 0.0, 1.5)
        return tar

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self._read_embedding(self.root / row[f"{self.source}_path"])
        target = self._read_target(self.root / row["label_path"])

        ## Tiles are not all 256x256 (255x256 also occurs); pad both to the crop size.
        h, w = image.shape[1], image.shape[2]
        ph, pw = max(0, self.patch_size - h), max(0, self.patch_size - w)
        if ph or pw:
            pad = ((0, 0), (0, ph), (0, pw))
            image = np.pad(image, pad, mode="reflect")
            target = np.pad(target, pad, mode="reflect")
            h, w = image.shape[1], image.shape[2]

        if self.is_train:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top, left = (h - self.patch_size) // 2, (w - self.patch_size) // 2

        sl = (slice(None), slice(top, top + self.patch_size),
              slice(left, left + self.patch_size))
        return torch.from_numpy(image[sl].copy()), torch.from_numpy(target[sl].copy())


# REVIEW REQUIRED
class LatentTokenDataset(TilePairDataset):
    """Patch-token embeddings (e.g. TerraMind/THOR, 16x16x768) kept at their native,
    low-resolution grid -- no upsampling. Padding and cropping happen at two scales:
    the embedding is cropped in token-space (patch_size // scale_factor tokens per
    side), and the target is cropped in pixel-space at the *exact* matching footprint
    (crop position and size both scaled by scale_factor from the token crop). Image
    and target come back at different resolutions on purpose -- for a model that
    decodes tokens itself, not LightUNet.
    """

    def __init__(self, rows, root, source="terramind_s1", patch_size=256,
                 scale_factor=16, is_train=True, height_norm=30.0,
                 band_mean=None, band_std=None):
        super().__init__(rows, root, source=source, patch_size=patch_size,
                          is_train=is_train, height_norm=height_norm,
                          band_mean=band_mean, band_std=band_std)
        self.scale_factor = scale_factor
        self.emb_patch_size = patch_size // scale_factor

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self._read_embedding(self.root / row[f"{self.source}_path"])
        target = self._read_target(self.root / row["label_path"])

        emb_patch_size = self.emb_patch_size
        ## Target crop size is derived from the *token* crop (emb_patch_size *
        ## scale_factor), not self.patch_size directly -- those only agree when
        ## patch_size divides evenly by scale_factor. Deriving it guarantees the
        ## target always covers exactly the ground footprint of the token crop,
        ## even when it doesn't.
        target_crop = emb_patch_size * self.scale_factor

        c, h_emb, w_emb = image.shape
        if h_emb < emb_patch_size or w_emb < emb_patch_size:
            pad_h = max(0, emb_patch_size - h_emb)
            pad_w = max(0, emb_patch_size - w_emb)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            h_emb, w_emb = image.shape[1], image.shape[2]

        _, h_tar, w_tar = target.shape
        if h_tar < target_crop or w_tar < target_crop:
            pad_h = max(0, target_crop - h_tar)
            pad_w = max(0, target_crop - w_tar)
            target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        ## Multi-scale cropping: pick the crop in token-space, then scale the same
        ## position and size up to pixel-space for the target -- so both crops cover
        ## the identical ground area, just at each source's native resolution.
        if self.is_train:
            top_emb = np.random.randint(0, h_emb - emb_patch_size + 1)
            left_emb = np.random.randint(0, w_emb - emb_patch_size + 1)
        else:
            top_emb = (h_emb - emb_patch_size) // 2
            left_emb = (w_emb - emb_patch_size) // 2

        image = image[:, top_emb:top_emb + emb_patch_size, left_emb:left_emb + emb_patch_size]

        top_tar, left_tar = top_emb * self.scale_factor, left_emb * self.scale_factor
        target = target[:, top_tar:top_tar + target_crop, left_tar:left_tar + target_crop]

        return torch.from_numpy(image.copy()), torch.from_numpy(target.copy())


## Per-split, per-class summary -- how much of each split actually carries each
## label class. Used both by Embed2HeightsDataModule's setup() printout and
## scripts/report_class_distribution.py.
def summarize_class_distribution(rows, threshold=0.01):
    n = len(rows)
    summary = {}
    for cls in LABEL_CLASSES:
        fracs = [r[f"{cls}_frac"] for r in rows]
        summary[cls] = {
            "mean_frac": sum(fracs) / n if n else 0.0,
            "pct_tiles_present": 100.0 * sum(1 for f in fracs if f > threshold) / n if n else 0.0,
        }
    return summary


# REVIEW REQUIRED
class Embed2HeightsDataModule:
    """Manifest -> region-grouped train/val loaders. Plain Python, no Lightning."""

    def __init__(self, root, manifest=None, source="alphaearth", patch_size=128,
                 batch_size=8, num_workers=0, val_frac=0.3, seed=42, height_norm=30.0,
                 scale_factor=16, max_train_tiles=None, stratify_threshold=0.01,
                 class_balance_boost=3.0, standardize_bands=True, band_stats_path=None):
        from pathlib import Path
        self.root = Path(root)
        self.manifest = Path(manifest) if manifest else self.root / "manifest.csv"
        self.source = source
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.seed = seed
        self.height_norm = height_norm
        self.scale_factor = scale_factor  # only used when source is in PATCH_SOURCES
        ## Caps *training* tiles only -- val stays full so different label budgets are
        ## still compared against the same validation set. A real label-efficiency sweep
        ## (see scripts/label_efficiency_sweep.py) needs this: multiple separate runs at
        ## different max_train_tiles, compared by best achieved accuracy -- not one run's
        ## epoch-by-epoch curve, which was the previous (incorrect) label-efficiency plot.
        self.max_train_tiles = max_train_tiles
        ## Shared by split_regions() (bucket regions by rare-class presence) and the
        ## training WeightedRandomSampler (which tiles count as "rare-class present").
        self.stratify_threshold = stratify_threshold
        ## Relative sampling weight for training tiles with building/water present,
        ## vs. weight 1.0 for every other tile -- e.g. 3.0 means such a tile is drawn
        ## ~3x as often per epoch as a tile with neither class. Only affects *sampling
        ## frequency*, not the loss itself (see losses.py's bg_weight for that lever).
        self.class_balance_boost = class_balance_boost
        self.standardize_bands = standardize_bands
        self.band_stats_path = Path(band_stats_path) if band_stats_path else self.root / "band_stats.json"
        self.train_ds = self.val_ds = None
        self.train_regions = self.val_regions = []
        self.n_train_tiles_used = None
        self.class_distribution = {}
        self.train_sampler = None

    def _load_band_stats(self):
        import json
        if not self.standardize_bands:
            return None, None
        if not self.band_stats_path.exists():
            print(f"  no band_stats.json at {self.band_stats_path} -- training on raw "
                  f"(unstandardized) embeddings; run scripts/compute_band_stats.py first")
            return None, None
        with open(self.band_stats_path) as f:
            stats = json.load(f)
        if self.source not in stats:
            print(f"  band_stats.json has no entry for source={self.source!r} -- "
                  f"skipping standardization for this source")
            return None, None
        return stats[self.source]["mean"], stats[self.source]["std"]

    def setup(self):
        rows = read_manifest(self.manifest)
        src_key = f"{self.source}_path"
        rows = [r for r in rows if r.get("label_path") and r.get(src_key)]
        if not rows:
            raise ValueError(
                f"No usable tiles in {self.manifest} for source={self.source} "
                f"(need keep=True plus non-empty label_path and {src_key})"
            )
        self.train_regions, self.val_regions = split_regions(
            rows, self.val_frac, self.seed, self.stratify_threshold)

        tr = [r for r in rows if r["region"] in set(self.train_regions)]
        va = [r for r in rows if r["region"] in set(self.val_regions)]
        common = {r["tile_id"] for r in tr} & {r["tile_id"] for r in va}
        assert not common, f"tile leakage between splits: {common}"

        if self.max_train_tiles is not None:
            ## Shuffle once with a fixed seed, then take the first N -- so budget=10 is a
            ## subset of budget=25's tiles, giving a genuine "adding more labels" sweep
            ## rather than unrelated random draws per budget.
            tr = sorted(tr, key=lambda r: r["tile_id"])
            random.Random(self.seed).shuffle(tr)
            tr = tr[:self.max_train_tiles]
        self.n_train_tiles_used = len(tr)
        self.class_distribution = {"train": summarize_class_distribution(tr, self.stratify_threshold),
                                    "val": summarize_class_distribution(va, self.stratify_threshold)}

        band_mean, band_std = self._load_band_stats()

        kw = dict(root=self.root, source=self.source, patch_size=self.patch_size,
                  height_norm=self.height_norm, band_mean=band_mean, band_std=band_std)
        if self.source in PATCH_SOURCES:
            dataset_cls = LatentTokenDataset
            kw["scale_factor"] = self.scale_factor
        else:
            dataset_cls = TilePairDataset
        self.train_ds = dataset_cls(tr, is_train=True, **kw)
        self.val_ds = dataset_cls(va, is_train=False, **kw)

        ## Oversample training tiles that carry the rare classes so each epoch sees
        ## more of them than plain uniform tile sampling would -- the imbalance is at
        ## the *tile* level (most tiles have ~0% building/water) as well as the pixel
        ## level, so shuffling tiles uniformly under-exposes the model to them.
        weights = [1.0 + self.class_balance_boost
                   if (r["building_frac"] > self.stratify_threshold
                       or r["water_frac"] > self.stratify_threshold) else 1.0
                   for r in tr]
        self.train_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True) if weights else None
        return self

    def _loader(self, ds, shuffle, sampler=None):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle if sampler is None else False,
                          sampler=sampler, num_workers=self.num_workers)

    def train_dataloader(self):
        return self._loader(self.train_ds, True, sampler=self.train_sampler)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)
