"""
Region-grouped data loading for the embed2heights subset.

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
from torch.utils.data import DataLoader, Dataset

EMB_NODATA = -128.0
EMB_SCALE = 127.0

## Sources whose native embedding grid is coarser than the label's pixel grid (patch
## tokens, e.g. TerraMind/THOR at 16x16x768) need PatchEmbeddingsDataset instead of
## TilePairDataset. AlphaEarth/Tessera are pixel-aligned and are not listed here.
PATCH_SOURCES = {"terramind_s1", "terramind_s2", "thor_s1", "thor_s2"}


def read_manifest(manifest_path, keep_only=True):
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["keep"] = r["keep"] == "True"
        r["valid_frac"] = float(r["valid_frac"])
        r["label_nonzero_frac"] = float(r["label_nonzero_frac"])
    return [r for r in rows if r["keep"]] if keep_only else rows


# REVIEW REQUIRED
## Whole regions go to exactly one split, so no region is ever in both.
def split_regions(rows, val_frac=0.3, seed=42):
    regions = sorted({r["region"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(regions)
    n_val = max(1, round(len(regions) * val_frac))
    val = set(regions[:n_val])
    train = set(regions[n_val:])
    if not train:
        raise ValueError(f"val_frac={val_frac} left no training regions ({len(regions)} total)")
    assert not (train & val), "region leakage between train and val"
    return sorted(train), sorted(val)


# REVIEW REQUIRED
class TilePairDataset(Dataset):
    """One 256x256 tile pair -> one cropped (embedding, target) sample."""

    def __init__(self, rows, root, source="alphaearth", patch_size=128,
                 is_train=True, height_norm=30.0):
        self.rows = rows
        self.root = root
        self.source = source
        self.patch_size = patch_size
        self.is_train = is_train
        self.height_norm = height_norm

    def __len__(self):
        return len(self.rows)

    ## Not every source is int8-quantized (only AlphaEarth is, so far) -- detect it from
    ## the data itself rather than assuming: quantized values are integer-valued and
    ## bounded to +/-127, real float embeddings essentially never are both. Nodata comes
    ## from the file's own metadata (falls back to EMB_NODATA if a file has none set).
    @staticmethod
    def _looks_quantized(values):
        if values.size == 0:
            return False
        return bool(np.abs(values).max() <= EMB_SCALE and np.allclose(values, np.round(values)))

    def _read_embedding(self, path):
        with rasterio.open(path) as s:
            raw = s.read().astype(np.float32)
            nodata = s.nodata if s.nodata is not None else EMB_NODATA

        valid = np.isfinite(raw) & (raw != nodata)
        if self._looks_quantized(raw[valid]):
            return np.where(valid, raw / EMB_SCALE, 0.0).astype(np.float32)
        return np.where(valid, raw, 0.0).astype(np.float32)

    def _read_target(self, path):
        with rasterio.open(path) as s:
            tar = np.nan_to_num(s.read().astype(np.float32))
        ## Band 3 is nDSM height in metres -> roughly [0, 1]; clip allows 1.5x outliers.
        tar[3] = np.clip(tar[3] / self.height_norm, 0.0, 1.5)
        return tar

    ## Hook for subclasses whose native embedding grid doesn't match the target's
    ## pixel grid (see PatchEmbeddingsDataset). No-op here: pixel-aligned sources
    ## already match, so this class's behavior is unchanged.
    def _postprocess_embedding(self, image, target_hw):
        return image

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = self._read_embedding(self.root / row[f"{self.source}_path"])
        target = self._read_target(self.root / row["label_path"])
        image = self._postprocess_embedding(image, target.shape[1:])

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
class PatchEmbeddingsDataset(TilePairDataset):
    """Patch-token embeddings (e.g. TerraMind/THOR, 16x16x768) nearest-neighbor
    upsampled to the label's native pixel grid before cropping, so they plug into
    the same crop/model pipeline as TilePairDataset.

    This is a data-loader-level fix, not the "latent-based fusion" research_questions.md
    describes for THOR/TerraMind (fusing at the model's latent representation, with a
    learned upsample) -- it gets a first single-source patch baseline running on the
    existing pixel-wise architecture. Revisit before drawing fusion conclusions from it:
    every 16x16 block of "pixels" here is a repeated constant, not new spatial detail.
    """

    def _postprocess_embedding(self, image, target_hw):
        c, ph, pw = image.shape
        th, tw = target_hw
        if (ph, pw) == (th, tw):
            return image
        ## Nearest-neighbor via index repeat: each patch token maps to a contiguous
        ## block of pixels, so a later crop never splits one token across two crops.
        row_idx = np.arange(th) * ph // th
        col_idx = np.arange(tw) * pw // tw
        return image[:, row_idx][:, :, col_idx]


# REVIEW REQUIRED
class Embed2HeightsDataModule:
    """Manifest -> region-grouped train/val loaders. Plain Python, no Lightning."""

    def __init__(self, root, manifest=None, source="alphaearth", patch_size=128,
                 batch_size=8, num_workers=0, val_frac=0.3, seed=42, height_norm=30.0):
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
        self.train_ds = self.val_ds = None
        self.train_regions = self.val_regions = []

    def setup(self):
        rows = read_manifest(self.manifest)
        if not rows:
            raise ValueError(f"No usable tiles in {self.manifest} (all rows have keep=False)")
        self.train_regions, self.val_regions = split_regions(rows, self.val_frac, self.seed)

        tr = [r for r in rows if r["region"] in set(self.train_regions)]
        va = [r for r in rows if r["region"] in set(self.val_regions)]
        common = {r["tile_id"] for r in tr} & {r["tile_id"] for r in va}
        assert not common, f"tile leakage between splits: {common}"

        dataset_cls = PatchEmbeddingsDataset if self.source in PATCH_SOURCES else TilePairDataset
        kw = dict(root=self.root, source=self.source, patch_size=self.patch_size,
                  height_norm=self.height_norm)
        self.train_ds = dataset_cls(tr, is_train=True, **kw)
        self.val_ds = dataset_cls(va, is_train=False, **kw)
        return self

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)
