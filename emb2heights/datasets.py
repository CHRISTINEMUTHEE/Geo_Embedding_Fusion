"""
Datasets and dataloaders for embedding -> (landcover, height) training.

Pipeline: find_file_pairs matches embedding tifs to label tifs by core ID,
PixelEmbeddingsDataset loads/crops them, build_dataloaders wires everything
from an ExperimentConfig.
"""

import glob
import os
import random
import re

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


## Strip prefixes/suffixes so 'gee_emb_0000_BE' and 'label_0000_BE_2023' both -> '0000_BE'
def _normalize_core_id(filename, keep_year=False):
    base = os.path.splitext(os.path.basename(filename))[0]

    if base.startswith('label_'):
        base = base[len('label_'):]

    # Longest/most-specific prefixes first
    for prefix in ['gee_emb_', 'tessera_emb_', 's2_', 's1_', 'emb_']:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    for suffix in ['_embedding', '_embeddings', '_merged', '_quantized']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]

    # Year suffix (e.g. '_2023') is stripped so year-less embeddings match dated labels
    if not keep_year:
        base = re.sub(r'_\d{4}$', '', base)

    return base


# REVIEW REQUIRED
## O(N) matching via a {core_id: label_path} lookup, robust to naming differences.
def find_file_pairs(emb_dir, tar_dir):
    emb_files = glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True)
    label_files = glob.glob(os.path.join(tar_dir, "**", "label_*.tif"), recursive=True)

    label_map = {_normalize_core_id(l): l for l in label_files}

    pairs = []
    for e_path in emb_files:
        norm_id = _normalize_core_id(e_path)
        if norm_id in label_map:
            pairs.append((e_path, label_map[norm_id]))
    return pairs


def find_embedding_files(emb_dir):
    """(emb_path, None) pairs for label-free inference on the held-out test set."""
    emb_files = sorted(glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True))
    return [(e_path, None) for e_path in emb_files]


# ---------------------------------------------------------
# DATASET 1: Pixel-based embeddings (AlphaEarth, Tessera)
# 1:1 spatial resolution (e.g. 256x256 -> 256x256)
# ---------------------------------------------------------
# REVIEW REQUIRED
## KNOWN FLAW: embeddings are int8-quantized (values +/-127, nodata -128) and are NOT
## dequantized here, and nan_to_num maps NaN to 0.0 -- indistinguishable from a real
## embedding value. Kept for baseline reproducibility; datamodule.py fixes both.
class PixelEmbeddingsDataset(Dataset):
    def __init__(self, file_pairs, patch_size=128, is_train=True, height_norm=30.0):
        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.is_train = is_train
        self.height_norm = height_norm

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        with rasterio.open(emb_path) as src:
            image = src.read().astype(np.float32)
        image = np.nan_to_num(image)

        has_target = tar_path is not None
        if has_target:
            with rasterio.open(tar_path) as src:
                target = src.read().astype(np.float32)
            target = np.nan_to_num(target)
            ## Height channel (index 3) -> roughly [0, 1]; clip allows 1.5x outliers
            target[3] = np.clip(target[3] / self.height_norm, 0.0, 1.5)

        # Reflect-pad tiles smaller than the patch size
        c, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            if has_target:
                target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            h, w = image.shape[1], image.shape[2]

        ## Train: random crop (augmentation). Val/test: center crop (deterministic).
        if self.is_train:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        image = image[:, top:top + self.patch_size, left:left + self.patch_size]
        image_t = torch.from_numpy(image)

        if not has_target:
            return image_t, torch.empty(0)

        target = target[:, top:top + self.patch_size, left:left + self.patch_size]
        return image_t, torch.from_numpy(target)


# REVIEW REQUIRED
## Config -> (train_loader, val_loader). Replaces the old Lightning DataModule.
## KNOWN FLAW: the split is random over tiles, so tiles from the same region land in both
## train and val (geographic leakage). Use Embed2HeightsDataModule for a region-grouped split.
def build_dataloaders(config):
    pairs = find_file_pairs(config.train_embeddings_dir, config.train_targets_dir)
    if len(pairs) == 0:
        raise ValueError(
            f"No (embedding, label) pairs found in {config.train_embeddings_dir} "
            f"and {config.train_targets_dir}. Check the directories."
        )

    rng = random.Random(config.random_seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * config.val_split))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    train_ds = PixelEmbeddingsDataset(
        train_pairs, patch_size=config.patch_size, is_train=True,
        height_norm=config.height_normalization_constant,
    )
    val_ds = PixelEmbeddingsDataset(
        val_pairs, patch_size=config.patch_size, is_train=False,
        height_norm=config.height_normalization_constant,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers)
    print(f"Found {len(pairs)} pairs -> {len(train_pairs)} train / {len(val_pairs)} val")
    return train_loader, val_loader
