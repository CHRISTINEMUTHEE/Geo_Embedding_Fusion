import numpy as np
import pytest
import torch

from emb2heights.datasets import (
    PixelEmbeddingsDataset,
    _normalize_core_id,
    build_dataloaders,
    find_embedding_files,
    find_file_pairs,
)
from tests.conftest import N_EMB_CHANNELS, write_tif


## The pairing logic hinges on both naming styles reducing to the same core ID.
@pytest.mark.parametrize("filename,expected", [
    ("gee_emb_0000_BE.tif", "0000_BE"),
    ("tessera_emb_0000_BE.tif", "0000_BE"),
    ("label_0000_BE_2023.tif", "0000_BE"),
    ("emb_3001_BE_2023_quantized.tif", "3001_BE"),
    ("s2_0007_FR_merged.tif", "0007_FR"),
])
def test_normalize_core_id(filename, expected):
    assert _normalize_core_id(filename) == expected


def test_normalize_core_id_keep_year():
    assert _normalize_core_id("label_0000_BE_2023.tif", keep_year=True) == "0000_BE_2023"


def test_find_file_pairs_matches_all(data_dirs):
    emb_dir, lab_dir = data_dirs
    pairs = find_file_pairs(emb_dir, lab_dir)
    assert len(pairs) == 6
    for emb_path, lab_path in pairs:
        assert _normalize_core_id(emb_path) == _normalize_core_id(lab_path)


def test_find_file_pairs_ignores_unmatched(data_dirs, tmp_path):
    emb_dir, lab_dir = data_dirs
    write_tif(f"{emb_dir}/gee_emb_9999_XX.tif", np.zeros((1, 8, 8)))  # no label
    pairs = find_file_pairs(emb_dir, lab_dir)
    assert len(pairs) == 6


def test_dataset_shapes_and_height_normalization(data_dirs):
    emb_dir, lab_dir = data_dirs
    pairs = find_file_pairs(emb_dir, lab_dir)
    ds = PixelEmbeddingsDataset(pairs, patch_size=32, is_train=False, height_norm=30.0)
    image, target = ds[0]
    assert image.shape == (N_EMB_CHANNELS, 32, 32)
    assert target.shape == (4, 32, 32)
    ## fractions untouched, height divided by 30 and clipped at 1.5
    assert target[:3].min() >= 0 and target[:3].max() <= 1
    assert target[3].max() <= 1.5
    assert target[3].max() < 1.0  # raw heights <=25m / 30 < 1, so clip never hit here


def test_dataset_pads_small_tiles(tmp_path):
    write_tif(tmp_path / "gee_emb_0000_BE.tif", np.zeros((2, 20, 20)))
    write_tif(tmp_path / "label_0000_BE_2023.tif", np.zeros((4, 20, 20)))
    pairs = [(str(tmp_path / "gee_emb_0000_BE.tif"), str(tmp_path / "label_0000_BE_2023.tif"))]
    image, target = PixelEmbeddingsDataset(pairs, patch_size=32)[0]
    assert image.shape == (2, 32, 32)
    assert target.shape == (4, 32, 32)


def test_dataset_without_labels(data_dirs):
    emb_dir, _ = data_dirs
    pairs = find_embedding_files(emb_dir)
    image, target = PixelEmbeddingsDataset(pairs, patch_size=32, is_train=False)[0]
    assert image.shape == (N_EMB_CHANNELS, 32, 32)
    assert target.numel() == 0


def test_val_crop_is_deterministic_train_crop_is_not(data_dirs):
    emb_dir, lab_dir = data_dirs
    pairs = find_file_pairs(emb_dir, lab_dir)
    val_ds = PixelEmbeddingsDataset(pairs, patch_size=32, is_train=False)
    assert torch.equal(val_ds[0][0], val_ds[0][0])


def test_build_dataloaders_split_and_batching(config):
    train_loader, val_loader = build_dataloaders(config)
    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 2
    images, targets = next(iter(train_loader))
    assert images.shape == (2, N_EMB_CHANNELS, 32, 32)
    assert targets.shape == (2, 4, 32, 32)


def test_build_dataloaders_empty_dir_raises(config, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    config = config.model_copy(update={"train_embeddings_dir": str(empty)})
    with pytest.raises(ValueError, match="No \\(embedding, label\\) pairs"):
        build_dataloaders(config)
