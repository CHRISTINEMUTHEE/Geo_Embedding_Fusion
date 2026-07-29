"""Shared fixtures: synthetic embedding/label GeoTIFF pairs and a ready config."""
import numpy as np
import pytest
import rasterio

from emb2heights.config import ExperimentConfig

N_EMB_CHANNELS = 8   # small stand-in for AlphaEarth's 64
TILE = 64            # small stand-in for the real 256x256 tiles


def write_tif(path, array):
    c, h, w = array.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=c, dtype="float32") as dst:
        dst.write(array.astype(np.float32))


@pytest.fixture
def data_dirs(tmp_path):
    """6 (embedding, label) pairs mimicking the real naming scheme."""
    emb_dir = tmp_path / "emb"
    lab_dir = tmp_path / "labels"
    emb_dir.mkdir()
    lab_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(6):
        emb = rng.normal(size=(N_EMB_CHANNELS, TILE, TILE))
        ## label: 3 fraction channels in [0,1] + height channel in meters [0,25]
        lab = np.concatenate([rng.uniform(0, 1, (3, TILE, TILE)),
                              rng.uniform(0, 25, (1, TILE, TILE))])
        write_tif(emb_dir / f"gee_emb_{i:04d}_BE.tif", emb)
        write_tif(lab_dir / f"label_{i:04d}_BE_2023.tif", lab)
    return str(emb_dir), str(lab_dir)


@pytest.fixture
def config(data_dirs, tmp_path):
    emb_dir, lab_dir = data_dirs
    return ExperimentConfig(
        experiment_name="test_exp",
        base_dir=str(tmp_path / "outputs"),
        train_embeddings_dir=emb_dir,
        train_targets_dir=lab_dir,
        batch_size=2,
        patch_size=32,
        num_workers=0,
        epochs=1,
        val_split=0.34,  # 2 of 6 pairs
    )
