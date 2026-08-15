# Modules

Core package: everything reusable across experiments. Plain PyTorch (no Lightning).

## Structure

- `config.py`: `ExperimentConfig` (pydantic) — all experiment knobs, derived output
  paths as properties, `load_config()` for YAML + CLI overrides
- `models.py`: model architectures (`LightUNet`) and the `build_model(config, n_channels)`
  factory; input channels are inferred from the data, not configured
- `datasets.py`: embedding/label file pairing (`find_file_pairs`),
  `PixelEmbeddingsDataset` (1:1 pixel embeddings like AlphaEarth/Tessera), and
  `build_dataloaders(config)`. **Known flaws** — it splits tiles at random (leaking
  across regions) and does not dequantize the int8 embeddings; kept as-is so the
  `01_alphaearth_lightunet` baseline stays reproducible. Prefer `datamodule.py`.
- `datamodule.py`: `Embed2HeightsDataModule` — region-grouped train/val split driven by
  `manifest.csv`, plus `TilePairDataset`, which dequantizes embeddings (`/127`) and masks
  the `-128` nodata sentinel. Use this for new work
- `losses.py`: `build_loss(config)` — currently MAE; composite terms (SSIM,
  gradient, Tversky) to be added incrementally
- `trainers.py`: `train(config)` — the training/validation loop, challenge-style
  metrics (IoU, masked height RMSE), loss curves, and result visualizations
- `cli.py`: `python -m emb2heights.cli --config <yaml>`

## Extending

- New model: add a class in `models.py`, a branch in `build_model`, and a value in
  `config.ModelNameEnum`.
- New dataset type (e.g. 16x16 patch embeddings like TerraMind): add a Dataset class
  in `datasets.py` and select it in `build_dataloaders`.
- New embedding source for the subset loader: add it to `SOURCES` in
  `scripts/acquire_subset.py` and pass `source=` to `Embed2HeightsDataModule`.
- New loss: implement in `losses.py` and return it from `build_loss`.

Outputs of every run land in `outputs/<experiment_name>/`: `best_model.pth`,
`last_model.pth`, `config.yaml` snapshot, `loss_curve.png`, `visualizations/`.
