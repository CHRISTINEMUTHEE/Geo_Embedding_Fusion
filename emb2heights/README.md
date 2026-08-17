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
  `manifest.csv`. `TilePairDataset` handles pixel-aligned sources (AlphaEarth, Tessera):
  quantization is detected from the data itself (integer-valued, bounded to +/-127), not
  assumed, and nodata comes from the file's own metadata. `PatchEmbeddingsDataset` (subclass)
  handles patch-token sources (TerraMind/THOR, e.g. 16x16x768) by nearest-neighbor
  upsampling to the label's pixel grid before cropping — a data-loader-level baseline, not
  the latent-based fusion `paper/research_questions.md` describes for those two sources.
  `PATCH_SOURCES` selects which class a given `source=` name gets. Use this module for new
  work
- `losses.py`: `build_loss(config)` — currently MAE; composite terms (SSIM,
  gradient, Tversky) to be added incrementally
- `trainers.py`: `train(config)` — the training/validation loop, challenge-style
  metrics (IoU, masked height RMSE), loss curves, and result visualizations.
  `build_train_val_loaders(config)` picks the loader: `datamodule.py` when
  `config.data_root` is set, otherwise the legacy `datasets.py` split
- `cli.py`: `python -m emb2heights.cli --config <yaml>`

## Extending

- New model: add a class in `models.py`, a branch in `build_model`, and a value in
  `config.ModelNameEnum`.
- New pixel-aligned embedding source (like AlphaEarth/Tessera): add it to `SOURCES` in
  `scripts/acquire_subset.py` and pass `source=` to `Embed2HeightsDataModule` — no new
  Dataset class needed, `TilePairDataset` handles any pixel-aligned source.
- New patch-token embedding source (like TerraMind/THOR): add it to both `SOURCES` in
  `scripts/acquire_subset.py` and `PATCH_SOURCES` in `datamodule.py`.
- New loss: implement in `losses.py` and return it from `build_loss`.

Outputs of every run land in `outputs/<experiment_name>/`: `best_model.pth`,
`last_model.pth`, `config.yaml` snapshot, `loss_curve.png`, `visualizations/`.
