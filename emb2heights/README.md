# Modules

Core package: everything reusable across experiments. Plain PyTorch (no Lightning).

## Structure

- `config.py`: `ExperimentConfig` (pydantic) — all experiment knobs, derived output
  paths as properties, `load_config()` for YAML + CLI overrides
- `models.py`: model architectures (`LightUNet` for pixel-aligned sources, `EfficientDecoder`
  for 16x16 patch-token sources) and the `build_model(config, n_channels)` factory; input
  channels are inferred from the data, not configured
- `datasets.py`: embedding/label file pairing (`find_file_pairs`),
  `PixelEmbeddingsDataset` (1:1 pixel embeddings like AlphaEarth/Tessera), and
  `build_dataloaders(config)`. **Known flaws** — it splits tiles at random (leaking
  across regions) and does not dequantize the int8 embeddings; kept as-is so the
  `01_alphaearth_lightunet` baseline stays reproducible. Prefer `datamodule.py`.
- `datamodule.py`: `Embed2HeightsDataModule` — region-grouped train/val split driven by
  `manifest.csv`. `TilePairDataset` handles pixel-aligned sources (AlphaEarth, Tessera):
  quantization is detected from the data itself (integer-valued, bounded to +/-127), not
  assumed, and nodata comes from the file's own metadata (`rasterio`'s `nodata`, whatever
  sentinel a given source actually uses — verified against real data as `-128`, `nan`, `0`,
  and "none set" across the six sources). `LatentTokenDataset` (subclass) handles patch-token
  sources (TerraMind/THOR, e.g. 16x16x768): kept at native resolution, no upsampling —
  cropping happens at two scales (`scale_factor`, default 16), so image and target come back
  at *different* resolutions on purpose, for a model that decodes tokens itself (the
  latent-based fusion `paper/research_questions.md` describes for those two sources, not
  LightUNet). `PATCH_SOURCES` selects which class a given `source=` name gets.
  `max_train_tiles` caps *training* tiles only (val stays full) via a fixed-seed shuffle-
  then-slice, so smaller budgets are nested subsets of larger ones — the mechanism a real
  label-efficiency sweep needs (see `scripts/label_efficiency_sweep.py`). Use this module
  for new work
- `losses.py`: `build_loss(config)` — `"mae"` (plain L1, all 4 channels equal) or
  `"weighted"` (`HeightLandcoverLoss`: height (channel 3, primary) and landcover
  (channels 0-2, auxiliary) weighted independently via `w_height`/`w_landcover` —
  `w_landcover=0.0` is the RQ3 ablation. Landcover uses `WeightedL1Loss` (`bg_weight`)
  since building/water are real-data-imbalanced (~1-3% mean coverage). Soft
  Tversky/Dice was tried and rejected: verified numerically to be miscalibrated for
  continuous fractional targets — see the module docstring before reintroducing it
- `trainers.py`: `train(config)` — the training/validation loop, challenge-style
  metrics (IoU, masked height RMSE), loss curves, and result visualizations.
  `build_train_val_loaders(config)` picks the loader: `datamodule.py` when
  `config.data_root` is set, otherwise the legacy `datasets.py` split.
  `evaluate_metrics()` reports both `mae_height`/`rmse_height` (unmasked, every pixel —
  the headline RQ1/RQ2 number) and `rmse_building`/`rmse_vegetation` (masked to where
  that class is present — diagnostic). `plot_height_rmse_vs_epoch` is a training-progress
  curve (one run, epoch by epoch) — not a label-efficiency curve; that needs separate
  runs at different `max_train_tiles`, compared by best accuracy (see
  `scripts/label_efficiency_sweep.py`)
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
