# Modules

Core package: everything reusable across experiments. Plain PyTorch (no Lightning).

## Structure

- `config.py`: `ExperimentConfig` (pydantic) — all experiment knobs, derived output
  paths as properties, `load_config()` for YAML + CLI overrides
- `models.py`: model architectures (`LightUNet` for pixel-aligned sources, `EfficientEncoderDecoder`
  for 16x16 patch-token sources) and the `build_model(config, n_channels)` factory; input
  channels are inferred from the data, not configured. `EfficientEncoderDecoder` is two regimes
  stitched together: a real encoder-decoder over the native 16x16 grid (16->8->4->8->16,
  with skip connections -- legitimate, since those intermediate feature maps genuinely
  exist), then blind progressive upsampling 16->32->64->128->256 (unavoidable for any
  architecture, since no embedding finer than 16x16 exists to skip from -- patch-token
  sources' real resolution ceiling, not fixable by decoder design). Parameter count
  (~1.68M) is matched to `LightUNet`'s (~2.16M) so a "patch-token sources underperform"
  result can't be attributed to this network simply being smaller or weaker -- it was
  ~590K (a bare squeeze + 4 plain upsamples, no encoder at all -- the original
  "EfficientDecoder" name no longer fit once an encoder was added) before this widening
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
  for new work. `split_regions()` buckets regions by whether they carry building/water
  above `stratify_threshold` before the val_frac split, so a rare class can't be dropped
  entirely into one side by an unlucky shuffle (falls back to a plain random split when
  the manifest lacks `<cls>_frac` columns — every region lands in one bucket, same as
  before). `Embed2HeightsDataModule.train_dataloader()` also oversamples tiles with
  building/water present via `class_balance_boost` (a `WeightedRandomSampler`, not a loss
  change — see `losses.py`'s `bg_weight` for that lever) and exposes `class_distribution`
  (per-split mean coverage and %-tiles-present per class — see
  `scripts/report_class_distribution.py`) and `height_distribution` (per-split %% of
  nonzero-height pixels in `HEIGHT_BIN_NAMES` = `low`/`medium`/`high`, fixed meter
  edges `HEIGHT_BIN_EDGES = (3.0, 15.0)` so bins are comparable across every
  source/split — chosen from the real data's percentiles for a roughly balanced
  39/41/19%% split rather than data-driven quantiles, which would shift per split).
  `TilePairDataset`/`LatentTokenDataset` accept
  `band_mean`/`band_std` for per-channel standardization; `Embed2HeightsDataModule` loads
  them from `<data_root>/band_stats.json` automatically when `standardize_bands=True`
  (default) — see `scripts/compute_band_stats.py`. Falls back to raw values with a
  printed warning if that file doesn't exist yet
- `losses.py`: `build_loss(config)` — `"mae"` (plain L1, all 4 channels equal) or
  `"weighted"` (`HeightLandcoverLoss`: height (channel 3, primary) and landcover
  (channels 0-2, auxiliary) weighted independently via `w_height`/`w_landcover` —
  `w_landcover=0.0` is the RQ3 ablation. Landcover uses `WeightedL1Loss` (`bg_weight`)
  since building/water are real-data-imbalanced (~1-3% mean coverage). Soft
  Tversky/Dice was tried and rejected: verified numerically to be miscalibrated for
  continuous fractional targets — see the module docstring before reintroducing it
- `trainers.py`: `train(config, save_artifacts=True)` — the training/validation loop,
  challenge-style metrics (IoU, masked height RMSE), loss curves, and result
  visualizations. `save_artifacts=False` skips every disk write and the
  `visualize_results()` forward passes, returning only the metrics dict — used by
  `label_efficiency_sweep.py`, which doesn't need N full per-run artifact sets.
  `build_train_val_loaders(config)` picks the loader: `datamodule.py` when
  `config.data_root` is set, otherwise the legacy `datasets.py` split.
  `evaluate_metrics()` reports both `mae_height`/`rmse_height` (unmasked, every pixel —
  the headline RQ1/RQ2 number) and `rmse_building`/`rmse_vegetation` (masked to where
  that class is present — diagnostic). IoU (`iou_building`/`iou_vegetation`/`iou_water`)
  is pooled as raw intersection/union counts across the *whole* validation set and
  divided once — not computed per-batch and averaged, which is batch-size-sensitive
  (verified: same val tiles, same predictions, `batch_size` 4 vs 23 alone swung
  `iou_building` 0.28→0.36). `binary_iou_from_channel` still returns a single-call ratio
  (useful standalone, e.g. in tests) but `evaluate_metrics` uses `_mask_counts()` directly
  to accumulate instead. `iou_threshold` and `height_mask_threshold` are separate config
  fields — they used to be one shared `threshold`, so changing one silently changed the
  other. `height_bin_accuracy`/`height_bin_f1_macro`/`height_bin_f1_{low,medium,high}`/
  `height_bin_confusion` treat height as a 3-way classification problem
  (`HEIGHT_BIN_NAMES`, see `datamodule.py`): both true and predicted height are binned,
  then a `[true_bin, pred_bin]` confusion matrix is pooled across the whole val set the
  same way IoU is (raw counts, not a per-batch average) before computing accuracy and
  per-class F1 directly from that pooled matrix (verified against `sklearn.metrics.f1_score`
  — not used directly since it needs raw y_true/y_pred arrays, not pooled counts). Only
  defined where `true_height > 0` — bare ground isn't a height class. Printed every
  epoch during training and by `scripts/evaluate_sources.py` (confusion matrix as a
  JSON string in its CSV, since a nested list isn't a CSV scalar), and logged to WandB
  (`val/height_bin_*` scalars each epoch, final confusion matrix as both a `wandb.Table`
  (raw counts) and a `wandb.Image` heatmap (`plot_height_bin_confusion`, row-normalized
  color = per-class recall, annotated with raw counts) saved locally too, at
  `config.height_confusion_path`).
  `train()` writes `class_distribution.json`
  (`config.class_distribution_path`) next to `config.yaml` with both the landcover
  class distribution and the height-bin distribution, and (`config.use_wandb=True`,
  needs `WANDB_API_KEY` in the environment — see `_init_wandb`) logs the same data plus
  every per-epoch metric to WandB (`entity`/`project` from `config.wandb_entity`/
  `config.wandb_project`) — off by default so tests and local runs never touch the
  network. Every other local artifact `train()` writes is also mirrored to WandB when
  `use_wandb=True` (only if `save_artifacts=True` too, since these are re-logged from
  the saved files, not regenerated): `loss_curve.png` → `train/loss_curve`,
  `height_rmse_vs_epochs.png` → `val/height_rmse_curve`, the sample true-vs-predicted
  maps from `visualize_results()` → `val/sample_visualizations` (one `wandb.Image` per
  sample), and `class_distribution.json` itself → uploaded raw via `wandb.save()`
  (visible/downloadable from the run's Files tab, not just as the flattened `data/*`
  scalars). `height_distribution()` collects true (and, given a model, predicted) height
  values over nonzero-height validation pixels across the *whole* loader (mean/median
  need pooled values, not a per-batch average) — used by `scripts/evaluate_sources.py`
  for a shared `baseline` row plus each source's own predicted mean/median, to catch
  systematic bias a middling RMSE alone can hide. `plot_height_rmse_vs_epoch` is a training-progress
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
`last_model.pth`, `config.yaml` snapshot, `class_distribution.json`, `loss_curve.png`,
`height_rmse_vs_epochs.png`, `height_bin_confusion.png`, `visualizations/`.
