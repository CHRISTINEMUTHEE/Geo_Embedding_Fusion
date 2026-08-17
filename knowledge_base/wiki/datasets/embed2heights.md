---
title: "embed2heights (ESA Φ-lab challenge dataset)"
tags: [dataset, alphaearth, tessera, terramind, thor, ndsm, land-cover, challenge]
date_created: 2026-08-15
date_updated: 2026-08-15
---

## Overview
The AI-ready dataset this project actually trains on: pre-computed embeddings from four geospatial
foundation models paired with LiDAR-derived reference labels, published for the ESA Φ-lab
*embed2heights* challenge. The task is multi-task — segment building/vegetation/water as sub-pixel
fractions and regress above-ground height from an nDSM.

Labels derive from IGN airborne LiDAR over France; the training split is described as major French
cities plus selected rural areas.

## Statistics
- Train: 2,024 tiles. Test: 946 tiles (labels withheld — challenge holdout).
- Tiles: 256×256 at 10 m. **Not all tiles are 256×256** — 255×256 and a third variant also occur.
- Total: 147.8 GB across 19,846 files.
- Inputs, per tile:
  - AlphaEarth (`alphaearth_emb`): 64 channels, 33.93 GB train / 5.69 GB test — see [[alphaearth]]
  - TESSERA (`tessera_emb`): 128 channels, 67.82 GB train / 27.78 GB test — see [[tessera]]
  - TerraMind and THOR (S1 + S2): **16×16×768 patch-level**, not 1:1 with labels, ~10.5 GB total
- Labels (`labels`), 4 bands: `[0]` building %, `[1]` vegetation %, `[2]` water %, `[3]` nDSM height (m).
- 168 region codes; median 11 tiles per region, max 55. Train and test share 165 of 168 regions.

## Known Issues
These were verified directly against the published files, and they drive the loading code in
`emb2heights/datamodule.py`:

- **No georeferencing at all.** Every file has `crs=None` and an identity transform, and all
  19,846 STAC `bbox` entries in `catalog.v1.parquet` are zero. Region codes (`BE`, `KE`, `OG`,
  `QI`, …) are anonymized two-letter labels, not country codes. Coordinates are unrecoverable, so
  TorchGeo's `GeoSampler` / `RasterDataset` cannot be used, and no true spatial split is possible.
- **Tile IDs do not encode adjacency.** Edge-continuity between consecutive-ID tiles scores 0.585
  vs 0.522 for random pairs — no signal. Tiles within a region are scattered samples, not a grid.
  The region code is therefore the only usable grouping for leakage-free validation.
- **Embeddings are int8-quantized but stored as float32**: integer values within ±127 and a
  `-128` nodata sentinel. They must be divided by 127 to recover the ~[-1, 1] range, and `-128`
  must be masked *before* arithmetic. Naive `nan_to_num` maps NaN to 0.0, which collides with a
  legitimate embedding value.
- **Tile quality varies sharply.** Some tiles are almost entirely nodata (`gee_emb_1431_KE.tif` is
  98.4% NaN across all 64 bands) while most are 100% valid. In a 6-region, 80-tile sample, 4 tiles
  fell below a 50% valid-pixel threshold.
- **Two fully degenerate tiles**: `gee_emb_1087_QP.tif` (every pixel −128) and
  `label_1582_LB_2024.tif` (every pixel 0).
- **Temporal spread**: labels carry years 2021–2024, and embeddings are annual, so year alignment
  differs per tile. TESSERA's published coverage is 2024-only with earlier years backfilling.

## Access
- Official: EOTDL dataset `embed2heights` — <https://www.eotdl.com/datasets/embed2heights>.
  Requires a login; `scripts/acquire.py` reads the staged STAC catalog and pulls selected dirs.
- Public mirror (no login, MIT): Hugging Face `troni21/esa_philab_embed2heights`, carrying the
  full tree plus `catalog.v1.parquet`. `scripts/acquire_subset.py` uses this to pull a
  region-balanced <1% subset (~1.4 GB) for local development.
- Baseline repository: <https://github.com/VMarsocci/emb2heights-baselines>

## Evaluation
Challenge metric weights: mIoU buildings 25%, trees 15%, water 15%; RMSE building height 25%,
vegetation height 20%. This is why `emb2heights/trainers.py` reports per-class IoU plus masked
height RMSE over building and vegetation pixels separately.

## Papers
- [[hamoudzadeh_2026_inferring_height_alphaearth]] — the closest published setup: AlphaEarth → IGN
  DSM height regression over France, though on a georeferenced DSM rather than these tiles.
- [[feng_2025_tessera_temporal_embeddings]] — source of the TESSERA embeddings included here.
- Related concepts: [[canopy_height_estimation]], [[label_efficiency]],
  [[geoembedding_downstream_tasks]].
