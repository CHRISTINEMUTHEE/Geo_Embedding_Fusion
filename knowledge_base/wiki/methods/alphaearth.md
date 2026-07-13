---
title: "AlphaEarth Foundations (GAEF / GSE)"
tags: [alphaearth, foundation-model, embeddings]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Overview
Google AlphaEarth Foundations — a task-agnostic geospatial foundation model producing eight annual 64-dimensional embeddings at 10 m resolution (2017-2024) by fusing optical (Sentinel-2, Landsat), radar (Sentinel-1), climate (ERA5), and LiDAR (GEDI) data into a single unified, unit-hypersphere-normalized representation per pixel.

## Details
Introduced by Brown et al. 2025 (cited across all three ingested papers, sometimes as "GSE" — Google Satellite Embeddings). Used as a frozen feature source: downstream tasks train a lightweight supervised head (classifier or regressor) on top of the embeddings without fine-tuning AlphaEarth itself.
- Benavides-Martinez et al. show its 64 dimensions are highly redundant for land cover classification (2-12 dims recover 98% of full-embedding accuracy per class) and organize into a specialist/generalist functional hierarchy.
- Hamoudzadeh et al. use all 64 dimensions directly as input to U-Net/U-Net++ for DSM height regression in France, achieving test R²=0.84 with the better architecture.
- Feng et al. (TESSERA paper) use it as the primary foundation-model baseline ("GSE"), reporting it saturates at high canopy heights in tall tropical forest, where TESSERA does not.

## Strengths and Weaknesses
- Strengths: broad multi-modal fusion at global 10 m scale; strong sparse-label downstream performance; consistent global basis (unit hypersphere normalization).
- Weaknesses: opaque/black-box latent space with limited out-of-the-box physical interpretability; annually-composited rather than temporally-resolved; saturates on some tall/dense-canopy height regimes.

## Papers
- [[benavides_martinez_2026_alphaearth_hierarchical_structure]] — interpretability analysis of its embedding space
- [[hamoudzadeh_2026_inferring_height_alphaearth]] — sole embedding source for height regression in France
- [[feng_2025_tessera_temporal_embeddings]] — main foundation-model baseline ("GSE") that TESSERA is compared against

## Related
- Concepts: [[geospatial_foundation_models]], [[embedding_interpretability]]
- Methods: [[tessera]]
