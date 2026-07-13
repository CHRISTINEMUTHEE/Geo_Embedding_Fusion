---
title: "TESSERA: Precomputed FAIR Global Pixel Embeddings for Earth Representation and Analysis"
authors: [Zhengpeng Feng, Clement Atzberger, Sadiq Jaffer, Jovana Knezevic, Silja Sormunen, Robin Young, Madeline C. Lisaius, Markus Immitzer, Toby Jackson, James Ball, David A. Coomes, Anil Madhavapeddy, Andrew Blake, Srinivasan Keshav]
year: 2025
venue: "arXiv preprint (2506.20380)"
source: "sources/papers/feng_2025_tessera_temporal_embeddings.pdf"
tags: [tessera, alphaearth, canopy-height, foundation-model, label-efficiency, fusion]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Summary
Introduces TESSERA, a self-supervised foundation model producing precomputed, FAIR-distributed 128-dimensional pixel embeddings at 10 m global resolution by encoding full annual Sentinel-1/Sentinel-2 time series rather than single composites, preserving temporal/phenological signal. Matches or outperforms AlphaEarth ("GSE") and other foundation models across five downstream benchmarks, notably outperforming it on canopy height regression in tall tropical forest.

## Method
Dual-encoder architecture: separate 4-block transformer encoders for Sentinel-1 (VV/VH) and Sentinel-2 (10 bands), each with day-of-year positional encoding and attention pooling, fused via an MLP into a 128-dim vector per pixel. Trained self-supervised (modified Barlow Twins + mixup) on ~800M pixel-annual-time-series (2017-2024, 3,012 global MGRS tiles) with global batch shuffling to break spatial autocorrelation. Downstream tasks use lightweight frozen-embedding heads (MLP, Random Forest, or small UNet) — no fine-tuning of TESSERA itself.

## Key Results
- **Canopy height (Danum Valley, Borneo, vs. airborne LiDAR):** TESSERA R²=0.66, RMSE=8.88±0.98 m, bias=-0.62 m — far ahead of global canopy products GLAD/ETH/Meta (R²<0.05, RMSE>14 m), and outperforms AlphaEarth/GSE head-to-head with an identical UNet architecture. GSE was observed to saturate at lower canopy heights in this tall tropical forest.
- **Crop classification (Austria, INVEKOS):** TESSERA+MLP beats GSE and PRESTO across all label regimes (1-30% labeled data).
- **Biomass (Finland, BioMassters):** TESSERA ≈ GSE ≈ competition winner using only 1% labels.
- **Burned area (California):** TESSERA F1>0.96, clearly ahead of GSE; reaches >90% of final performance with ~330 labeled pixels (~0.01% of data).

## Limitations
- Sentinel-1 (SAR) branch contributes only marginally in ablations; optical data dominates.
- Global embedding coverage currently limited to 2024; 2017-2023 backfill in progress.
- Performance degrades at extreme biomass values.
- All comparisons use lightweight frozen-embedding heads — no full fine-tuning ceiling is reported.

## Relevance
This is the paper that answers where TESSERA outperforms AlphaEarth at tree canopy height: Danum Valley, Borneo, R²=0.66/RMSE=8.88m vs. AlphaEarth saturating in tall canopy. TESSERA's architecture — temporal, dual-modality, self-supervised — is structurally different from AlphaEarth, making it a natural fusion candidate for this project: AlphaEarth's own dimensions are known to be redundant for land cover (see [[benavides_martinez_2026_alphaearth_hierarchical_structure]]), while TESSERA covers a specific regime (tall canopy height) where AlphaEarth is documented to saturate. TESSERA's label-efficiency results are also a direct precedent for the label-count sweeps this project runs to compare single vs. fused embeddings.

## Related
- Concepts: [[geospatial_foundation_models]], [[label_efficiency]], [[canopy_height_estimation]]
- Methods: [[tessera]], [[alphaearth]]
- Datasets: [[danum_valley_canopy_height]]
