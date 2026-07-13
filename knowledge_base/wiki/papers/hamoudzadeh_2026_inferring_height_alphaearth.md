---
title: "Inferring Height from Earth Embeddings: First Insights Using Google AlphaEarth"
authors: [Alireza Hamoudzadeh, Valeria Belloni, Roberta Ravanelli]
year: 2026
venue: "arXiv preprint (2602.17250)"
source: "sources/papers/hamoudzadeh_2026_inferring_height_alphaearth.pdf"
tags: [alphaearth, height-prediction, dsm, deep-learning, france]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Summary
First systematic test of whether AlphaEarth's 64-band embeddings can drive deep-learning-based surface height mapping. Compares U-Net and U-Net++ (ResNet-18 encoders) against a Ridge regression baseline over Nouvelle-Aquitaine, France (~8,000 km²), using France's IGN RGE ALTI DSM as ground truth.

## Method
Inputs: AlphaEarth embeddings (2020, 10 m, all 64 bands). Target: IGN RGE ALTI DSM (native 5 m, resampled to 10 m). Train/test split ~70/30 (~5,550 km² / ~2,315 km²) within the same region. 512×512 patches, AdamW (lr 1e-3), MSE loss, early stopping. Ridge regression used as a linear, non-spatial baseline.

## Key Results
- Train: R²=0.97 for both U-Net and U-Net++; Ridge R²=0.64.
- Test: U-Net R²=0.78 (RMSE 19.26 m); U-Net++ R²=0.84 (RMSE 16.42 m, median diff -2.62 m) — better generalization than plain U-Net; Ridge R²=0.38 (RMSE 32.22 m) and produced physically implausible negative heights (~-80 to -90 m).
- Both networks systematically overestimate heights below 100 m and underestimate above 100 m.

## Limitations
- Test region skewed toward higher elevations (110-160 m) than training data, driving a real train/test distribution shift.
- Temporal mismatch: AlphaEarth is single-year (2020); DSM surveys span the 2010s to early 2020s, risking landscape-change artifacts.
- Single French region only — morphological/geographic diversity untested.
- Target is surface/terrain height (DSM, includes buildings and bare ground), not vegetation-specific canopy height.

## Relevance
Highly relevant as a same-country (France), same-embedding-source (AlphaEarth) height-regression baseline for this project, giving concrete single-embedding R²/RMSE numbers to compare a fusion approach against. Its documented distribution-shift sensitivity is also a direct precedent for this project's own caution (in the paper's impact statement) about claiming cross-region generalization from France-only training data. Note the target-variable mismatch: this paper predicts DSM/terrain height, not the canopy/building height this project targets — see [[canopy_height_estimation]] for the distinction.

## Related
- Concepts: [[geospatial_foundation_models]], [[canopy_height_estimation]]
- Methods: [[unet_unetplusplus]], [[alphaearth]]
- Datasets: [[french_dsm_nouvelle_aquitaine]]
