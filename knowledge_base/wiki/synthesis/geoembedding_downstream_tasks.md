---
title: "How Geospatial Embedding Models Are Being Used for Downstream Tasks"
tags: [foundation-models, fusion, downstream-tasks, synthesis]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Current Understanding
Across the three ingested sources, a consistent pattern emerges: GFMs (AlphaEarth, TESSERA) are used as **frozen feature extractors** — downstream tasks train only a lightweight head (linear/MLP/Random Forest/small UNet) on precomputed embeddings, never fine-tuning the foundation model itself. This is exactly the design assumption behind this project's own fusion experiments (fuse frozen embeddings, then sweep label counts on top).

Downstream performance is not uniform across embedding sources or task types:
- AlphaEarth's 64 dimensions are individually informative but highly redundant for land cover — 2-12 dimensions recover 98% of full-embedding accuracy per class ([[benavides_martinez_2026_alphaearth_hierarchical_structure]]).
- The same embeddings transfer reasonably to DSM/terrain height regression in France with a spatially-aware head (U-Net++, test R²=0.84), though with real distribution-shift degradation ([[hamoudzadeh_2026_inferring_height_alphaearth]]).
- On canopy height specifically, a structurally different foundation model (TESSERA — temporal, dual-modality, self-supervised) outperforms AlphaEarth, particularly in tall/dense tropical canopy where AlphaEarth appears to saturate ([[feng_2025_tessera_temporal_embeddings]]).

Label efficiency is treated as a first-class metric, not an afterthought: TESSERA reports usable performance from <1% (in some tasks ~0.01%) of labels; AlphaEarth's own value proposition is framed the same way. This directly parallels this project's approach of sweeping label counts to trace label-efficiency curves for single vs. fused embeddings over France.

Interpretability and fusion look like two sides of the same question this project is asking: Benavides-Martinez et al.'s finding that AlphaEarth encodes non-uniform, sometimes-redundant, class-specific and shared/generalist information suggests fusing AlphaEarth with a structurally different embedding (like TESSERA) could add value less through raw extra dimensionality and more through covering specialist/generalist "gaps" AlphaEarth leaves for certain classes or height regimes.

## Open Questions
- Does AlphaEarth's saturation on tall canopy heights (observed in Danum Valley) also occur for the vegetation/building heights relevant to this project's France study area, and would fusing in a temporally-resolved embedding (TESSERA-like) specifically correct that regime rather than improving accuracy uniformly?
- Hamoudzadeh et al.'s DSM task and this project's height task differ (terrain/surface height vs. vegetation/building height, see [[canopy_height_estimation]]) — how much of AlphaEarth's demonstrated France performance is expected to carry over to this project's actual target, and is that the right single-embedding baseline to beat?
- None of these three sources test embedding fusion directly — all three either use one embedding source or compare foundation models head-to-head, not combined. This project's central question (does fusion help, and does it change label efficiency) has no direct precedent in this set of sources.

## Evidence
- Frozen-embedding + lightweight-head paradigm — supported by [[benavides_martinez_2026_alphaearth_hierarchical_structure]], [[hamoudzadeh_2026_inferring_height_alphaearth]], [[feng_2025_tessera_temporal_embeddings]]
- TESSERA outperforms AlphaEarth specifically on canopy height in tall tropical forest — supported by [[feng_2025_tessera_temporal_embeddings]] (R²=0.66 vs. AlphaEarth saturating; RMSE 8.88 m vs. global products >14 m)
- AlphaEarth embeddings transfer to France height regression with real but bounded distribution-shift degradation — supported by [[hamoudzadeh_2026_inferring_height_alphaearth]] (test R²=0.84, RMSE=16.42 m, U-Net++)
- AlphaEarth's 64 dimensions are functionally non-uniform and largely redundant per land-cover class — supported by [[benavides_martinez_2026_alphaearth_hierarchical_structure]]

## Gaps
- No source directly tests fused/combined embeddings for height or land cover prediction — this is the gap `paper/paper.tex` and `paper/research_questions.md` are trying to fill.
- No source tests the label-efficiency of a fused embedding relative to either single source.
- Canopy/vegetation height vs. terrain/DSM height are conflated across sources; this project should be explicit about which target it predicts and cite the matching baseline accordingly.
