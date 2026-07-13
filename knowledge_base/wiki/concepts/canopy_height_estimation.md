---
title: "Canopy / Surface Height Estimation"
tags: [height-prediction, canopy-height, dsm, remote-sensing]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Definition
The downstream task of predicting height at pixel level from remote-sensing inputs, typically supervised by airborne LiDAR or a digital surface model (DSM).

## Details
Two distinct variants appear in the ingested sources and should not be conflated:
1. **Canopy height** — height of vegetation/tree crowns above ground, validated against airborne LiDAR (e.g. TESSERA's Danum Valley, Borneo benchmark).
2. **Surface/terrain height (DSM)** — includes buildings and bare ground, not vegetation-specific (Hamoudzadeh et al.'s French IGN RGE ALTI DSM target).

This project's own height-prediction task (see `paper/paper.tex`, `paper/research_questions.md`) should specify explicitly which of these two targets it predicts, since the two published baselines below are not directly comparable to each other.

## Papers
- [[feng_2025_tessera_temporal_embeddings]] — canopy height, Danum Valley LiDAR benchmark; TESSERA outperforms AlphaEarth and global canopy products
- [[hamoudzadeh_2026_inferring_height_alphaearth]] — DSM/terrain height, France; single-embedding (AlphaEarth) baseline

## Related Concepts
- [[label_efficiency]]
- [[geospatial_foundation_models]]
