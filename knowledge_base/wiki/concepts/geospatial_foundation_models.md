---
title: "Geospatial Foundation Models (GFMs)"
tags: [foundation-models, remote-sensing, embeddings]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Definition
Models pretrained on large volumes of multi-source Earth observation data (optical, radar, climate, LiDAR, etc.) to produce general-purpose, task-agnostic embeddings of the Earth's surface that transfer to many downstream tasks with few labels.

## Details
Framed as "virtual satellites" that characterize Earth's surface at high detail. GFMs address two classic remote-sensing bottlenecks: converting heterogeneous multi-source data into usable information, and the poor generalization of task-specific models trained on scarce high-quality labels. Examples spanning the ingested sources: AlphaEarth Foundations, TESSERA, and (named in related work) Prithvi, Copernicus-FM, Galileo, SMARTIES, CROMA. Downstream use in all three ingested papers follows the same pattern: freeze the foundation model, train only a lightweight head (classifier/regressor) on its embeddings.

## Papers
- [[benavides_martinez_2026_alphaearth_hierarchical_structure]] — interpretability analysis of one GFM's (AlphaEarth) embedding space
- [[hamoudzadeh_2026_inferring_height_alphaearth]] — uses a GFM's embeddings directly for height regression
- [[feng_2025_tessera_temporal_embeddings]] — introduces a new GFM and benchmarks it against others

## Related Concepts
- [[label_efficiency]] — GFMs' core value proposition is enabling label-efficient downstream learning
- [[embedding_interpretability]] — GFMs' latent embeddings raise interpretability challenges
