# Knowledge Base Log

Append-only chronological record of all knowledge base operations. Never edit retroactively.

Format: `## [YYYY-MM-DD] operation | description` followed by files touched.

Operations: `ingest`, `query`, `lint`, `manual`.

<!-- Entries below this line -->

## [2026-07-13] manual | added 3 sources to sources/papers/, not yet ingested
- sources/papers/benavides_martinez_2026_alphaearth_hierarchical_structure.pdf — AlphaEarth embedding dimension interpretability (arXiv 2603.16911)
- sources/papers/hamoudzadeh_2026_inferring_height_alphaearth.pdf — height inference from AlphaEarth embeddings (arXiv 2602.17250)
- sources/papers/feng_2025_tessera_temporal_embeddings.pdf — TESSERA, reports 12.2m vs 16.1m RMSE canopy height regression outperforming AlphaEarth on Borneo dataset (arXiv 2506.20380)

## [2026-07-13] ingest | benavides_martinez_2026, hamoudzadeh_2026, feng_2025_tessera
Created:
- wiki/papers/benavides_martinez_2026_alphaearth_hierarchical_structure.md
- wiki/papers/hamoudzadeh_2026_inferring_height_alphaearth.md
- wiki/papers/feng_2025_tessera_temporal_embeddings.md
- wiki/concepts/geospatial_foundation_models.md
- wiki/concepts/embedding_interpretability.md
- wiki/concepts/label_efficiency.md
- wiki/concepts/canopy_height_estimation.md
- wiki/methods/alphaearth.md
- wiki/methods/tessera.md
- wiki/methods/progressive_ablation.md
- wiki/methods/unet_unetplusplus.md
- wiki/datasets/esa_worldcover.md
- wiki/datasets/french_dsm_nouvelle_aquitaine.md
- wiki/datasets/danum_valley_canopy_height.md
- wiki/synthesis/geoembedding_downstream_tasks.md (synthesis: how GFM embeddings are used for downstream tasks, and the fusion/label-efficiency gap this project fills)
Updated: index.md (Papers, Concepts, Methods, Datasets, Synthesis sections)
