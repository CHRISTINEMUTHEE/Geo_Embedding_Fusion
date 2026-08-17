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

## 2026-08-15 — ingest (manual)

Added `wiki/datasets/embed2heights.md`: the ESA Φ-lab challenge dataset the code actually
downloads, previously undocumented in the KB.

Facts verified directly against the published files (not from the dataset card):
- No georeferencing anywhere — `crs=None`, identity transforms, all 19,846 STAC bboxes zero;
  region codes are anonymized. TorchGeo `GeoSampler` is therefore unusable and no true spatial
  split exists.
- Tile IDs do not encode adjacency (edge continuity 0.585 consecutive vs 0.522 random), so the
  region code is the only leakage-free grouping for train/val.
- Embeddings are int8-quantized in float32 (±127, nodata −128) and need `/127` plus explicit
  nodata masking.
- Degenerate tiles found: `gee_emb_1087_QP.tif` (all −128), `label_1582_LB_2024.tif` (all 0);
  tile validity varies from 1.6% to 100%.
- Public MIT mirror on Hugging Face (`troni21/esa_philab_embed2heights`) removes the EOTDL login
  requirement.

Updated: index.md (Datasets section)
