# Knowledge Base Index

Content catalog for the knowledge base. The agent reads this file first when answering queries or performing operations. Keep entries as one-liners: `- [Title](path) — description`.

## Papers

- [What on Earth is AlphaEarth?](wiki/papers/benavides_martinez_2026_alphaearth_hierarchical_structure.md) — Functional interpretability of AlphaEarth's 64 embedding dims for land cover; 2-12 dims recover 98% of accuracy per class
- [Inferring Height from Earth Embeddings](wiki/papers/hamoudzadeh_2026_inferring_height_alphaearth.md) — U-Net/U-Net++ on AlphaEarth embeddings for DSM height regression in France
- [TESSERA](wiki/papers/feng_2025_tessera_temporal_embeddings.md) — Temporal dual-modality foundation model; outperforms AlphaEarth on canopy height in tall tropical forest

## Concepts

- [Geospatial Foundation Models](wiki/concepts/geospatial_foundation_models.md) — Task-agnostic pretrained Earth-observation models
- [Embedding Interpretability](wiki/concepts/embedding_interpretability.md) — Physical-variable vs. functional interpretation of embedding dimensions
- [Label Efficiency](wiki/concepts/label_efficiency.md) — Accuracy achievable per labeled sample count
- [Canopy / Surface Height Estimation](wiki/concepts/canopy_height_estimation.md) — Canopy height vs. DSM/terrain height as distinct targets

## Methods

- [AlphaEarth Foundations (GAEF / GSE)](wiki/methods/alphaearth.md) — Google's 64-dim, 10m, annual global embedding model
- [TESSERA](wiki/methods/tessera.md) — Self-supervised dual-encoder temporal embedding model, 128-dim
- [Progressive Ablation](wiki/methods/progressive_ablation.md) — Dimension-ranking/pruning method via MDI + nested retraining
- [U-Net / U-Net++ for Height Regression](wiki/methods/unet_unetplusplus.md) — Spatially-aware architectures for embedding-based height prediction

## Datasets

- [ESA WorldCover 2020](wiki/datasets/esa_worldcover.md) — Global 10m, 11-class land cover reference
- [French DSM, Nouvelle-Aquitaine](wiki/datasets/french_dsm_nouvelle_aquitaine.md) — AlphaEarth + IGN RGE ALTI DSM, France
- [Danum Valley Canopy Height](wiki/datasets/danum_valley_canopy_height.md) — Airborne LiDAR canopy height, Borneo tropical forest

## Synthesis

- [Overview](wiki/overview.md) — Research landscape and project positioning
- [How Geospatial Embedding Models Are Being Used for Downstream Tasks](wiki/synthesis/geoembedding_downstream_tasks.md) — Cross-paper synthesis: frozen-embedding paradigm, label efficiency, and the fusion gap this project fills
