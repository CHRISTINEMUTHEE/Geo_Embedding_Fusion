---
title: "What on Earth is AlphaEarth? Hierarchical Structure and Functional Interpretability for Global Land Cover"
authors: [Iván Felipe Benavides-Martínez, Justin Guthrie, Jhon Edwin Arias, Yeison Alberto Garcés-Gómez, Angela Ines Guzman-Alvis, Cristiam Victoriano Portilla-Cabrera, Somnath Mondal, Andrew J. Allyn, Auroop R. Ganguly]
year: 2026
venue: "arXiv preprint (2603.16911)"
source: "sources/papers/benavides_martinez_2026_alphaearth_hierarchical_structure.pdf"
tags: [alphaearth, interpretability, land-cover, geospatial-foundation-models, embeddings]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Summary
Proposes a "functional interpretability" framework that reverse-engineers what each of AlphaEarth's 64 embedding dimensions contributes to land cover classification, via 130,000+ binary classification experiments plus progressive feature ablation. Finds dimensions fall on a specialist-to-generalist spectrum, and that 98% of full-embedding classification accuracy is recoverable using as few as 2-12 of the 64 dimensions, depending on class.

## Method
Uses ESA WorldCover 2020 labels (independent of AlphaEarth's own training data) against AlphaEarth's 64-dim embeddings. For each of 11 land cover classes, runs one-vs-rest binary classification with a randomly chosen algorithm (Random Forest, Gradient Boosted Trees, XGBoost, LightGBM), ranks dimensions by Mean Decrease in Impurity (MDI), then progressively retrains on the top-1 through top-30 dimensions to find the "tipping point" — the minimum subset recovering 98% of full-embedding performance. Dimensions are labeled specialist (1 class), or low/mid/high-generalist (2/3/4+ classes) based on how many classes' minimum subsets they appear in.

## Key Results
- 43 of 64 dimensions received a functional interpretation; 21 remained uninterpreted (not required by any class's minimum subset).
- Minimum dimensions for 98% baseline accuracy: Water 2, Mangroves 4, Built-up 5, Tree cover 6, Herbaceous wetland 7, Cropland 8, Bare/sparse & Grassland 10, Moss/lichen & Shrubland 12.
- Restricting inference to the minimum subset cuts classification time 20-80% depending on class.
- Specialist dimensions map to biophysically distinct classes (e.g. A64 → permanent water, A09/A35 → built-up); shared (generalist) dimensions plausibly represent ecotones/transition zones between classes.

## Limitations
- Importance rankings depend on the ML algorithm and MDI metric used.
- The 98%-recovery threshold is a deliberate choice (justified by WorldCover's own label uncertainty) but different thresholds would yield different subsets/classifications.
- Interpretations are task-dependent approximations, not validated against physical variables.
- Functional roles may not be geographically stable; classification accuracy itself varies by region.

## Relevance
Directly relevant to this project's embedding-fusion question: shows AlphaEarth's 64 dimensions are highly redundant for at least one of the project's two downstream tasks (land cover), meaning fusion strategies should account for per-dimension informativeness rather than treating embeddings as uniformly dense signal. Also establishes a precedent for asking *what a GFM embedding actually encodes* for a downstream task, which is the same interpretive question underlying this project's fusion-vs-single-embedding comparison.

## Related
- Concepts: [[geospatial_foundation_models]], [[embedding_interpretability]]
- Methods: [[progressive_ablation]], [[alphaearth]]
- Datasets: [[esa_worldcover]]
