---
title: "Progressive Ablation (Embedding Dimension Ranking)"
tags: [feature-importance, ablation, interpretability]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Overview
A model-agnostic procedure for ranking and pruning input feature dimensions by iteratively retraining a model on nested subsets of features ordered by an importance score. Used to quantify how redundant AlphaEarth's 64-dimensional embeddings are for land cover classification.

## Details
For each land cover class and experiment, a model is first trained on all 64 embedding dimensions and Mean Decrease in Impurity (MDI) ranks the dimensions by importance. Models are then retrained on the top-1, top-2, ..., top-30 dimensions. The "tipping point" for a class is the smallest dimension subset whose mean performance reaches 98% of the full 64-dimension baseline; dimensions within that subset are "associated" with the class, and the number of classes a dimension is associated with sets its functional label (specialist = 1 class; low/mid/high-generalist = 2/3/4+ classes).

## Strengths and Weaknesses
- Strengths: performance-grounded rather than an arbitrary importance-value cutoff; surfaces both redundancy and cross-class functional structure from one procedure.
- Weaknesses: results depend on the importance metric (MDI) and the ML algorithm used to compute it; sensitive to the chosen recovery threshold (98%); doesn't test whether pruned subsets transfer regionally.

## Papers
- [[benavides_martinez_2026_alphaearth_hierarchical_structure]] — introduces and applies this procedure to AlphaEarth embeddings

## Related
- Concepts: [[embedding_interpretability]]
- Methods: [[alphaearth]]
