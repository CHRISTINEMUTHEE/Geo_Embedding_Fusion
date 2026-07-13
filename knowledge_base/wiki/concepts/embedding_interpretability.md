---
title: "Embedding Interpretability"
tags: [interpretability, embeddings, explainable-ai]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Definition
The study of what information is encoded in a foundation model's high-dimensional embedding space, and how individual dimensions or subspaces relate to observable physical or categorical variables.

## Details
Two complementary framings appear in the literature: (1) **physical-variable interpretation** — associating embedding dimensions with continuous environmental variables like temperature or elevation; (2) **functional interpretation** — characterizing dimensions by their empirical contribution to a discrete downstream task (e.g. land cover classification), via progressive ablation, yielding a specialist-to-generalist functional spectrum. Motivated by criticism that GFMs like AlphaEarth are opaque "black-box" representations whose reliability for high-stakes, fine-scale decisions is constrained by limited interpretability.

## Papers
- [[benavides_martinez_2026_alphaearth_hierarchical_structure]] — introduces the functional interpretability framework and its specialist/generalist taxonomy for AlphaEarth

## Related Concepts
- [[geospatial_foundation_models]] — the class of models whose embeddings this concept studies
