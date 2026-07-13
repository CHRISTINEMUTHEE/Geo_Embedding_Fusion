---
title: "Label Efficiency"
tags: [label-efficiency, few-shot, foundation-models]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Definition
The capacity of a model or representation to reach a target level of downstream-task accuracy using as few labeled training examples as possible, treated as a first-class evaluation axis alongside raw accuracy.

## Details
Central selling point of GFMs generally — task-agnostic embeddings meant to "support learning with sparse labels." TESSERA quantifies this concretely: >90% of final burned-area-detection performance with ~330 labeled pixels (~0.01% of training data), and competitive biomass-regression accuracy with only 1% of labels. This is the same axis this project sweeps (label counts) to compare single vs. fused embeddings for France land cover/height prediction — see `paper/paper.tex` and `paper/research_questions.md`.

## Papers
- [[feng_2025_tessera_temporal_embeddings]] — reports concrete label-efficiency numbers across five benchmarks
- [[benavides_martinez_2026_alphaearth_hierarchical_structure]] — motivates GFMs' sparse-label promise; its dimension-pruning result is a related but distinct computational-efficiency finding, not a labeled-sample count

## Related Concepts
- [[geospatial_foundation_models]]
