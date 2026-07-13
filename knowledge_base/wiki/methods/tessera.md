---
title: "TESSERA"
tags: [tessera, foundation-model, self-supervised, embeddings]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Overview
A self-supervised geospatial foundation model that produces precomputed, FAIR-distributed 128-dimensional pixel embeddings at global 10 m resolution by encoding full annual Sentinel-1/Sentinel-2 time series — rather than single composites — preserving phenological/temporal signal.

## Details
Dual-encoder architecture: separate 4-block transformer encoders for SAR (Sentinel-1 VV/VH) and optical (Sentinel-2, 10 bands) inputs, each with day-of-year positional encoding and attention pooling over the time series, fused via an MLP into one 128-dim vector per pixel. Trained self-supervised with a modified Barlow Twins objective plus mixup regularization on ~800M pixel-time-series samples (2017-2024, 3,012 global MGRS tiles), using global batch shuffling to break spatial autocorrelation. Downstream tasks use lightweight heads (MLP, Random Forest, or a small UNet) trained on the frozen embeddings — TESSERA itself is not fine-tuned.

## Strengths and Weaknesses
- Strengths: preserves temporal/phenological information lost by compositing; strong label efficiency (competitive results from <1% of labels on several benchmarks); outperforms AlphaEarth/GSE on canopy height regression in tall tropical forest, where GSE saturates; fully precomputed and FAIR-distributed, so end users never run the model themselves.
- Weaknesses: Sentinel-1 (SAR) branch contributes only marginally in ablations (optical dominates); global embedding coverage currently limited to 2024, with 2017-2023 backfill in progress; evaluated via lightweight heads only, so the ceiling with heavier task-specific fine-tuning is untested in the source paper.

## Papers
- [[feng_2025_tessera_temporal_embeddings]] — introduces the model and all benchmark comparisons, including the canopy height result vs. AlphaEarth

## Related
- Concepts: [[geospatial_foundation_models]], [[label_efficiency]], [[canopy_height_estimation]]
- Methods: [[alphaearth]]
