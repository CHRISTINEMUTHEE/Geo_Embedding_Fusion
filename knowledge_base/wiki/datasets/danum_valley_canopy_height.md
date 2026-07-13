---
title: "Danum Valley Canopy Height (Borneo)"
tags: [canopy-height, lidar, benchmark, tropical-forest]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Overview
An airborne-LiDAR-derived canopy height reference over a 5×6 km old-growth tropical forest plot in Danum Valley, Borneo, used as ground truth for evaluating pixel-embedding-based canopy height regression.

## Statistics
- 5×6 km old-growth tropical forest plot
- Four-fold spatial cross-validation, 12 independent runs reported for TESSERA's evaluation
- Canopy heights in this tall tropical forest are high enough that AlphaEarth/GSE embeddings were observed to saturate

## Known Issues
Single-site benchmark (one tropical forest plot) — generalization of relative model rankings (TESSERA > AlphaEarth/GSE > global products) to other biomes or forest types is untested within the source paper.

## Access
Airborne LiDAR survey of Danum Valley; used alongside three existing global canopy height products (GLAD, ETH, Meta) and a regional model (Lang et al.) as additional baselines.

## Papers
- [[feng_2025_tessera_temporal_embeddings]] — TESSERA (R²=0.66, RMSE=8.88 m) substantially outperforms global canopy products (R²<0.05, RMSE>14 m) and AlphaEarth/GSE head-to-head with an identical UNet
