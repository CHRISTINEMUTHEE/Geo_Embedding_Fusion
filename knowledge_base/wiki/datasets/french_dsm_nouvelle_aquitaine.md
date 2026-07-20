---
title: "French DSM, Nouvelle-Aquitaine (IGN RGE ALTI + AlphaEarth)"
tags: [dsm, france, height-prediction, benchmark]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Overview
A regional surface-height reference dataset combining AlphaEarth embeddings (2020, 10 m, 64-band) with France's IGN RGE ALTI Digital Surface Model (DSM, natively 5 m, resampled to 10 m) over the Nouvelle-Aquitaine region (~8,000 km²).

## Statistics
- ~5,550 km² train/validation, ~2,315 km² test (~70/30 split)
- DSM surveys compiled across the 2010s to early 2020s (temporal mismatch with the single-year 2020 AlphaEarth embeddings)
- Test region skewed toward higher elevations (110-160 m) than the training region

## Known Issues
Temporal mismatch between embeddings and DSM surveys can introduce landscape-change artifacts; the DSM includes buildings and terrain, not vegetation-specific canopy height (see [[canopy_height_estimation]]); limited morphological diversity from being a single French region contributed to the observed train/test distribution shift.

## Access
AlphaEarth embeddings from Google; DSM from IGN (Institut national de l'information géographique et forestière), France's national mapping agency.

## Papers
- [[hamoudzadeh_2026_inferring_height_alphaearth]] — introduces this train/test setup for AlphaEarth-based height regression in France
