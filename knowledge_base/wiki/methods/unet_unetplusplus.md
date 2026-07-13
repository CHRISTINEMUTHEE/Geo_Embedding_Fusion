---
title: "U-Net / U-Net++ for Embedding-Based Height Regression"
tags: [unet, deep-learning, height-prediction]
date_created: 2026-07-13
date_updated: 2026-07-13
---

## Overview
Convolutional encoder-decoder architectures adapted from image segmentation to pixel-wise height regression: U-Net, and its nested-skip-connection variant U-Net++, used to predict surface height from AlphaEarth embedding patches.

## Details
Both use a ResNet-18 encoder pretrained on ImageNet. U-Net++ adds nested, dense skip connections between encoder and decoder stages, intended to better preserve multi-scale spatial context than plain U-Net. Trained with AdamW (lr 1e-3, weight decay 1e-4), MSE loss, on 512×512 embedding patches over Nouvelle-Aquitaine, France, against IGN RGE ALTI DSM reference.

## Strengths and Weaknesses
- Strengths: spatially-aware architectures capture transferable topographic patterns from embeddings far better than a pixel-wise linear baseline; U-Net++'s nested skips measurably improved robustness to train/test distribution shift (test R²=0.84 vs. 0.78 for plain U-Net; RMSE 16.42 m vs. 19.26 m).
- Weaknesses: both still degraded substantially out-of-distribution (test RMSE ~16-19 m vs. train RMSE ~7-8 m); systematic bias (over/under-estimation) around the 100 m height mark in both architectures.

## Papers
- [[hamoudzadeh_2026_inferring_height_alphaearth]] — introduces this comparison for AlphaEarth-based height regression in France

## Related
- Methods: [[alphaearth]]
- Datasets: [[french_dsm_nouvelle_aquitaine]]
