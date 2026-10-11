"""
Loss functions for embedding -> (landcover, height) training.

Two independently-weighted terms: height (channel 3, primary target -- RQ1/RQ2)
and landcover (channels 0-2, auxiliary head -- RQ3). Splitting them means
w_landcover=0 is the literal RQ3 ablation, everything else held fixed.

Landcover channels are continuous sub-pixel fractions, not discrete classes
(verified against real data: 200-500+ unique values per tile, <10% of nonzero
building pixels are exactly 1.0). Building/water are also heavily imbalanced
(~1-3% mean pixel coverage, ~70-75% of tiles near-zero).

Two class-imbalance tools were considered and rejected before WeightedL1Loss:
- Focal loss: cross-entropy-based, needs a discrete class probability. Using it
  here would mean thresholding away the sub-pixel percentages this data encodes.
- Soft Tversky/Dice: verified numerically to be miscalibrated for continuous
  regression targets. For a fixed fractional target (e.g. 0.3), loss(pred) is
  *monotonically decreasing* all the way to pred=1.0 -- it is never minimized at
  pred=target. This holds for both a fully continuous version and a
  threshold-target/continuous-prediction variant, and for symmetric (Dice,
  alpha=beta) as well as recall-biased (beta>alpha) weightings. It's a general
  property of overlap-of-positive-mass losses: correctly calibrated for binary
  presence, not for matching a continuous fraction. Using it as a training loss
  here would push the landcover head to over-predict toward 1.0 wherever any
  land cover is present.

If bg_weight-weighted L1 alone isn't enough once real IoU numbers are in, the
correct escalation is magnitude-aware reweighting (weight each pixel by how
rare *that target value* is, not just zero-vs-nonzero) -- that stays a proper
regression loss, unlike Dice/Tversky. Not implemented yet.
"""
import torch
import torch.nn as nn


class WeightedL1Loss(nn.Module):
    """Per-pixel L1, reweighted so nonzero-label (foreground) pixels count more
    than exactly-zero (background) ones. Presence-based, not full inverse-frequency
    -- simplest fix for the measured building/water imbalance. Minimum is always at
    pred=target regardless of the weight, unlike Tversky/Dice (see module docstring)."""

    def __init__(self, bg_weight=0.05):
        super().__init__()
        self.bg_weight = bg_weight

    def forward(self, pred, target):
        err = (pred - target).abs()
        weight = torch.where(target > 0, torch.ones_like(target),
                             torch.full_like(target, self.bg_weight))
        return (err * weight).sum() / weight.sum().clamp_min(1e-6)


# REVIEW REQUIRED
class HeightLandcoverLoss(nn.Module):
    """total = w_height * L1(height) + w_landcover * weighted-L1(landcover)"""

    def __init__(self, w_height=1.0, w_landcover=1.0, bg_weight=0.05):
        super().__init__()
        self.w_height = w_height
        self.w_landcover = w_landcover
        self.height_loss = nn.L1Loss()
        self.landcover_loss = WeightedL1Loss(bg_weight=bg_weight)

    def forward(self, pred, target):
        height_l = self.height_loss(pred[:, 3:4], target[:, 3:4])
        lc_l = self.landcover_loss(pred[:, :3], target[:, :3])
        return self.w_height * height_l + self.w_landcover * lc_l


def build_loss(config):
    if config.loss_name == "mae":
        return nn.L1Loss()
    if config.loss_name == "weighted":
        return HeightLandcoverLoss(w_height=config.w_height, w_landcover=config.w_landcover,
                                   bg_weight=config.bg_weight)
    raise ValueError(f"Unsupported loss_name: {config.loss_name}")
