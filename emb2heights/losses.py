"""
Loss functions for embedding -> (landcover, height) training.

Currently plain MAE over all 4 output channels. The config's lambdas
[MAE, SSIM, Gradient, Tversky] anticipate a composite loss; add terms here
incrementally once the MAE baseline trains end to end.
"""
import torch.nn as nn


# To include other losses, add them here and update the config.
def build_loss(config):
    if config.loss_name == "mae":
        mae = nn.L1Loss()
        return mae

    
# 

