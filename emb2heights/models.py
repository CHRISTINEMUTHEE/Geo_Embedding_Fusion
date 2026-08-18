"""
Models module for implementing models.

This file serves as a placeholder where you can:
1. Define a custom model.
2. Import and adapt models from popular libraries.
3. Create model factory functions.

Examples of libraries you might want to use:
- TorchGeo: https://torchgeo.readthedocs.io/en/stable/api/models.html
- Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch
- Torchvision: https://pytorch.org/vision/stable/models.html
- Timm: https://github.com/huggingface/pytorch-image-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. LIGHT UNET COMPONENTS
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
# Add an upsampling block
class UpsampleBlock(nn.Module):
    """ Bi linear Upsampling + Convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Architecture; Light version (32->64 ->128 ->256 ->128 ->64 ->32)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        # Upsampling Blocks and Convolutions
        self.up1 = UpsampleBlock(256, 128)
        self.conv1 = DoubleConv(256, 128)
        
        self.up2 = UpsampleBlock(128, 64)
        self.conv2 = DoubleConv(128, 64)
         
        self.up3 = UpsampleBlock(64, 32)
        self.conv3 = DoubleConv(64, 32)
        # Output Convolution
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)

        logits = self.outc(x)
        return logits

# ==========================================
# 2. Efficient Decoder COMPONENTS
# ==========================================
class StandardUpsampleBlock(nn.Module):
    '''Uses standard dense convolutions and GELU activation'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # standard 3*3 convolution(groups=1) + BatchNorm + GELU = M2GPU thrives here!
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class EfficientDecoder(nn.Module):
    '''Mempry efficient decoder for 16*16 -> 256*256 upsampling on M2 Max'''
    def __init__(self, n_channels, n_classes):
        super().__init__()
        # The Squeeze : 768 -> 256 at 16*16 resolution to prevent memory explosion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(n_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        #Progressive Upsampling with channels as resolution doubles
        self.up1 = StandardUpsampleBlock(256, 128)
        self.up2 = StandardUpsampleBlock(128, 64)
        self.up3 = StandardUpsampleBlock(64, 32)
        self.up4 = StandardUpsampleBlock(32, 16)
        # Prediction Head
        ## padding=0: a 1x1 conv must not grow 256x256 to 258x258 (MAE needs pred==target HW)
        self.head = nn.Conv2d(16, n_classes, kernel_size=1, padding=0)


    def forward(self, x):
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)
# REVIEW REQUIRED
## Factory: model choice from config, n_channels inferred from data by the caller.
def build_model(config, n_channels):
    if config.model_name == "lightunet":
        return LightUNet(n_channels, config.n_classes)
    elif config.model_name == "efficientdecoder":
        return EfficientDecoder(n_channels, config.n_classes)
    raise ValueError(f"Model {config.model_name} not implemented")

