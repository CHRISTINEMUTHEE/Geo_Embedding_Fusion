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
# 2. Efficient Encoder Decoder COMPONENTS
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

class ConvBlock(nn.Module):
    '''Conv3x3 + BN + GELU -- EfficientEncoderDecoder's basic feature block (GELU, not
    LightUNet's ReLU, to match the rest of this module's existing convention).'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class EfficientEncoderDecoder(nn.Module):
    '''Encoder-decoder for 16x16 patch-token embeddings (TerraMind/THOR), two regimes
    stitched together:

    1. A real encoder-decoder over the native 16x16 grid (16->8->4->8->16, with
       skip connections) -- legitimate multi-scale feature fusion across
       neighbouring tokens, since those intermediate 8x8/4x4 feature maps genuinely
       exist inside the network, unlike anything finer than 16x16.
    2. Blind progressive upsampling 16->32->64->128->256 -- unavoidable for *any*
       architecture, since no embedding finer than the native 16x16 grid exists to
       skip from. This is patch-token sources' real, irreducible resolution
       ceiling; no architecture change removes it.

    Parameter count is matched to LightUNet's (~2.16M) so a "patch-token sources
    underperform" result can't be attributed to this network simply being smaller
    or weaker than the one pixel-aligned sources get -- was ~590K before this
    widening (5-stage blind squeeze-and-upsample, no encoder at all -- the
    original "EfficientDecoder" name no longer fit once an encoder was added).
    '''
    def __init__(self, n_channels, n_classes):
        super().__init__()
        ## Native-grid squeeze: n_channels -> 128 at 16x16. 1x1 (not 3x3) -- a 3x3 here
        ## costs 9x more per parameter for the same effect (no neighbours to mix yet,
        ## enc1 right after does that), and this line item alone would otherwise be
        ## the single biggest cost in the network at n_channels=768.
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        ## --- Regime 1: real encoder-decoder over the native 16x16 grid ---
        self.enc1 = ConvBlock(128, 128)                # 16x16
        self.pool1 = nn.MaxPool2d(2)                   # -> 8x8
        self.enc2 = ConvBlock(128, 128)                # 8x8
        self.pool2 = nn.MaxPool2d(2)                   # -> 4x4
        self.bottleneck = ConvBlock(128, 128)          # 4x4 -- deepest point, all 256 tokens fused

        self.dec2_up = StandardUpsampleBlock(128, 128)  # 4x4 -> 8x8
        self.dec2 = ConvBlock(256, 128)                 # concat with enc2 skip
        self.dec1_up = StandardUpsampleBlock(128, 128)  # 8x8 -> 16x16
        self.dec1 = ConvBlock(256, 128)                 # concat with enc1 skip

        ## --- Regime 2: blind upsampling past the native resolution (no skips possible) ---
        self.up1 = StandardUpsampleBlock(128, 128)  # 16 -> 32
        self.up2 = StandardUpsampleBlock(128, 64)   # 32 -> 64
        self.up3 = StandardUpsampleBlock(64, 32)    # 64 -> 128
        self.up4 = StandardUpsampleBlock(32, 32)    # 128 -> 256
        # Prediction Head
        ## padding=0: a 1x1 conv must not grow 256x256 to 258x258 (MAE needs pred==target HW)
        self.head = nn.Conv2d(32, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.stem(x)                              # 16x16, 128ch

        e1 = self.enc1(x)                              # 16x16
        e2 = self.enc2(self.pool1(e1))                 # 8x8
        b = self.bottleneck(self.pool2(e2))             # 4x4

        d2 = self.dec2_up(b)                           # 8x8
        d2 = self.dec2(torch.cat([d2, e2], dim=1))     # 8x8
        d1 = self.dec1_up(d2)                          # 16x16
        d1 = self.dec1(torch.cat([d1, e1], dim=1))     # 16x16

        x = self.up1(d1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.head(x)
# REVIEW REQUIRED
## Factory: model choice from config, n_channels inferred from data by the caller.
def build_model(config, n_channels):
    if config.model_name == "lightunet":
        return LightUNet(n_channels, config.n_classes)
    elif config.model_name == "efficientencoderdecoder":
        return EfficientEncoderDecoder(n_channels, config.n_classes)
    raise ValueError(f"Model {config.model_name} not implemented")

