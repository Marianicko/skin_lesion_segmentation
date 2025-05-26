import torch.nn as nn
from config import Config
from unet import UNet


class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.NUM_CLASSES,
            bilinear=Config.BILINEAR,
        )

    def forward(self, x):
        return self.model(x)