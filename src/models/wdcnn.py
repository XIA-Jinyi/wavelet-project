"""WDCNN model — wavelet-domain CNN for steganalysis."""
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class StegoCNN(nn.Module):
    """4-channel wavelet-domain CNN for cover/stego classification.

    Input: (N, 4, H/2, W/2) — standardised LL, LH, HL, HH subbands.
    """
    def __init__(self, in_channels=4, drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16),   #  256 → 128
            ConvBlock(16, 32),             #  128 → 64
            ConvBlock(32, 64),             #  64 → 32
            ConvBlock(64, 128),            #  32 → 16
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
