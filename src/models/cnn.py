"""Pure CNN model -- operates on raw 512x512 grayscale images."""
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


class PureCNN(nn.Module):
    """Single-channel CNN for cover/stego classification on raw pixels.

    Input: (N, 1, 512, 512) -- raw grayscale images standardised per-image.
    Architecture identical to StegoCNN but with in_channels=1.
    """
    def __init__(self, in_channels=1, drop=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16),   # 512 -> 256
            ConvBlock(16, 32),             # 256 -> 128
            ConvBlock(32, 64),             # 128 -> 64
            ConvBlock(64, 128),            # 64 -> 32
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
