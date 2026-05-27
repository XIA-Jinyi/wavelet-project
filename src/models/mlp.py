"""Feature MLP model for steganalysis."""
import torch.nn as nn


class FeatureMLP(nn.Module):
    """MLP classifier for hand-crafted wavelet features."""
    def __init__(self, input_dim=12, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
