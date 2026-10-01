import torch
from torch import nn


def conv1d_block(in_ch, filters, kernel, dropout):
    return nn.Sequential(
        nn.Conv1d(in_ch, filters, kernel),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Dropout(dropout),
    )


class HIKNet(nn.Module):
    def __init__(self, n_channels=6, n_timesteps=199, filters=150, kernel=15, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            conv1d_block(n_channels, filters, kernel, dropout),
            conv1d_block(filters, filters, kernel, dropout),
        )

    def forward(self, x):
        return self.features(x)
