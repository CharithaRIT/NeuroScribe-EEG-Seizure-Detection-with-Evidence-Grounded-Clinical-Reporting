"""
cnn_gru.py
----------
Baseline 1 model: 3-layer 1D CNN + 2-layer Bidirectional GRU seizure classifier.

Architecture:
    Input  : (batch, n_channels, window_samples)
    CNN    : 3 x (Conv1d → BatchNorm → ReLU → MaxPool)
    GRU    : 2-layer bidirectional GRU, last time-step
    Output : (batch,) logits  →  sigmoid → seizure probability

Typical usage:
    from src.models.cnn_gru import CNNGRUClassifier
    model = CNNGRUClassifier(n_channels=23, dropout=0.3)
"""

import torch.nn as nn


class CNNBlock(nn.Module):
    """One CNN stage: Conv1d → BatchNorm → ReLU → MaxPool2."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.block(x)


class CNNGRUClassifier(nn.Module):
    """
    3-layer 1D CNN + 2-layer Bidirectional GRU seizure detector.

    Args:
        n_channels: Number of EEG channels (default 23 for CHB-MIT bipolar).
        dropout:    Dropout rate applied before the final FC layer.
    """
    def __init__(self, n_channels: int = 23, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            CNNBlock(n_channels, 64,  kernel=7),   # (B, 64,  T/2)
            CNNBlock(64,        128,  kernel=5),   # (B, 128, T/4)
            CNNBlock(128,       256,  kernel=3),   # (B, 256, T/8)
        )
        self.gru = nn.GRU(
            input_size=256, hidden_size=256,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(256 * 2, 1)   # bidirectional → 512

    def forward(self, x):
        x = self.cnn(x)              # (B, 256, T/8)
        x = x.permute(0, 2, 1)      # (B, T/8, 256) — time-first for GRU
        x, _ = self.gru(x)          # (B, T/8, 512)
        x = self.dropout(x[:, -1])  # last time step → (B, 512)
        return self.fc(x).squeeze(1) # (B,) logits
